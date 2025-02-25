from typing import List, Optional, Callable

import torch
import torch.nn.functional as F

from config import RunConfig

from constants import OUT_INDEX, OUT2_INDEX, Q1_EDGES, Q1_IMAGE, Q2_EDGES, Q2_IMAGE, STYLE_INDEX, STYLE2_INDEX

from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline
from utils import attention_utils
from utils.adain import masked_adain, adain
from utils.model_utils import get_stable_diffusion_model
from utils.segmentation import Segmentor
from utils.image_utils import resize_mask
import matplotlib.pyplot as plt


class AppearanceTransferModel:

    def __init__(self, config: RunConfig, pipe: Optional[CrossImageAttentionStableDiffusionPipeline] = None):
        self.config = config
        self.pipe = get_stable_diffusion_model() if pipe is None else pipe
        self.register_attention_control()
        self.segmentor = Segmentor(prompt=config.prompt, object_nouns=[config.object_noun])
        self.latents_q1_im = None
        self.latents_q2_im = None
        self.latents_q1_edges = None
        self.latents_q2_edges = None
        self.latents_kv =None
        self.latents_kv2 = None

        self.zs_q1_im = None
        self.zs_q2_im = None
        self.zs_q1_edges = None
        self.zs_q2_edges = None
        self.zs_kv = None
        self.zs_kv2 = None

        self.image_query1_mask_32, self.image_query1_mask_64 = None, None
        self.image_query2_mask_32, self.image_query2_mask_64 = None, None
        self.image_key_mask_32, self.image_key_mask_64 = None, None
        self.image_value_mask_32, self.image_value_mask_64 = None, None

        self.enable_edit = False
        self.step = 0
        self.query_mask_32, self.query_mask_64 = None, None
        self.value_mask_32, self.value_mask_64 = None, None
        self.key_mask_32, self.key_mask_64 = None, None
        self.attn_mask = None

    def set_latents(self, latents: List[torch.tensor]):
        
        self.latents_q1_im = latents[0]
        self.latents_q2_im = latents[1]
        self.latents_q1_edges = latents[2]
        self.latents_q2_edges = latents[3]
        self.latents_kv = latents[4]
        self.latents_kv2 = latents[5]

    def set_noise(self, noises:List[torch.tensor]):

        self.zs_q1_im = noises[0]
        self.zs_q2_im = noises[1]
        self.zs_q1_edges = noises[2]
        self.zs_q2_edges = noises[3]
        self.zs_kv = noises[4]
        self.zs_kv2 = noises[5]

    def set_masks(self, masks: List[torch.Tensor]):
        self.image_app_mask_32, self.image_struct_mask_32, self.image_app_mask_64, self.image_struct_mask_64 = masks 

    def get_adain_callback(self):

        def callback(st: int, timestep: int, latents: torch.FloatTensor) -> Callable:
            self.step = st
            # Compute the masks using prompt mixing self-segmentation and use the masks for AdaIN operation
            if self.config.use_masked_adain and self.step == self.config.adain_range.start:
                masks = self.segmentor.get_object_masks()
                self.set_masks(masks)
            # Apply AdaIN operation using the computed masks
            if self.config.adain_range.start <= self.step < self.config.adain_range.end:
                if self.config.use_masked_adain:
                    latents[OUT_INDEX] = masked_adain(latents[OUT_INDEX], latents[STYLE_INDEX], self.image_struct_mask_64, self.image_app_mask_64)
                    latents[OUT2_INDEX] = masked_adain(latents[OUT2_INDEX], latents[STYLE2_INDEX], self.image_struct_mask_64, self.image_app_mask_64)

                else:
                    latents[OUT_INDEX] = adain(latents[OUT_INDEX], latents[STYLE_INDEX])
                    latents[OUT2_INDEX] = adain(latents[OUT2_INDEX], latents[STYLE2_INDEX])


        return callback

    def register_attention_control(self):

        model_self = self

        class AttentionProcessor:

            def __init__(self, place_in_unet: str):
                self.place_in_unet = place_in_unet
                if not hasattr(F, "scaled_dot_product_attention"):
                    raise ImportError("AttnProcessor2_0 requires torch 2.0, to use it, please upgrade torch to 2.0.")

            def __call__(self,
                         attn,
                         hidden_states: torch.Tensor,
                         encoder_hidden_states: Optional[torch.Tensor] = None,
                         attention_mask=None,
                         temb=None,
                         perform_swap: bool = False):

                residual = hidden_states

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = attn.to_q(hidden_states)

                is_cross = encoder_hidden_states is not None
                if not is_cross:
                    encoder_hidden_states = hidden_states

                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                should_mix = False

                # Potentially apply our cross image attention operation
                # To do so, we need to be in a self-attention alyer in the decoder part of the denoising network
                if perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit:
                    if attention_utils.should_mix_keys_and_values(model_self, hidden_states):
                        should_mix = True

                        if model_self.step % 5 == 0 and model_self.step < 40 and model_self.config.overlay_type is None:
                            # Inject the structure's keys and values

                            key[OUT_INDEX] = key[Q1_EDGES]
                            value[OUT_INDEX] = value[Q1_EDGES]
                            query[OUT_INDEX] = query[Q1_EDGES]

                            key[OUT2_INDEX] = key[Q2_EDGES]
                            value[OUT2_INDEX] = value[Q2_EDGES]
                            query[OUT2_INDEX] = query[Q2_EDGES]

                        else:
                            # Inject the appearance's keys and values
                            key[OUT_INDEX] = key[STYLE_INDEX]
                            value[OUT_INDEX] = value[STYLE_INDEX]

                            key[OUT2_INDEX] = key[STYLE2_INDEX]
                            value[OUT2_INDEX] = value[STYLE2_INDEX]

                            if model_self.config.overlay_type is None:
                                # query[OUT_INDEX] = query[OUT_INDEX] + query[OUT_INDEX]*fg_q_mask
                                query[OUT_INDEX] = (2*query[OUT_INDEX] +query[Q1_IMAGE])/2
                                query[OUT2_INDEX] = (2*query[OUT2_INDEX] +query[Q2_IMAGE])/2


                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                
                # Compute the cross attention and apply our contrasting operation
                hidden_states, attn_weight = attention_utils.compute_scaled_dot_product_attention(
                    query, key, value,
                    model_self.config.attn_output,
                    model_self.step,
                    model_self.config.list_tokens,
                    model_self.config.overlay_type,
                    model_self.attn_mask,
                    edit_map=perform_swap and model_self.enable_edit and should_mix,
                    is_cross=is_cross,
                    contrast_strength=model_self.config.contrast_strength,
                )
                
                # Update attention map for segmentation
                if model_self.config.use_masked_adain and model_self.step == model_self.config.adain_range.start - 1:
                    model_self.segmentor.update_attention(attn_weight, is_cross)

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query[OUT_INDEX].dtype)

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor

                return hidden_states

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'ResnetBlock2D':
                pass
            if net_.__class__.__name__ == 'Attention':
                net_.set_processor(AttentionProcessor(place_in_unet + f"_{count + 1}"))
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = self.pipe.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")

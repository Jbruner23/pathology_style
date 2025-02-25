import math
import torch
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

# matplotlib.use('TkAgg')

from constants import OUT_INDEX, OUT2_INDEX, STYLE_INDEX, Q1_IMAGE, Q1_EDGES, Q2_IMAGE, Q2_EDGES


def should_mix_keys_and_values(model, hidden_states: torch.Tensor) -> bool:
    """ Verify whether we should perform the mixing in the current timestep. """
    is_in_32_timestep_range = (
            model.config.cross_attn_32_range.start <= model.step < model.config.cross_attn_32_range.end
    )
    is_in_64_timestep_range = (
            model.config.cross_attn_64_range.start <= model.step < model.config.cross_attn_64_range.end
    )
    is_hidden_states_32_square = (hidden_states.shape[1] == 32 ** 2)
    is_hidden_states_64_square = (hidden_states.shape[1] == 64 ** 2)
    should_mix = (is_in_32_timestep_range and is_hidden_states_32_square) or \
                 (is_in_64_timestep_range and is_hidden_states_64_square)
    return should_mix


def compute_scaled_dot_product_attention(Q, K, V,output, timestep ,list_tokens,overlay_type, attn_mask,
                                         edit_map=False, is_cross=False, contrast_strength=1.0):
    """ Compute the scale dot product attention, potentially with our contrasting operation. """
    if edit_map and not is_cross:

        attn_weight = (Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) 
        attn_weight = torch.softmax(attn_weight, dim=-1)

        attn_weight[OUT_INDEX] = torch.stack([
            torch.clip(enhance_tensor(attn_weight[OUT_INDEX][head_idx], contrast_factor=contrast_strength),
                       min=0.0, max=1.0)
            for head_idx in range(attn_weight.shape[1])
        ])

        attn_weight[OUT2_INDEX] = torch.stack([
            torch.clip(enhance_tensor(attn_weight[OUT2_INDEX][head_idx], contrast_factor=contrast_strength),
                       min=0.0, max=1.0)
            for head_idx in range(attn_weight.shape[1])
        ])

    else:
        attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)

    return attn_weight @ V, attn_weight


def enhance_tensor(tensor: torch.Tensor, contrast_factor: float = 1.67) -> torch.Tensor:
    """ Compute the attention map contrasting. """
    adjusted_tensor = (tensor - tensor.mean(dim=-1)) * contrast_factor + tensor.mean(dim=-1)
    return adjusted_tensor


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2) 

def vis_attention(new_attn, attn_weight):
    a = new_attn.detach().to('cpu')
    a_mean = a.mean(dim=0)
    token1 = a_mean[0].view(64,64)
    
    b = attn_weight[0].detach().to('cpu')
    b_mean = b.mean(dim=0)
    b_token1 = b_mean[0].view(64,64)

    # query_spatial = (query_spatial - query_spatial.min()) / (query_spatial.max() - query_spatial.min())
    plt.imshow(token1, cmap='viridis')
    plt.show()

def save_heatmap(attn_weight, timestep, batch_idx, key_token, is_contrast=False, is_softmax=False, output="/Users/jannabruner/Documents/research/sign_language_project/video-illustration/results/heatmaps"):
    if is_softmax:
        if is_contrast:
            name = f"after_softmax_no_contrast_avg_heads_idx_{batch_idx}_key_token_{key_token}_timestep_{timestep}"
        else: 
            name = f"after_softmax_with_contrast_avg_heads_idx_{batch_idx}_key_token_{key_token}_timestep_{timestep}"

    else:
        name = f"before_softmax_avg_heads_idx_{batch_idx}_key_token_{key_token}_timestep_{timestep}"

    a_w = attn_weight.detach().to('cpu')
    a = a_w[batch_idx].detach().to('cpu')
    a_mean = a.mean(dim=0)
    a_token = a_mean[key_token].view(64,64)                         
    plt.imshow(a_token, cmap='plasma', aspect='auto') 
    plt.colorbar(label='Value')  # Colorbar with a label
    plt.title(f'{name}')
    plt.axis('off')
    plt.savefig(f"{output}/{name}.png", dpi=300)  # High-quality image
    plt.close()

def save_max_heatmap(new_attn, timestep, key_token, is_sofmax=False, output="/Users/jannabruner/Documents/research/sign_language_project/video-illustration/results/heatmaps"):

    if is_sofmax:
        name = f"after_softmax_max_mean_heads_key_token_{key_token}_timestep_{timestep}"
    else:
        name = f"before_softmax_max_mean_heads_key_token_{key_token}_timestep_{timestep}"

        
    new_attn = new_attn.detach().to('cpu').mean(dim=0)
    new_attn_token = new_attn[key_token].view(64,64)
    plt.imshow(new_attn_token, cmap='plasma', aspect='auto') 
    plt.colorbar(label='Value')  # Colorbar with a label
    plt.title(f'{name}')
    plt.axis('off')
    plt.savefig(f"{output}/{name}.png", dpi=300)  # High-quality image
    plt.close()
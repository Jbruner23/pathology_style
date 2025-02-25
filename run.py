import sys
from typing import List

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers.training_utils import set_seed
from typing import NamedTuple, Optional
from pathlib import Path

sys.path.append(".")
sys.path.append("..")

from appearance_transfer_model import AppearanceTransferModel
from config import RunConfig, Range
from utils import latent_utils
from utils.latent_utils import load_latents_or_invert_images

class Range(NamedTuple):
    start: int
    end: int


device = 'cuda' if torch.cuda.is_available() else "cpu"
@pyrallis.wrap()
def main(cfg: RunConfig):
    run(cfg)


def run(cfg: RunConfig) -> List[Image.Image]:
    pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))
    set_seed(cfg.seed)
    model = AppearanceTransferModel(cfg)
    latents, noises = load_latents_or_invert_images(model=model, cfg=cfg)
    model.set_latents(latents)
    model.set_noise(noises)

    print("Running appearance transfer...")
    images = run_appearance_transfer(model=model, cfg=cfg)
    print("Done.")
    return images


def run_appearance_transfer(model: AppearanceTransferModel, cfg: RunConfig) -> List[Image.Image]:
    init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
    batch_size = init_latents.size()[0]
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.enable_edit = True  # Activate our cross-image attention layers
    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)
    images = model.pipe(
        prompt=[cfg.prompt] * batch_size,
        latents=init_latents,
        guidance_scale=1.0,
        num_inference_steps=cfg.num_timesteps,
        swap_guidance_scale=cfg.swap_guidance_scale,
        callback=model.get_adain_callback(),
        eta=1,
        zs=init_zs,
        generator=torch.Generator(device).manual_seed(cfg.seed),
        cross_image_attention_range=Range(start=start_step, end=end_step),
    ).images
    # Save images
    images[0].save(cfg.output_path / f"out_{str(cfg.image_frame_1_path).split('/')[-1][:-4]}_seed_{cfg.seed}.png")
    images[1].save(cfg.output_path / f"out_{str(cfg.image_frame_2_path).split('/')[-1][:-4]}_seed_{cfg.seed}.png")

    images[2].save(cfg.output_path / f"out_q1_image_seed_{cfg.seed}.png")
    images[3].save(cfg.output_path / f"out_q1_edges_seed_{cfg.seed}.png")
    images[4].save(cfg.output_path / f"out_q2_image_seed_{cfg.seed}.png")
    images[5].save(cfg.output_path / f"out_q2_edges_seed_{cfg.seed}.png")
    images[6].save(cfg.output_path / f"out_style_seed_{cfg.seed}.png")
    images[7].save(cfg.output_path / f"out_style2_seed_P{cfg.seed}.png")
    
    joined_images = np.concatenate(images, axis=1)
    Image.fromarray(joined_images).save(cfg.output_path / f"out_joined---seed_{cfg.seed}.png")
    return images


if __name__ == '__main__':
    # main()

    cfg = RunConfig(
        image_frame_1_path = Path("/Users/jannabruner/Documents/MSc_IDC_Computer_Science/research/histopathology/samples/qualitive_classification_StainFuser/source2_1.png"),
        edges_frame_1_path = Path("/Users/jannabruner/Documents/MSc_IDC_Computer_Science/research/histopathology/samples/qualitive_classification_StainFuser/lineart_seg/source2_1.png"),
        image_frame_2_path = Path("/Users/jannabruner/Documents/MSc_IDC_Computer_Science/research/histopathology/samples/qualitive_classification_StainFuser/source2_1.png"),
        edges_frame_2_path = Path("/Users/jannabruner/Documents/MSc_IDC_Computer_Science/research/histopathology/samples/qualitive_classification_StainFuser/lineart_seg/source2_1.png"),
        # style_image_path=Path("/Users/jannabruner/Documents/MSc_IDC_Computer_Science/research/histopathology/samples/qualitive_classification_StainFuser/target2_1.png"),
        # style2_image_path=Path("/Users/jannabruner/Documents/MSc_IDC_Computer_Science/research/histopathology/samples/qualitive_classification_StainFuser/target2_2.png"),
        latents_style_path=Path("/Users/jannabruner/Documents/MSc_IDC_Computer_Science/research/histopathology/code/cross-imgs-multiframe-pathology/output/pathology/query=source2_1--key=target2_1--value=target2_1/latents/target2_1.pt"),
        latents_style2_path=Path("/Users/jannabruner/Documents/MSc_IDC_Computer_Science/research/histopathology/code/cross-imgs-multiframe-pathology/output/pathology/query=source2_1--key=target2_1--value=target2_1/latents/target2_1.pt"),
        use_masked_adain=False,
        seed=42,
        domain_name="pathology",
        prompt=None,
        load_latents=False,
        skip_steps=0,
        num_timesteps=1,
        cross_attn_32_range=Range(start=0,end=0),
        cross_attn_64_range=Range(start=0,end=3),
        adain_range=Range(start=25,end=45),
        swap_guidance_scale=3.5,
        contrast_strength=1.67,
    )

    run(cfg)

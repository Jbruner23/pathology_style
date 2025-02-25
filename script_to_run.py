import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from pathlib import Path
from config import RunConfig
from run import run
import gc
import os
import glob


patches_dir = "/content/data_for_style/data_for_style/macrophage_test_patches"
images_path = glob.glob(patches_dir+'/*/*.tif')

lineart_dir = "/content/data_for_style/data_for_style/macrophage_test_patches_lineart"
seed = 42

from typing import NamedTuple, Optional

class Range(NamedTuple):
    start: int
    end: int
# STYLE TRANSFER LOOP

latents_style = "/content/drive/MyDrive/research/hostopathology_dataset/style_macrophage/histology/query1=TCGA-55-1594-01Z-00-DX1_003_query2=TCGA-55-1594-01Z-00-DX1_003/latents/macro_TCGA-69-7760-01Z-00-DX1_003.pt"
for i in range(0,len(images_path), 2):
    image_name_1 = images_path[i].split('/')[-1]
    image_name_2 = images_path[i+1].split('/')[-1]
    print(f"Procceing {image_name_1} and {image_name_2}")

    config = RunConfig(
        image_frame_1_path = Path(images_path[i]),
        image_frame_2_path = Path(images_path[i+1]),
        edges_frame_1_path = Path(os.path.join(lineart_dir, image_name_1)),
        edges_frame_2_path = Path(os.path.join(lineart_dir,image_name_2)),
        latents_style_path = Path(latents_style),
        latents_style2_path = Path(latents_style),
        use_masked_adain = False,
        seed = 42,
        domain_name = "histopathology",
        prompt = None,
        load_latents = False,
        skip_steps = 30,
        num_timesteps = 100,
        cross_attn_32_range = Range(start=0, end=0),  # Using a tuple; adjust if your Range type is required
        cross_attn_64_range = Range(start=0, end=100),
        adain_range = Range(start=10, end=90),
        swap_guidance_scale = 3.5,
        contrast_strength = 1.67,
        output_path = Path("/content/drive/MyDrive/research/hostopathology_dataset/style_macrophage"),
    )

    gc.collect()
    torch.cuda.empty_cache()

    # Run your style transfer pipeline (which processes 512x512 images)
    images = run(cfg=config)  # 'images' is assumed to be a list of numpy arrays (each 512x512)
    torch.cuda.empty_cache()


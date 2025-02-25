from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import cv2
from PIL import Image

from appearance_transfer_model import AppearanceTransferModel
from config import RunConfig
from utils import image_utils
from utils.ddpm_inversion import invert

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_latents_or_invert_images(model: AppearanceTransferModel, cfg: RunConfig):

    def path_exists(path):
        return path is not None and path.exists()
    
    if all([cfg.load_latents, path_exists(cfg.img1_latent_save_path), path_exists(cfg.img2_latent_save_path),
            path_exists(cfg.edges1_latent_save_path), path_exists(cfg.edges2_latent_save_path), path_exists(cfg.style_latent_save_path), path_exists(cfg.style2_latent_save_path)]):

    # if cfg.load_latents and cfg.img1_latent_save_path.exists() and cfg.img2_latent_save_path.exists() and cfg.style_latent_save_path.exists():
        print("Loading existing latents...")
        latents = load_latents(cfg.img1_latent_save_path, cfg.img2_latent_save_path,cfg.edges1_latent_save_path, cfg.edges2_latent_save_path,  cfg.style_latent_save_path,  cfg.style2_latent_save_path)
        noises = load_noise(cfg.img1_latent_save_path, cfg.img2_latent_save_path,cfg.edges1_latent_save_path, cfg.edges2_latent_save_path, cfg.style_latent_save_path,  cfg.style2_latent_save_path)
        print("Done.")
    else:
        if cfg.latents_style_path is not None:
            print("Loading style latents...")
            latents_kv = load_specific_latents_path(cfg.latents_style_path)
            zs_kv = load_specific_noise_path(cfg.latents_style_path)
            print("Done.")
        if cfg.latents_style2_path is not None:
            print("Loading style2 latents...")
            latents_kv2 = load_specific_latents_path(cfg.latents_style2_path)
            noises_fv = load_specific_noise_path(cfg.latents_style2_path)

            print("Inverting images...")

        image_q1, edges_q1, image_q2, edges_q2, image_style, image_style2 = image_utils.load_images(cfg=cfg, save_path=cfg.output_path)
        model.enable_edit = False  # Deactivate the cross-image attention layers
        latents ,noises  = invert_images(image_q1=image_q1, image_q2=image_q2,edges_q1=edges_q1, edges_q2=edges_q2, image_kv=image_style, image_kv2=image_style2, sd_model=model.pipe, cfg=cfg)
        model.enable_edit = True
        if cfg.latents_style_path is not None:
            latents[4] = latents_kv
            noises[4] = zs_kv
        if cfg.latents_style2_path is not None:
            latents[5] = latents_kv2
            noises[5] = noises_fv

        print("Done.")
    return latents, noises

def load_latents(img1_latent_save_path: Path,  img2_latent_save_path: Path, edges1_latent_save_path: Path, 
                 edges2_latent_save_path: Path, style_latent_save_path: Path, style2_latent_save_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:

    latents_q1_im = torch.load(img1_latent_save_path)
    latents_q2_im = torch.load(img2_latent_save_path)
    latents_q1_edges = torch.load(edges1_latent_save_path)
    latents_q2_edges = torch.load(edges2_latent_save_path)
    latents_kv = torch.load(style_latent_save_path)
    latents_kv2 = torch.load(style2_latent_save_path)

    if type(latents_q1_im) == list:
        latents_q1_im = [l.to(device) for l in latents_q1_im]
        latents_q2_im = [l.to(device) for l in latents_q2_im]
        latents_q1_edges =  [l.to(device) for l in latents_q1_edges]
        latents_q2_edges =  [l.to(device) for l in latents_q2_edges]
        latents_kv = [l.to(device) for l in latents_kv]
        latents_kv2 = [l.to(device) for l in latents_kv2]
    else:
        latents_q1_im = latents_q1_im.to(device)
        latents_q2_im = latents_q2_im.to(device)
        latents_q1_edges =  latents_q1_edges.to(device)
        latents_q2_edges =  latents_q2_edges.to(device)
        latents_kv = latents_kv.to(device)
        latents_kv2 = latents_kv2.to(device)
    return [latents_q1_im, latents_q2_im,latents_q1_edges, latents_q2_edges, latents_kv, latents_kv2]
   

def load_noise(img1_latent_save_path: Path, img2_latent_save_path: Path,  edges1_latent_save_path: Path,
                edges2_latent_save_path: Path, style_latent_save_path: Path, style2_latent_save_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    
    noise_q1_im = torch.load(img1_latent_save_path.parent / (img1_latent_save_path.stem + "_ddpm_noise.pt"))
    noise_q2_im = torch.load(img2_latent_save_path.parent / (img2_latent_save_path.stem + "_ddpm_noise.pt"))
    noise_q1_edges = torch.load(edges1_latent_save_path.parent / (edges2_latent_save_path.stem + "_ddpm_noise.pt"))
    noise_q2_edges = torch.load(img2_latent_save_path.parent / (img2_latent_save_path.stem + "_ddpm_noise.pt"))
    noise_kv = torch.load(style_latent_save_path.parent / (style_latent_save_path.stem + "_ddpm_noise.pt"))
    noise_kv2 = torch.load(style2_latent_save_path.parent / (style2_latent_save_path.stem + "_ddpm_noise.pt"))
    
    noise_q1_im = noise_q1_im.to(device)
    noise_q2_im = noise_q2_im.to(device)
    noise_q1_edges = noise_q1_edges.to(device)
    noise_q2_edges = noise_q2_edges.to(device)
    noise_kv = noise_kv.to(device)
    noise_kv2 = noise_kv.to(device)
    return  [noise_q1_im, noise_q2_im,noise_q1_edges, noise_q2_edges, noise_kv, noise_kv2]


def invert_images(sd_model: AppearanceTransferModel, image_q1: Image.Image, image_q2: Image.Image,edges_q1: Image.Image,
                  edges_q2: Image.Image, image_kv:Image.Image,image_kv2:Image.Image, cfg: RunConfig):
    
    input_q1_im = torch.from_numpy(np.array(image_q1)).float() / 127.5 - 1.0
    input_q2_im = torch.from_numpy(np.array(image_q2)).float() / 127.5 - 1.0
        
    input_q1_edges = torch.from_numpy(np.array(edges_q1)).float() / 127.5 - 1.0
    input_q2_edges = torch.from_numpy(np.array(edges_q2)).float() / 127.5 - 1.0

    if image_kv is not None:
        input_kv = torch.from_numpy(np.array(image_kv)).float() / 127.5 - 1.0
    if image_kv2 is not None:
        input_kv2 = torch.from_numpy(np.array(image_kv2)).float() / 127.5 - 1.0
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    zs_q1_im, latents_q1_im = invert(x0=input_q1_im.permute(2, 0, 1).unsqueeze(0).to(device),
                                 pipe=sd_model,
                                 prompt_src=cfg.prompt,
                                 num_diffusion_steps=cfg.num_timesteps,
                                 cfg_scale_src=3.5)
    
    zs_q2_im, latents_q2_im = invert(x0=input_q2_im.permute(2, 0, 1).unsqueeze(0).to(device),
                                       pipe=sd_model,
                                       prompt_src=cfg.prompt,
                                       num_diffusion_steps=cfg.num_timesteps,
                                       cfg_scale_src=3.5)
    
    zs_q1_edges, latents_q1_edges = invert(x0=input_q1_edges.permute(2, 0, 1).unsqueeze(0).to(device),
                                 pipe=sd_model,
                                 prompt_src=cfg.prompt,
                                 num_diffusion_steps=cfg.num_timesteps,
                                 cfg_scale_src=3.5)
    
    zs_q2_edges, latents_q2_edges = invert(x0=input_q2_edges.permute(2, 0, 1).unsqueeze(0).to(device),
                                       pipe=sd_model,
                                       prompt_src=cfg.prompt,
                                       num_diffusion_steps=cfg.num_timesteps,
                                       cfg_scale_src=3.5)
    
    if image_kv is not None:
        zs_kv, latents_kv = invert(x0=input_kv.permute(2, 0, 1).unsqueeze(0).to(device),
                                        pipe=sd_model,
                                        prompt_src=cfg.prompt,
                                        num_diffusion_steps=cfg.num_timesteps,
                                        cfg_scale_src=3.5)
    else:
        zs_kv = None
        latents_kv = None

    if image_kv2 is not None:
        zs_kv2, latents_kv2 = invert(x0=input_kv2.permute(2, 0, 1).unsqueeze(0).to(device),
                                        pipe=sd_model,
                                        prompt_src=cfg.prompt,
                                        num_diffusion_steps=cfg.num_timesteps,
                                        cfg_scale_src=3.5)
    else:
        zs_kv2 = None
        latents_kv2 = None

    # Save the inverted latents and noises
    # torch.save(latents_q1_im, cfg.latents_path / f"{cfg.image_frame_1_path.stem}.pt")
    # torch.save(latents_q2_im, cfg.latents_path / f"{cfg.image_frame_2_path.stem}.pt")

    # torch.save(latents_q1_edges, cfg.latents_path / f"{cfg.edges_frame_1_path.stem}.pt")
    # torch.save(latents_q2_edges, cfg.latents_path / f"{cfg.edges_frame_2_path.stem}.pt")

    if image_kv is not None:
        torch.save(latents_kv, cfg.latents_path / f"{cfg.style_image_path.stem}.pt")
    if image_kv2 is not None:
        torch.save(latents_kv2, cfg.latents_path / f"{cfg.style2_image_path.stem}.pt")

    # torch.save(zs_q1_im, cfg.latents_path / f"{cfg.image_frame_1_path.stem}_ddpm_noise.pt")
    # torch.save(zs_q2_im, cfg.latents_path / f"{cfg.image_frame_2_path.stem}_ddpm_noise.pt")

    # torch.save(zs_q1_edges, cfg.latents_path / f"{cfg.edges_frame_1_path.stem}_ddpm_noise.pt")
    # torch.save(zs_q2_edges, cfg.latents_path / f"{cfg.edges_frame_2_path.stem}_ddpm_noise.pt")

    if image_kv is not None:
        torch.save(zs_kv, cfg.latents_path / f"{cfg.style_image_path.stem}_ddpm_noise.pt")
    if image_kv2 is not None:
        torch.save(zs_kv2, cfg.latents_path / f"{cfg.style2_image_path.stem}_ddpm_noise.pt")


    latents =  [latents_q1_im, latents_q2_im,latents_q1_edges, latents_q2_edges, latents_kv, latents_kv2] 
    noises = [zs_q1_im, zs_q2_im, zs_q1_edges,zs_q2_edges,zs_kv, zs_kv2]
    return latents, noises


def get_init_latents_and_noises(model: AppearanceTransferModel, cfg: RunConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    # If we stored all the latents along the diffusion process, select the desired one based on the skip_steps
    if model.latents_q1_im.dim() == 4 and model.latents_q2_im.dim() == 4 and model.latents_q1_edges.dim() == 4 and model.latents_q2_edges.dim() == 4 and model.latents_kv.dim() == 4 and model.latents_kv.shape[0] > 1:
        model.latents_q1_im = model.latents_q1_im[cfg.skip_steps]
        model.latents_q2_im = model.latents_q2_im[cfg.skip_steps]

        model.latents_q1_edges = model.latents_q1_edges[cfg.skip_steps]
        model.latents_q2_edges = model.latents_q2_edges[cfg.skip_steps]

        model.latents_kv = model.latents_kv[cfg.skip_steps]
        model.latents_kv2 = model.latents_kv2[cfg.skip_steps]

    # we init out1 and  out2 from edges map
    init_latents = torch.stack([model.latents_q1_edges, model.latents_q2_edges ,model.latents_q1_im,
                                 model.latents_q1_edges, model.latents_q2_im, model.latents_q2_edges,
                                   model.latents_kv,  model.latents_kv2])
    
    init_zs = [model.zs_q1_edges[cfg.skip_steps:], model.zs_q2_edges[cfg.skip_steps:], model.zs_q1_im[cfg.skip_steps:],
                model.zs_q1_edges[cfg.skip_steps:], model.zs_q2_im[cfg.skip_steps:], model.zs_q2_edges[cfg.skip_steps:], 
                model.zs_kv[cfg.skip_steps:], model.zs_kv2[cfg.skip_steps:]]
    return init_latents, init_zs


def load_specific_latents_path(latents_path: Path) -> torch.Tensor:

    latents = torch.load(latents_path)

    if type(latents) == list:
        latents = [l.to(device) for l in latents]

    else:
        latents = latents.to(device)

    return latents


def load_specific_noise_path(latents_path: Path) -> torch.Tensor:

    noise = torch.load(latents_path.parent / (latents_path.stem + "_ddpm_noise.pt"))
    noise = noise.to(device)

    return  noise
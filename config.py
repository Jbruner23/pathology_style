from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, List


class Range(NamedTuple):
    start: int
    end: int


@dataclass
class RunConfig:
    # first frame image path
    image_frame_1_path: Path
    # second frame image path
    image_frame_2_path: Path  
    edges_frame_1_path: Path
    edges_frame_2_path: Path

    style_image_path:    Optional[Path] = None
    style2_image_path:   Optional[Path] = None
    latents_style_path:  Optional[Path] = None
    latents_style2_path: Optional[Path] = None
    
    
    # applying weights to attention scores experiment
    attn_output: Optional[Path] = None # if you wish to save visualization of attention scores 
    attn_mask: Optional[Path] = None # path to mask, applied on attention scores after softmax

    # fg/bg experiments

    #visualizing specific tokens    
    list_tokens: Optional[List] = None # list of tokens to visualize the attnetion on queries (Attn before softmax)
    
    # overlay experiments
    overlay_type: Optional[str] = None # None/'min_overlay'/'max_overlay'/'mean_overlay' output overlay based on img1 and img2 
    
    #applying masks to KQV, I changed it to fg/bg attention compatability
    # the mask to apply on queriers, list of Paths
    query_masks_path: Optional[List] = None
    # factor to multiply the query mask, list of floats
    query_scale: Optional[List] = None
    # the mask to apply on key
    key_masks_path: Optional[List] = None
    # factor to multiply the key mask
    key_scale: Optional[float] = None
    # the mask to apply on value
    value_masks_path: Optional[List] = None
    # factor to multiply the value mask
    value_scale: Optional[float] = None
    
    
    # apply masks on 64x64 res attention layers
    inject_64_res: bool = False
    # apply masks on 32x32 res attention layers
    inject_32_res: bool = False
    # Domain name (e.g., buildings, animals)
    domain_name: Optional[str] = None
    # Output path
    output_path: Path = Path('./output')
    # Random seed
    seed: int = 42
    # Input prompt for inversion (will use domain name as default)
    prompt: Optional[str] = None
    # Number of timesteps
    num_timesteps: int = 100
    # Whether to use a binary mask for performing AdaIN
    use_masked_adain: bool = False
    # Timesteps to apply cross-attention on 64x64 layers
    cross_attn_64_range: Range = Range(start=10, end=90)
    # Timesteps to apply cross-attention on 32x32 layers
    # cross_attn_32_range: Range = Range(start=10, end=70)
    cross_attn_32_range: Range = Range(start=0, end=0)
    # Timesteps to apply AdaIn
    adain_range: Range = Range(start=25, end=45)
    # Swap guidance scale
    swap_guidance_scale: float = 3.5
    # Attention contrasting strength
    contrast_strength: float = 1.67
    # Object nouns to use for self-segmentation (will use the domain name as default)
    object_noun: Optional[str] = None
    # Whether to load previously saved inverted latent codes
    load_latents: bool = True
    # Number of steps to skip in the denoising process (used value from original edit-friendly DDPM paper)
    skip_steps: int = 32

    def __post_init__(self):
        save_name = f'query1={self.image_frame_1_path.stem}_query2={self.image_frame_2_path.stem}'
        # self.output_path = self.output_path / save_name
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Handle the domain name, prompt, and object nouns used for masking, etc.
        if self.use_masked_adain and self.domain_name is None:
            raise ValueError("Must provide --domain_name and --prompt when using masked AdaIN")
        if not self.use_masked_adain and self.domain_name is None:
            self.domain_name = "object"
        if self.prompt is None:
            self.prompt = f"A photo of a {self.domain_name}"
        if self.object_noun is None:
            self.object_noun = self.domain_name

        self.output_path = self.output_path / self.domain_name / save_name

        # Define the paths to store the inverted latents to
        self.latents_path = Path(self.output_path) / "latents"
        self.latents_path.mkdir(parents=True, exist_ok=True)
        self.img1_latent_save_path = self.latents_path / f"{self.image_frame_1_path.stem}.pt"
        self.img2_latent_save_path = self.latents_path / f"{self.image_frame_2_path.stem}.pt"
        self.edges1_latent_save_path = self.latents_path / f"{self.edges_frame_1_path.stem}.pt"
        self.edges2_latent_save_path = self.latents_path / f"{self.edges_frame_2_path.stem}.pt"
        if self.style_image_path is not None:
            self.style_latent_save_path = self.latents_path / f"{self.style_image_path.stem}.pt"
        else:
            self.style_latent_save_path = None
        if self.style2_image_path is not None:
            self.style2_latent_save_path = self.latents_path / f"{self.style2_image_path.stem}.pt"
        else: 
            self.style2_latent_save_path = None

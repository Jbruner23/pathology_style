# style transfer for pathology

Implementation based on https://github.com/garibida/cross-image-attention. 

Code was modified to use combination of queries - edges and rgb image. 
Batch size = 2 (2 source images with 2 target styles)

use latents or style image:

image_frame_1_path: Path
image_frame_2_path: Path  
edges_frame_1_path: Path
edges_frame_2_path: Path

style_image_path:    Optional[Path] = None
style2_image_path:   Optional[Path] = None
latents_style_path:  Optional[Path] = None
latents_style2_path: Optional[Path] = None

for best results in the config file: 
1. disable the injection of 32x32 attention layers of the decoder (only 64x64 attention layers of the decoder will be modified)
2. use "domain_name". 
3. for this code set use_masked_adain=False (was not tested with masked adain) 


python run.py --image_frame_1_path ../../images/reference_styles/32.png --edges_frame_1_path ../../images/bonn_frame_0026/bonn_frame_0026.jpg --image_frame_2_path ../../images/reference_styles/32.png --edges_frame_2_path ../../images/bonn_frame_0026/bonn_frame_0026.jpg --style_image_path ../../images/reference_styles/32.png --style2_image_path ../../images/bonn_frame_0026/bonn_frame_0026.jpg --output_path ../../results/ --use_masked_adain False --load_latents False --num_timesteps 100 --skip_steps 30 --contrast_strength 1.67 --swap_guidance_scale 3.5 --domain_name "histopathology" 

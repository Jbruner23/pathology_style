import pathlib
from typing import Optional, Tuple
import sys
import os
sys.path.append(os.curdir)
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from config import RunConfig
import torch 
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_images(cfg: RunConfig, save_path: Optional[pathlib.Path] = None) -> Tuple[Image.Image, Image.Image]:

    image_query1 = load_size(cfg.image_frame_1_path)
    image_query2 = load_size(cfg.image_frame_2_path)
  
    edges_query1 = load_size(cfg.edges_frame_1_path)
    edges_query2 = load_size(cfg.edges_frame_2_path)  
    if cfg.style_image_path is not None:
        style_image = load_size(cfg.style_image_path)
    else:
        style_image = None
    if cfg.style2_image_path is not None:
        style2_image = load_size(cfg.style2_image_path)
    else:
        style2_image = None

    if save_path is not None:
        Image.fromarray(image_query1).save(save_path / f"image_query1.png")
        Image.fromarray(image_query2).save(save_path / f"image_query2.png")
        Image.fromarray(edges_query1).save(save_path / f"edges_query1.png")
        Image.fromarray(edges_query2).save(save_path / f"edges_query2.png")
        if cfg.style_image_path is not None:
            Image.fromarray(style_image).save(save_path / f"style_image.png")

        if cfg.style2_image_path is not None:
            Image.fromarray(style2_image).save(save_path / f"style2_image.png")

    return image_query1, edges_query1, image_query2, edges_query2, style_image, style2_image


def load_size(image_path: pathlib.Path,
              left: int = 0,
              right: int = 0,
              top: int = 0,
              bottom: int = 0,
              size: int = 512) -> Image.Image:
    import numpy as np
    if isinstance(image_path, (str, pathlib.Path)):
        image = np.array(Image.open(str(image_path)).convert('RGB'))  
    else:
        image = image_path

    h, w, _ = image.shape

    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]

    h, w, c = image.shape

    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]

    image = np.array(Image.fromarray(image).resize((size, size)))
    return image

def resize_mask(image_mask_path: Path, res=(64,64), interpolation=cv2.INTER_NEAREST, combine_masks=True):
    """
    resize a given mask
    
    Args:
        image_mask (numpy.ndarray): The binary mask (e.g., 512x512).
        res (tuple): resolution of the new mask (h,w)
        interpolation (cv2 interpolation type): a method for interpolating 
        combine_masks: wether to combine into one mask

    Returns:
        resized mask (e.g., 64x64).
    """
    start_mask = torch.zeros((res))
    end_mask = torch.zeros((res))
    h, w = res[0], res[1]

    for i in range(0,4):
        
        image_mask = load_size(image_mask_path[i])
        # Resize mask using nearest-neighbor interpolation
        resized_mask = cv2.resize(image_mask, (w, h), interpolation=interpolation)
        resized_mask =  torch.from_numpy(np.array(resized_mask)).float() / 255.0

        if combine_masks:
            start_mask[resize_mask >0] = 1.0

    for i in range(4,8):
        image_mask = load_size(image_mask_path[i])
        # Resize mask using nearest-neighbor interpolation
        resized_mask = cv2.resize(image_mask, (w, h), interpolation=interpolation)
        resized_mask =  torch.from_numpy(np.array(resized_mask)).float() / 255.0

        if combine_masks:
            end_mask[resize_mask >0] = 1.0

    # mask_64 = mask_64.permute(2, 0, 1).unsqueeze(0).to(device)

    # input_mask = torch.from_numpy(np.array(image_mask)).float() / 127.5 - 1.0

    # x0 = input_mask.permute(2, 0, 1).unsqueeze(0).to(device)
    # latent_mask = (sd_model.vae.encode(x0).latent_dist.mode() * 0.18215).float()

  
    # latents = (1 / 0.18215) * latents
    # with torch.no_grad():
    #     image = vae.decode(latents).sample
    # image = (image / 2 + 0.5).clamp(0, 1)

    return start_mask[:,:,0], end_mask[:,:,0]

def attention_per_head(query_activations):
    # Extract query and reshape for 8 heads
    query_tensor = query_activations[0].squeeze(0)  # Shape: [4096, 320]
    query_heads = query_tensor.view(4096, 8, 40)  # Shape: [4096, 8, 40]

    # Average over head dimensions (like before)
    query_head_1 = query_heads[:, 0, :].mean(dim=1).view(64, 64).numpy()  # For head 1
    query_head_2 = query_heads[:, 1, :].mean(dim=1).view(64, 64).numpy()  # For head 2

    # Plot the first two attention heads
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(query_head_1, cmap='viridis')
    ax[0].set_title('Query Head 1')

    ax[1].imshow(query_head_2, cmap='viridis')
    ax[1].set_title('Query Head 2')

    plt.show()

def attention_average_all_features(query_activations):
    import numpy as np

# Take the first layer's query (you might want to loop over query_activations)
    query_tensor = query_activations[0]  # Shape: [1, 4096, 320]
    query_tensor = query_tensor.squeeze(0)  # Remove batch dimension: [4096, 320]

    # Average over feature dimensions (optionally, you can take PCA here)
    query_spatial = query_tensor.mean(dim=1)  # Shape: [4096]
    query_spatial = query_spatial.view(64, 64).numpy()  # Reshape to (64, 64)

    # Plot the query as a heatmap
    plt.imshow(query_spatial, cmap='viridis')
    plt.colorbar()
    plt.title('Query Activation Heatmap')
    plt.show()

def attention_to_PCA(query_activations):
    from sklearn.decomposition import PCA

    query_tensor = query_activations[0].squeeze(0)  # Shape: [4096, 320]

    # Reduce dimensionality from 320 to 2
    pca = PCA(n_components=2)
    query_reduced = pca.fit_transform(query_tensor)  # Shape: [4096, 2]

    # Normalize and reshape each channel into (64, 64)
    channel_1 = query_reduced[:, 0].reshape(64, 64)
    channel_2 = query_reduced[:, 1].reshape(64, 64)

    # Plot the two PCA components
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(channel_1, cmap='viridis')
    ax[0].set_title('PCA Component 1')

    ax[1].imshow(channel_2, cmap='viridis')
    ax[1].set_title('PCA Component 2')

    plt.show()


def save_generated_masks(model, cfg: RunConfig):
    tensor2im(model.image_app_mask_32).save(cfg.output_path / f"mask_style_32.png")
    tensor2im(model.image_struct_mask_32).save(cfg.output_path / f"mask_struct_32.png")
    tensor2im(model.image_app_mask_64).save(cfg.output_path / f"mask_style_64.png")
    tensor2im(model.image_struct_mask_64).save(cfg.output_path / f"mask_struct_64.png")


def tensor2im(x) -> Image.Image:
    return Image.fromarray(x.cpu().numpy().astype(np.uint8) * 255)

def image_from_mask(image: Path, mask:Path, output_path:Path, dilate=False):
    image = Path("Users/jannabruner/Documents/research/sign_language_project/video-illustration/images/bonn_frame_0026/bonn_frame_0026.jpg")
    image = np.array(Image.open(image))
    mask = "/Users/jannabruner/Documents/research/sign_language_project/video-illustration/images/bonn_frame_0026/left_hand_mask.png"
    mask = np.array(Image.open(mask))
    output_path = Path("/Users/jannabruner/Documents/research/sign_language_project/video-illustration/images/bonn_frame_0026")

    if dilate:
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    image[mask==0] = 255
    Image.fromarray(image).save(output_path + image.stem + '/overlay.png')

    import cv2
import os

# Function to extract Canny edges and save the output
def extract_canny_edges(input_dir, output_dir, low_threshold=30, high_threshold=120):
    """
    Extracts Canny edges from images in the input directory and saves them in the output directory.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where output images will be saved.
        low_threshold (int): Low threshold for the Canny edge detector.
        high_threshold (int): High threshold for the Canny edge detector.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        # Construct full file path
        file_path = os.path.join(input_dir, filename)

        # Check if it is a file
        if os.path.isfile(file_path):
            # Read the image
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Could not read {filename}. Skipping.")
                continue
            blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

            # Apply Canny edge detection
            edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

            # Invert colors to make edges black and background white
            edges_inverted = cv2.bitwise_not(edges)

            # Save the resulting image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, edges_inverted)
            print(f"Processed and saved: {output_path}")




def edges_overley(images_path, output_dir,  low_threshold=30, high_threshold=120):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_names =  os.listdir(images_path)
    image_names = [img for img in image_names if img.endswith('png') or img.endswith('jpg')]
    for filename in image_names:

        img_path = os.path.join(images_path, filename)  
        gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        rgb_img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
        edges_inverted = cv2.bitwise_not(edges)
        edges_colored = cv2.cvtColor(edges_inverted, cv2.COLOR_GRAY2RGB)

        images = [rgb_img, edges_colored]

        blended_image = np.min(images, axis=0).astype(np.uint8)
        # darkest_image = Image.fromarray(blended_image)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, blended_image)


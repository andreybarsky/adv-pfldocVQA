import torch
from typing import Tuple

def mask_exclude_white(image: torch.Tensor):
    """
    Creates a mask for the image, excluding white pixels.
    """
    mask = torch.ones_like(image, dtype=torch.bool)
    white_pixels_mask = (image[:, :, 0] == 255) & \
                        (image[:, :, 1] == 255) & \
                        (image[:, :, 2] == 255)
    mask[white_pixels_mask] = 0
    return mask

def mask_include_all(image: torch.Tensor):
    """
    Creates a mask for the image, including all pixels.
    """
    mask = torch.ones_like(image, dtype=torch.bool)
    return mask

def mask_bottom_right_corner(image: torch.Tensor, ratio=0.15) -> torch.Tensor:
    mask = torch.zeros_like(image, dtype=torch.bool)

    # the images in this version have C*W*H shape
    H, W, _ = image.shape
    size = int(min(W,H)*ratio)
    x_start, y_start = W - size, H - size
    mask[y_start:, x_start:, :] = 1  # Exclude the bottom-right corner

    return mask
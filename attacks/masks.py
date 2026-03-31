import torch
from typing import Tuple
from jnd.perceptual import sobel, robust_sobel, srgb_to_oklab, robust_canny

from jnd import util as ut

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

    area_ratio = mask.sum() / torch.prod(torch.tensor(mask.shape))
    print(f'Sobel mask computed over image with desired side ratio={ratio} resulting in area ratio={area_ratio:.3f}')
    
    return mask

def mask_sobel(image: torch.Tensor, threshold=0.1):
    edges = sobel(image)
    edge_mask = (edges > threshold)

    ratio = edge_mask.sum() / torch.prod(torch.tensor(edge_mask.shape))
    print(f'Sobel mask computed over image with threshold={threshold} resulting in mask ratio={ratio:.3f}')

    # stack across all three channels:
    return torch.stack([edge_mask]*3, -1)

def mask_canny(image: torch.Tensor, threshold=0.1):
    edges = canny(image)
    edge_mask = (edges > threshold)

    ratio = edge_mask.sum() / torch.prod(torch.tensor(edge_mask.shape))
    print(f'Canny mask computed over image with threshold={threshold} resulting in mask ratio={ratio:.3f}')

    # stack across all three channels:
    return torch.stack([edge_mask]*3, -1)

def mask_robust_sobel(image: torch.Tensor, 
                      threshold=0.1, # if not None, binarise at this threshold
                      normalised=False, # if true, postprocess to range [0,1]
                      channelwise=False, # if true, compute edges per channel
                      **kwargs):
    assert image.shape[-1] == 3 # assume that the input is rgb or lab
    
    if not channelwise:
        # flatten to grayscale for edge detection
        flat_image = image.mean(dim=-1)
        robust_edges = robust_sobel(flat_image, **kwargs)
        # restack as edges in grayscale space:
        robust_edges = torch.stack([robust_edges]*3, -1)
        ch_str = 'monochrome'
    else:
        channel_edges = [robust_sobel(image[:,:,c]) for c in range(3)]
        robust_edges = torch.stack(channel_edges, -1)
        ch_str = 'channelwise'

    if threshold is not None:
        edge_mask = (robust_edges > threshold)
    
        ratio = edge_mask.sum() / edge_mask.numel()
        print(f'{ch_str.title()} robust sobel mask (kwargs: {kwargs}) computed over image with thresholding={threshold} resulting in mask ratio={ratio:.1%}')

    else:
        print(f'Robust sobel SOFT mask (kwargs: {kwargs}) computed over image resulting in l1: {ut.l1(robust_edges):.4f}')
        edge_mask = robust_edges

        if normalised:
            print(f'Normalising soft mask from range [{edge_mask.min():.2f}, {edge_mask.max():.2f}] to [0.00, 1.00]')
            edge_mask = edge_mask / edge_mask.max()


    return edge_mask

def mask_edge_channelwise(image: torch.Tensor, lab=True, threshold=0.2, scale_threshold=False, robust=True, normalised=True, edge_type='sobel', **edge_kwargs):
    assert len(image.shape) == 3 # ensure unbatched input
    assert image.shape[-1] == 3  # ensure BHWC format
    assert edge_type in ('canny', 'sobel')
    
    if lab:
        if image.max() > 1:
            print(f'Rescaling image from int units to float units')
            image = image / 255.
        print(f'Converting from RGB to LAB space')
        image = srgb_to_oklab(image)

    if robust:
        # edge_func = robust_sobel
        edge_func = robust_canny if edge_type == 'canny' else robust_sobel
    else:
        edge_func = canny if edge_type == 'canny' else sobel
    print(f'Using edge function: {edge_func}')

    # edge_kwargs = edge_kwargs or {}
    # channel_edges = torch.stack([edge_func(image[:,:,c:c+1], **edge_kwargs) for c in range(3)], -1) # for sobel
    channel_edges = torch.stack([edge_func(image[:,:,c], **edge_kwargs) for c in range(3)], -1) # for canny
    if threshold is not None:
        if scale_threshold:
            threshold = image.std(dim=(0,1)) * threshold # proportional to the sd in each channel
        edge_mask = (channel_edges > threshold)
        ratio = edge_mask.sum() / torch.prod(torch.tensor(edge_mask.shape))
        print(f'Channelwise {edge_type} mask computed over image with binary threshold={threshold} resulting in mask ratio={ratio:.3f}')        
    else:
        edge_mask = channel_edges
        print(f'Channelwise {edge_type} filter computed over image with min={edge_mask.min():.3f}, max={edge_mask.max():.3f}, mean={edge_mask.mean():.3f}')
        if normalised:
            print(f'Normalising to range [0,1]')
            edge_mask = edge_mask / edge_mask.max()
    
    for c in range(3):
        c_ratio = edge_mask[:,:,c].sum() / torch.prod(torch.tensor(edge_mask.shape[:-1])) 
        print(f'  channel {c}: {c_ratio:.3f}')

    return edge_mask 

def mask_sobel_blue(image: torch.Tensor, threshold=0.1):
    """same as above but only in the blue channel"""
    edges = sobel(image)
    edge_mask = (edges > threshold)

    ratio = edge_mask.sum() / torch.prod(torch.tensor(edge_mask.shape))
    print(f'Sobel mask computed over image with threshold={threshold} resulting in mask ratio={ratio:.3f} \n(in practice {(ratio/3):.3f} since only one channel is used)')

    # use in blue channel only:
    return torch.stack([torch.zeros_like(edge_mask)]*2 + [edge_mask], -1)    

def mask_sobel_red(image: torch.Tensor, threshold=0.1):
    """same as above but only in the red channel"""
    edges = sobel(image)
    edge_mask = (edges > threshold)

    ratio = edge_mask.sum() / torch.prod(torch.tensor(edge_mask.shape))
    print(f'Sobel mask computed over image with threshold={threshold} resulting in mask ratio={ratio:.3f} \n(in practice {(ratio/3):.3f} since only one channel is used)')

    # use in blue channel only:
    return torch.stack([edge_mask] + [torch.zeros_like(edge_mask)]*2, -1)        

def mask_sobel_chromatic(image: torch.Tensor, threshold=0.1):
    """same as above but only in the last two channels (intended for a,b channels in lab space)"""
    edges = sobel(image)
    edge_mask = (edges > threshold)

    ratio = edge_mask.sum() / torch.prod(torch.tensor(edge_mask.shape))
    print(f'Sobel mask computed over image with threshold={threshold} resulting in mask ratio={ratio:.3f} \n(in practice {(ratio*(2/3)):.3f} since only two channels are used)')

    # use in blue channel only:
    return torch.stack([torch.zeros_like(edge_mask)] + [edge_mask]*2, -1)     



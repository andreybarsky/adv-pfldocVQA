import torch
import numpy as np

def chinfo(tensor):
    """summarise the min, max, and median values per channel of this tensor"""
    if len(tensor.shape) == 2:
        # if given a raw HxW map, treat it as single channel:
        tensor = tensor.unsqueeze(-1)
    
    channels = torch.unbind(tensor, -1)
    for c, chan in enumerate(channels):
        print(f'Channel {c}:')
        print(f'  Range [{chan.min():.2f}, {chan.max():.2f}]')
        print(f'  Mean: {chan.mean():.3f}, Median: {chan.median():.2f}')
        print(f'  SD: {chan.std():.3f}')

def channel_clamp(x, mins:list, maxs:list):
    """clamp the values in each channel (last dim) of a tensor"""
    channels = torch.unbind(x, dim=-1)
    new_channels = []
    for ch, mn, mx in zip(channels, mins, maxs):
        new_channels.append(torch.clamp(ch, mn, mx))
    return torch.stack(new_channels, -1)

def lab_clamp(lab_tensor):
    """clamp the values of a LAB-space tensor
    to a reasonable gamut"""
    return channel_clamp(lab_tensor,
              mins=[0., -0.4, -0.4], 
              maxs=[1., 0.4, 0.4])

def rgb_clamp(rgb_tensor, rgb_max=None):
    """clamp the values of a RGB-space tensor
    to valid RGB range"""
    if rgb_max is None:
        rgb_max = 1. if rgb_tensor.max() < 255 else 255.
    return channel_clamp(rgb_tensor,
              mins=[0]*3, 
              maxs=[rgb_max]*3)            
def l1(tensor):
    """calculate l1 norm of a tensor"""
    return tensor.abs().mean().item()

def linf(tensor):
    return tensor.abs().mean().item()

def delta_l1(clean, adv):
    """calculate l1 norm of the diff between two tensors"""
    delta = adv - clean
    return l1(delta)

def delta_linf(clean, adv):
    delta = adv - clean
    return linf(delta)

def lab_delta_l1(clean_rgb, adv_rgb):
    """calculate l1 norm in lab space between two rgb tensors"""
    # convert to lab space:
    from jnd.perceptual import srgb_to_oklab
    clean_lab = srgb_to_oklab(clean_rgb)
    adv_lab = srgb_to_oklab(adv_rgb)

    return delta_l1(clean_lab, adv_lab)

def lab_delta_linf(clean_rgb, adv_rgb):
    # convert to lab space:
    from jnd.perceptual import srgb_to_oklab
    clean_lab = srgb_to_oklab(clean_rgb)
    adv_lab = srgb_to_oklab(adv_rgb)

    return delta_linf(clean_lab, adv_lab)    

def rgb_delta_l1(clean_lab, adv_lab):
    """calculate l1 norm in lab space between two rgb tensors"""
    # convert to rgb space:    
    from jnd.perceptual import oklab_to_srgb
    clean_rgb = oklab_to_srgb(clean_lab)
    adv_rgb = oklab_to_srgb(adv_lab)

    return delta_l1(clean_lab, adv_lab)

def rgb_delta_linf(clean_lab, adv_lab):
    # convert to rgb space:
    from jnd.perceptual import oklab_to_srgb
    clean_rgb = oklab_to_srgb(clean_lab)
    adv_rgb = oklab_to_srgb(adv_lab)

    return delta_linf(clean_rgb, adv_rgb)        

def fuzzy_numel(tensor):
    """return the number of elements that are greater than 0 but less than 1"""
    return torch.all(torch.stack([tensor > 0, tensor < 1], 0), 0).sum()

def num_quantiles(tensor):
    """count the number of distinct value quantiles in this tensor"""
    return tensor.unique().numel()
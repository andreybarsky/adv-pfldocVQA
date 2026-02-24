import pywt
import numpy as np

def pad_to_swt(img, level):
    """Pad image so dimensions are divisible by 2^level."""
    factor = 2 ** level
    h, w = img.shape[:2]
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='edge')

def swt_forward(img, wavelet='db4', level=3):
    """img: RGB numpy array -> list of per-channel SWT coefficient structures"""

    h,w = img.shape[:2]
    if (h % (2**level) != 0) or (w % (2**level) != 0):
        img = pad_to_swt(img, level)
        print(f'Image dimensions not properly divisible, padding from {h,w} to {img.shape[:2]}')
    
    coeffs_per_channel = []
    for ch in range(img.shape[2]):
        data = img[:, :, ch].astype(np.float64)
        coeffs = pywt.swt2(data, wavelet, level=level)
        coeffs_per_channel.append(coeffs)
    return coeffs_per_channel

def swt_inverse(coeffs_per_channel, original_shape, wavelet='db4', dtype='float'):
    """Inverse SWT -> uint8 or float32 RGB numpy array"""
    h, w = original_shape[:2]
    channels = []
    for coeffs in coeffs_per_channel:
        rec = pywt.iswt2(coeffs, wavelet)
        channels.append(rec[:h, :w])
    out = np.stack(channels, axis=2)
    if dtype == 'int':
        return np.clip(out, 0, 255).astype(np.uint8)
    elif dtype == 'float':
        return np.clip(out, 0, 1).astype(np.float32)
    else:
        raise Exception('dtype should be "int" or "float"')

def swt_bandpass(coeffs_per_channel, keep_levels=None, kill_approx=False):
    """Filter SWT coefficients by level.
    
    keep_levels: set of 1-indexed levels (1=coarsest) to keep.
    kill_approx: if True, zero out the coarsest approximation only.
    """
    filtered = []
    for coeffs in coeffs_per_channel:
        new_coeffs = []
        for i, (cA, (cH, cV, cD)) in enumerate(coeffs):
            level_idx = i + 1
            if kill_approx and i == 0:
                new_cA = np.zeros_like(cA)
            else:
                new_cA = cA.copy()
            if keep_levels is not None and level_idx not in keep_levels:
                new_detail = (np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD))
            else:
                new_detail = (cH.copy(), cV.copy(), cD.copy())
            new_coeffs.append((new_cA, new_detail))
        filtered.append(new_coeffs)
    return filtered

def swt_attenuate(coeffs_per_channel, level_gains=None, approx_gain=None):
    """Attenuate SWT coefficients by level with arbitrary gain factors.
    
    level_gains: dict of {level_idx: gain} where level_idx is 1-indexed.
                 gain=0.0 kills the band, gain=1.0 keeps it, gain=2.0 boosts it.
    approx_gain: if set, multiply the coarsest approximation coefficients by this.
    """
    if level_gains is None:
        level_gains = {}
    
    filtered = []
    for coeffs in coeffs_per_channel:
        new_coeffs = []
        for i, (cA, (cH, cV, cD)) in enumerate(coeffs):
            level_idx = i + 1
            
            if approx_gain is not None and i == 0:
                new_cA = cA * approx_gain
            else:
                new_cA = cA.copy()
            
            g = level_gains.get(level_idx, 1.0)
            new_detail = (cH * g, cV * g, cD * g)
            new_coeffs.append((new_cA, new_detail))
        filtered.append(new_coeffs)
    return filtered

def swt_add_noise(coeffs_per_channel, level_noise=None, approx_noise=None):
    """Add noise scaled per level to SWT coefficients.
    
    level_noise: dict of {level_idx: strength} for detail coefficients.
    approx_noise: strength for coarsest approximation coefficients.
    """
    if level_noise is None:
        level_noise = {}
    
    result = []
    for coeffs in coeffs_per_channel:
        new_coeffs = []
        for i, (cA, (cH, cV, cD)) in enumerate(coeffs):
            level_idx = i + 1
            
            if approx_noise is not None and i == 0:
                noise = np.random.randn(*cA.shape) * approx_noise
                new_cA = cA + noise
            else:
                new_cA = cA.copy()
            
            s = level_noise.get(level_idx, 0.0)
            if s > 0:
                new_detail = (
                    cH + np.random.randn(*cH.shape) * s,
                    cV + np.random.randn(*cV.shape) * s,
                    cD + np.random.randn(*cD.shape) * s,
                )
            else:
                new_detail = (cH.copy(), cV.copy(), cD.copy())
            new_coeffs.append((new_cA, new_detail))
        result.append(new_coeffs)
    return result
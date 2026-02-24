import numpy as np
from scipy.fftpack import dct, idct

def dct2(block):
    return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(block):
    return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def block_process(img, func, B=8):
    h, w, c = img.shape
    # Pad to multiple of B
    ph = (B - h % B) % B
    pw = (B - w % B) % B
    padded = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='edge')
    out = np.zeros_like(padded, dtype=np.float64)
    for ch in range(c):
        for i in range(0, padded.shape[0], B):
            for j in range(0, padded.shape[1], B):
                out[i:i+B, j:j+B, ch] = func(padded[i:i+B, j:j+B, ch])
    return out, (h, w)

def dct_forward(img, block_size=8):
    """img: uint8 RGB numpy array -> DCT coefficients (float64), original shape"""
    return block_process(img.astype(np.float64), dct2, block_size)

def dct_inverse(coeffs, original_shape, block_size=8):
    """DCT coefficients -> uint8 RGB numpy array, cropped to original shape"""
    rec, _ = block_process(coeffs, idct2, block_size)
    h, w = original_shape
    return np.clip(rec[:h, :w], 0, 255).astype(np.uint8)

def dct_inverse_float(coeffs, original_shape, block_size=8):
    """DCT coefficients -> float32 RGB numpy array, cropped to original shape"""
    rec, _ = block_process(coeffs, idct2, block_size)
    h, w = original_shape
    return np.clip(rec[:h, :w], 0, 1).astype(np.float32)

def dct_bandpass(coeffs, low=0, high=14, keep_dc=True):
    """Zero out DCT coefficients outside the frequency band [low, high].
    
    Frequency is measured as (i + j) for position (i, j) in each 8x8 block,
    ranging from 0 (DC) to 14 (highest frequency corner).
    """
    mask = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if low <= i + j <= high:
                mask[i, j] = 1.0

    if keep_dc:
        mask[0,0] = 1.0
    
    filtered = coeffs.copy()
    for ch in range(filtered.shape[2]):
        for i in range(0, filtered.shape[0], 8):
            for j in range(0, filtered.shape[1], 8):
                filtered[i:i+8, j:j+8, ch] *= mask
    return filtered

def dct_bandpass_smooth(coeffs, low=0, high=14, sigma=1.5, keep_dc=True):
    """Apply a Gaussian-tapered bandpass to DCT coefficients.
    
    Same frequency metric as dct_bandpass: freq = i + j for position (i, j).
    Instead of a hard cutoff, the mask rolls off with a Gaussian of width sigma
    at both the low and high edges.
    """
    mask = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            freq = i + j
            if low <= freq <= high:
                mask[i, j] = 1.0
            elif freq < low:
                mask[i, j] = np.exp(-0.5 * ((freq - low) / sigma) ** 2)
            else:
                mask[i, j] = np.exp(-0.5 * ((freq - high) / sigma) ** 2)

    if keep_dc:
        mask[0, 0] = 1.0

    filtered = coeffs.copy()
    for ch in range(filtered.shape[2]):
        for i in range(0, filtered.shape[0], 8):
            for j in range(0, filtered.shape[1], 8):
                filtered[i:i+8, j:j+8, ch] *= mask
    return filtered


def full_bandpass_smooth(img, low=0, high=14, sigma=1.2, keep_dc=True):
    """performs forward DCT transform, bandpasses, and IDCT transforms back"""
    coeffs, shape = dct_forward(img)
    coeffs2 = dct_bandpass_smooth(coeffs, low, high, sigma, keep_dc)
    img2 = dct_inverse_float(coeffs2, shape)
    return img2
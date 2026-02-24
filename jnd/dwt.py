import pywt
import numpy as np

def dwt_forward(img, wavelet='haar', level=3):
    """img: uint8 RGB numpy array -> list of per-channel pywt coefficient structures"""
    coeffs_per_channel = []
    for ch in range(img.shape[2]):
        coeffs = pywt.wavedec2(img[:, :, ch].astype(np.float64), wavelet, level=level)
        coeffs_per_channel.append(coeffs)
    return coeffs_per_channel

def dwt_inverse(coeffs_per_channel, original_shape, wavelet='haar', dtype='int'):
    """Inverse DWT -> uint8 RGB numpy array"""
    h, w = original_shape[:2]
    channels = []
    for coeffs in coeffs_per_channel:
        rec = pywt.waverec2(coeffs, wavelet)
        channels.append(rec[:h, :w])
    out = np.stack(channels, axis=2)
    if dtype == 'int':
        return np.clip(out, 0, 255).astype(np.uint8)
    elif dtype == 'float':
        return np.clip(out, 0, 1).astype(np.float32)
    else:
        raise Exception('dtype should be "int" or "float"')

def dwt_bandpass(coeffs_per_channel, keep_levels=None, keep_directions=None, kill_approx=False):
    """Filter wavelet coefficients by decomposition level.
    
    Structure per channel: [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
      - Index 0: approximation (lowest freq)
      - Index 1: coarsest detail level (lowest freq detail)
      - Index -1: finest detail level (highest freq detail)
    
    keep_levels: set of detail level indices (1-indexed from coarse) to keep.
                 e.g. {1,2} keeps the two coarsest detail levels.
                 None keeps all.
    keep_directions: same as above but for (horizontal, vertical, diagonal) directions.
    kill_approx: if True, zero out the approximation coefficients.
    """
    filtered = []
    for c, coeffs in enumerate(coeffs_per_channel):
        n_detail = len(coeffs) - 1
        new_coeffs = []
        # Approximation
        if kill_approx:
            new_coeffs.append(np.zeros_like(coeffs[0]))
        else:
            new_coeffs.append(coeffs[0].copy())
        # Detail levels
        for i in range(1, len(coeffs)):
            if keep_levels is not None and i not in keep_levels: # throw this one away
                # print(f'Zeroing coeffs {c} at detail level {i}')
                new_coeffs.append(tuple(np.zeros_like(c) for c in coeffs[i]))
            else:
                # Directions
                new_level = []
                for j in range(3):
                    if keep_directions is not None and j not in keep_directions:
                        new_level.append(np.zeros_like(coeffs[i][j]))
                    else:
                        new_level.append(coeffs[i][j].copy())
                new_coeffs.append(tuple(new_level))
                # new_coeffs.append(tuple(c.copy() for c in coeffs[i]))
        filtered.append(new_coeffs)
    return filtered

def dwt_freqmult(coeffs_per_channel, which_levels=None, factor=0.5):
    """Multiply wavelet coefficients by decomposition level.
    
    Structure per channel: [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
      - Index 0: approximation (lowest freq)
      - Index 1: coarsest detail level (lowest freq detail)
      - Index -1: finest detail level (highest freq detail)
    
    which_levels: set of detail level indices (1-indexed from coarse) to multiply.
                 e.g. {1,2} multiply the two coarsest detail levels.
                 None does nothing
    factor: the scalar term to multiply each level by. 
    """
    filtered = []
    for c, coeffs in enumerate(coeffs_per_channel):
        # n_detail = len(coeffs) - 1
        new_coeffs = []
        # Detail levels
        for i in range(0, len(coeffs)):
            if which_levels is not None and i in which_levels:
                # print(f'Multiplying coeffs {c} at detail level {i} with {factor}')
                new_coeffs.append(tuple(c.copy()*factor for c in coeffs[i]))
            else:
                new_coeffs.append(tuple(c.copy() for c in coeffs[i]))
        filtered.append(new_coeffs)
    return filtered
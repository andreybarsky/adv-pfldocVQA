# struct Lab {float L; float a; float b;};
# struct RGB {float r; float g; float b;};

import numpy as np


def srgb_to_linear(srgb, ensure_float = True):
    if srgb.max() > 1:
        if ensure_float:
            # quietly cast from int to float
            srgb = srgb / 255.0
        else:
            raise Exception(f'srgb_to_linear expects float-valued input but got: {type(srgb)}')
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(lnr):
    return np.where(lnr <= 0.0031308, lnr * 12.92, 1.055 * lnr ** (1/2.4) - 0.055)

def srgb_to_oklab(rgb_arr):
    linear = srgb_to_linear(rgb_arr)
    
    r, g, b = [linear[:,:,i] for i in range(3)]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    
    l_ = l**( 1/3)
    m_ = m** (1/3)
    s_ = s** (1/3)
    
    return np.stack( [0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_,
                      1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_,
                      0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_], axis=2)


def oklab_to_srgb(lab_arr, clip=True):
    L, a, b = [lab_arr[:,:,i] for i in range(3)]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_*l_*l_
    m = m_*m_*m_
    s = s_*s_*s_

    linear = np.stack([
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], axis=2)    
    
    srgb = linear_to_srgb(linear)
    
    ### in practice some values are very slightly out of image bounds ( <1e-7)
    # overbound = srgb.max() - 1
    # underbound = srgb.min()
    # assert overbound < 1e-7 and underbound < 1e-7
    ### so clip to avoid complaints:
    if clip:
        # assert srgb.max() < 1.05 # throw error if we're dealing with uint scale
        # assert srgb.min > -0.05 # and if we're dealing with zero-centred noise, which is badly behaved in the transition
        srgb = np.clip(srgb, 0, 1)
    return srgb

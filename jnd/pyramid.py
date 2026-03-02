import numpy as np
from scipy import ndimage

def bandlimited_noise(shape, base_sigma, level, magnitude):
    """create gaussian noise of base image resolution,
    but blur to match specific frequency level"""
    noise = np.random.randn(*shape) * magnitude
    if level == 0:
        return noise - gaussian_blur(noise, base_sigma)
    else:
        sigma_low = base_sigma * (2 ** (level - 1))
        sigma_high = base_sigma * (2 ** level)
        return gaussian_blur(noise, sigma_low) - gaussian_blur(noise, sigma_high)

def inject_noise(img, freq_level, noise_magnitude, 
                 pyramid_size=4, base_sigma=0.63, pyramid=None,
                 noise_eps=None):
    if pyramid is None: # create pyramid:
        pyramid = laplacian_pyramid(img, pyramid_size, base_sigma)
    
    # create noise matching specified resolution:
    freq_noise = bandlimited_noise(img.shape, base_sigma, freq_level, noise_magnitude)
    
    if noise_eps is not None:
        freq_noise = np.clip(freq_noise, -noise_eps, noise_eps)
        
    new_level = pyramid[freq_level] + freq_noise
    # subtract old (clean) level from the sum, add the new (noised) one:
    return sum(pyramid) - pyramid[freq_level] + new_level

def gaussian_blur(img, sigma, kernel_size=None):
    if kernel_size is None:
        kernel_size = int(6 * sigma + 1) | 1  # ensure odd
    ax = np.arange(kernel_size) - kernel_size // 2
    kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    
    # Separable: blur rows then columns
    out = np.apply_along_axis(lambda row: np.convolve(row, 
                                  kernel_1d, mode='same'), axis=1, arr=img)
    out = np.apply_along_axis(lambda col: np.convolve(col, 
                                  kernel_1d, mode='same'), axis=0, arr=out)
    return out


def laplacian_pyramid(img, levels=4, base_sigma=1.0):
    pyramid = []
    blurs = [img]
    for i in range(levels-1):
        sigma = base_sigma * (2 ** i)
        blurs.append(gaussian_blur(img, sigma))  # always from original
    
    for i in range(levels-1):
        pyramid.append(blurs[i] - blurs[i + 1])
    pyramid.append(blurs[-1])  # residual
    return pyramid

def get_gaussian_kernel(kernel_shape,sigma):
    
    # initialize impulse response shape
    kernel = np.zeros((kernel_shape[0],kernel_shape[1]))
    
    # compute the indices of the center of the window
    kernel_center = (np.floor(kernel_shape[0]/2).astype(int),
                     np.floor(kernel_shape[1]/2).astype(int))
    
    # set impulse at center
    kernel[kernel_center] = 1
    
    # compute impulse response
    kernel = ndimage.gaussian_filter(kernel,sigma,mode='constant')
    
    # normalize filter such that the center is 1
    kernel = kernel / kernel[kernel_center]
    
    return kernel
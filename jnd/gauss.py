# import the pad_img() function

# from pad_img import pad_img

# import the get_gaussian_kernel() function

# from get_gaussian_kernel import get_gaussian_kernel

# import NumPy library

import numpy as np

# import FFT_2D function

from FFT_2D import FFT_2D

# import iFFT_2D function

from iFFT_2D import iFFT_2D

# import fftpack library

from scipy import fftpack as dsp

# this function performs gaussian blurring of an image.
# It accepts an image as an input and an optional standard deviation.


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

def pad_img(img,pad_size,fill_value=0):
    
    if img.ndim == 2: # grayscale
        
        # initialize new image array filled with fill_value
        
        img_pad = np.full(
                  (img.shape[0]+pad_size[0]+pad_size[1],
                   img.shape[1]+pad_size[2]+pad_size[3]),
                   fill_value,
                   dtype=img.dtype)
        
        # insert the original image into the new image
        
        img_pad[pad_size[0]:img.shape[0]+pad_size[0],
                pad_size[2]:img.shape[1]+pad_size[2]] = img
        
    else: # RGB
        
        # initialize new image array filled with fill_value
        
        img_pad = np.full(
                  (img.shape[0]+pad_size[0]+pad_size[1],
                   img.shape[1]+pad_size[2]+pad_size[3],
                   img.shape[2]),
                   fill_value,
                   dtype=img.dtype)
        
        # insert each channel of the original image into each
        # channel of the new image
        
        for i in range(3):
            
            img_pad[pad_size[0]:img.shape[0]+pad_size[0],
                    pad_size[2]:img.shape[1]+pad_size[2],
                    i] = img[:,:,i]
    
    return img_pad

def gaussian_blur(img,sigma=50):
    
    # define padding size as twice the dimensions of the original image
    # to ensure linear, and not circular, convolution in spatial domain
    
    pad_size = (np.floor(img.shape[0]/2).astype(int),
                np.ceil(img.shape[0]/2).astype(int),
                np.floor(img.shape[1]/2).astype(int),
                np.ceil(img.shape[1]/2).astype(int))
    
    # pad input image with zeros using the pad_img function
    
    padded_img = pad_img(img,pad_size)
    
    # generate the gaussian frequency window to be used for
    # gaussian blur
    
    freq_window = get_gaussian_kernel((padded_img.shape[0],
                                       padded_img.shape[1]),
                                       sigma)
    
    if img.ndim == 2: # grayscale
        
        # perform 2D FFT using hand-made function
        
        padded_img_FFT = FFT_2D(padded_img)
        
        # shift the FFT to center of frequency spectrum
        
        padded_img_FFT = dsp.fftshift(padded_img_FFT)
        
        # this is the same as spatial convolution
        
        G = np.multiply(padded_img_FFT,freq_window)
        
        # shift result back to zero frequency
        
        G = dsp.ifftshift(G)
        
        # perform inverse 2D FFT using hand-made function and then
        # constrain values to the range 0 - 255
        
        g = np.clip(np.abs(iFFT_2D(G)),0,255).astype(np.uint8)
        
        # extract filtered image from zero-padded image
        
        img_out = g[np.floor(padded_img.shape[0]/4).astype(int):
                    np.floor(padded_img.shape[0]/4).astype(int)+
                    np.floor(padded_img.shape[0]/2).astype(int),
                    np.floor(padded_img.shape[1]/4).astype(int):
                    np.floor(padded_img.shape[1]/4).astype(int)+
                    np.floor(padded_img.shape[1]/2).astype(int)]
    
    else: # RGB
        
        # initialize 3-channel arrays to store results
        
        padded_img_FFT = np.zeros(padded_img.shape,dtype=np.complex128)
        
        G = np.zeros(padded_img.shape,dtype=np.complex128)
        
        g = np.zeros_like(padded_img)
        
        img_out = np.zeros_like(img)
        
        # loop for each color channel
        
        for i in range(3):
            
            # perform 2D FFT using hand-made function
            
            padded_img_FFT[:,:,i] = FFT_2D(padded_img[:,:,i])
            
            # shift the FFT to center of frequency spectrum 
            
            padded_img_FFT[:,:,i] = dsp.fftshift(padded_img_FFT[:,:,i])
            
            # this is the same as spatial convolution
            
            G[:,:,i] = np.multiply(padded_img_FFT[:,:,i],freq_window)
            
            # shift result back to zero frequency
            
            G[:,:,i] = dsp.ifftshift(G[:,:,i])
            
            # perform inverse 2D FFT using hand-made function and then
            # constrain values to the range 0 - 255
            
            g[:,:,i] = np.clip(np.abs(
                               iFFT_2D(G[:,:,i])),0,255).astype(np.uint8)
            
            # extract filtered channel from zero-padded channel
            
            img_out[:,:,i] = g[np.floor(padded_img.shape[0]/4).astype(int):
                               np.floor(padded_img.shape[0]/4).astype(int)+
                               np.floor(padded_img.shape[0]/2).astype(int),
                               np.floor(padded_img.shape[1]/4).astype(int):
                               np.floor(padded_img.shape[1]/4).astype(int)+
                               np.floor(padded_img.shape[1]/2).astype(int),
                               i]
    return img_out
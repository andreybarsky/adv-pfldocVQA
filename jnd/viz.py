from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

horiz_px = 1920
vert_px = 1080
diag_inches = 27
diag_px = ((horiz_px**2 + vert_px**2) ** 0.5)

CM_PER_INCH = 2.54
INCH_PER_CM = 1/2.54

### pixels per inch:
calc_dpi = diag_px / diag_inches # (actually ppi)
print(f'Calculated screen DPI: {calc_dpi:.2f}')
calib = {'calc_dpi': calc_dpi}

view_distance_cm = 65 # approximate
view_distance_inches = view_distance_cm * INCH_PER_CM

# global_dpi = rcParams['figure.dpi']




### pixels per degree:
def get_ppd(dpi=calc_dpi):
    ppd = dpi * view_distance_inches * np.tan(np.pi/180)
    print(f'Pixels per degree: {ppd:.2f}')
    return ppd
    

def rshow(img: np.ndarray, 
          title=None, 
          normalised=True, 
          dpi=None, 
          figsize=None, 
          colorbar=False,
          save_as=None, # filepath to save to before showing
          show=True, **kwargs):
    """Function that displays an image at exactly native resolution"""
    if dpi is None:
        if 'plot_dpi' in calib:
            dpi = calib['plot_dpi'] # true corrected value for whatever rescaling browser+jupyter+matplotlib does
        else:
            dpi = calib['calc_dpi'] # estimated value calculated from monitor resolution and physical dimension
    
    if isinstance(img, Image.Image):
        img = np.asarray(img)
        
    if len(img.shape) == 2:
        # add dummy RGB channel:
        img = np.stack([img]*3, axis=2)
        
    if figsize is None:
        # show at native resolution
        h_px, w_px = img.shape[:2]
        
        pad_h = 80  # pixels for title + status text
        # figsize = (WIDTH_PX / MONITOR_DPI, (HEIGHT_PX + pad_h) / MONITOR_DPI)
        figsize = w_px / float(dpi), (h_px+pad_h) / float(dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, pad_h/(h_px+pad_h), 1, h_px/(h_px+pad_h)])        
        
        plt.yticks([])
        # figsize = w_px / float(dpi), (h_px+pad_h) / float(dpi)
        # plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    if normalised:
        img = normalise(img)
    ax.imshow(img, interpolation='none', **kwargs)
    plt.gca().set_title(title)
    if colorbar:
        plt.colorbar()

    if save_as is not None:
        print(f'Figure saved to: {save_as}')
        plt.savefig(save_as)
        
    if show:
        plt.show()
    else:
        return plt.gca()


def rshows(imgs: list, # of np.ndarrays 
          titles: list=None, # of strings
          suptitle: str=None,
          normalised: bool=True, 
          dpi: float=None, 
          figsize: tuple=None, 
          colorbar: bool=False,
          pad_h: int=80,
          pad_w: int=40,
          save_as: bool=None, # filepath to save to before showing    
          show: bool=True, **kwargs):
    """as rshow, but for multiple img inputs in a row
    (for in column/s, just use this or rshow sequentially)"""
    if dpi is None:
        if 'plot_dpi' in calib:
            dpi = calib['plot_dpi'] # true corrected value for whatever rescaling browser+jupyter+matplotlib does
        else:
            dpi = calib['calc_dpi'] # estimated value calculated from monitor resolution and physical dimension
    
    # ensure list of ndarrays, record their :
    img_arrs = []
    img_heights = []
    img_widths = []
    
    for img in imgs:
        
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            
        if normalised:
            img = normalise(img)
            
        if len(img.shape) == 2:
            # add dummy RGB channel:
            img = np.stack([img]*3, axis=2)   
            
        h, w = img.shape[:2]
        img_heights.append(h)
        img_widths.append(w)
        img_arrs.append(img)
            

    
    titles = [None]*len(imgs) if titles is None   else titles
    if figsize is None:
        # show at native resolution
        
        # pad_h = 80  # pixels for title + status text
        # pad_w = 40 # padding between individual plots, and between plots and outer figure margin
        
        total_h = max(img_heights) + pad_h
        total_w = sum(img_widths) + pad_w*(len(imgs)+1)
        
        figh, figw = total_w / float(dpi), total_h / float(dpi) # figure height and width in inches
        
        # fig, axes = plt.subplots(1, len(imgs), figsize=figsize, dpi=dpi)
        fig = plt.figure(figsize=(figh, figw), dpi=dpi)
        figh_, figw_ = fig.get_size_inches()
        # print(f'Creating figure of size: ({figh_:.1f},{figw_:.1f}) inches')
        
        
        for i, (img, title) in enumerate(zip(imgs, titles)):
            # add each image as a new axis:
            
            
            imh, imw = img.shape[:2] # image height and width in pixels
            
            # t = 0
            l = pad_w*(i+1) + sum(img_widths[:i])
            # r = sum(img_widths[:i+1]) + pad_w*(i+1)
            b = total_h - imh - pad_h # top aligned
            # b = total_h - pad_h # bottom aligned
            ax_rect = [l/total_w, b/total_h, imw/total_w, imh/total_h]
            # print(f'Adding axis with dimensions (l/b/w/h): {[np.round(a, 3) for a in ax_rect]}')
            
            assert max(ax_rect) <= 1
            
            ax = fig.add_axes(ax_rect)
            
#                     ax = fig.add_axes([l=0, 
#                                        b=pad_h/(h_px+pad_h), 
#                                        w=1, 
#                                        h= h_px/(h_px+pad_h)])        

            
            ax.imshow(img, interpolation='none', **kwargs)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        
        
    else:
        fig, axes = plt.subplots(1, len(imgs), figsize=figsize)
        
        for img, ax, title in zip(imgs, axes, titles):
            ax.imshow(img, interpolation='none', **kwargs)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        
    plt.suptitle(suptitle)
    if colorbar:
        plt.colorbar()

    if save_as is not None:
        print(f'Figure saved to: {save_as}')
        plt.savefig(save_as)
        
    if show:
        plt.show()
    else:
        return plt.gca()    
    
    
    
def normalise(arr):
    """ normalise an array to range 0-1"""
    arr = arr - arr.min()
    arr = arr / arr.max()
    return arr



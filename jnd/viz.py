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
def get_ppd():
    ppd = calc_dpi * view_distance_inches * np.tan(np.pi/180)
    print(f'Pixels per degree: {ppd:.2f}')
    return ppd
    

def rshow(img: np.ndarray, title=None, normalised=True, dpi=None, figsize=None, colorbar=False, show=True, **kwargs):
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
        figsize = w_px / float(dpi), h_px / float(dpi)
    plt.figure(figsize=figsize)
    if normalised:
        img = normalise(img)
    plt.imshow(img, **kwargs)
    plt.gca().set_title(title)
    if colorbar:
        plt.colorbar()
        
    if show:
        plt.show()
    else:
        return plt.gca()

def normalise(arr):
    """ normalise an array to range 0-1"""
    arr = arr - arr.min()
    arr = arr / arr.max()
    return arr



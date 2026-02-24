from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

horiz_px = 1920
vert_px = 1080
diag_inches = 27
diag_px = ((horiz_px**2 + vert_px**2) ** 0.5)

### pixels per inch:
screen_dpi = diag_px / diag_inches # (actually ppi)
print(f'Calculated screen DPI: {screen_dpi:.2f}')
calib = {'dpi': screen_dpi}

view_distance_cm = 85 # approximate
view_distance_inches = view_distance_cm * 0.393701

# global_dpi = rcParams['figure.dpi']



### pixels per degree:
def get_ppd():
    ppd = screen_dpi * view_distance_inches * np.tan(np.pi/180)
    print(f'Pixels per degree: {ppd:.2f}')
    return ppd
    

def rshow(img: np.ndarray, title=None, normalised=True, dpi=None, figsize=None, colorbar=False, **kwargs):
    if dpi is None:
        dpi = calib['dpi']
    
    if isinstance(img, Image.Image):
        img = np.asarray(img)
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
    plt.show()

def normalise(arr):
    """ normalise an array to range 0-1"""
    arr = arr - arr.min()
    arr = arr / arr.max()
    return arr



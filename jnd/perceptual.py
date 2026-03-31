import torch
import torch.nn.functional as F
import numpy as np
from secmlt.manipulations.manipulation import Manipulation
from secmlt.optimization.constraints import Constraint
from skimage.feature import canny
from scipy.ndimage import distance_transform_edt

from jnd import util as ut

class PerceptualAdditiveManipulation(Manipulation):
    """Additive manipulation in the OKLab perceptual colour space."""

    def __init__(self, *args,  x_clean_rgb=None, **kwargs,):
        """caches the LAB-space clean image from its RGB starting point"""

        super().__init__(*args, **kwargs)

        self.x_clean_rgb = x_clean_rgb
        if x_clean_rgb is not None:
            self.x_clean_lab = srgb_to_oklab(x_clean_rgb, ensure_float=True).detach()           
        else:
            self.x_clean_lab = None
            
    
    def _apply_manipulation(
        self,
        x_rgb: torch.Tensor,     # in RGB space
        delta_lab: torch.Tensor, # in LAB space
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        ## convert x from rgb to lab space:
        # x_lab = srgb_to_oklab(x_rgb)
        assert self.x_clean_lab is not None
        x_lab = self.x_clean_lab

        ## apply lab-space delta:
        x_adv_lab = x_lab + delta_lab

        ## and convert back:
        x_adv_rgb = oklab_to_srgb(x_adv_lab, clip=False) * 255. # back to int space

        ## domain constraint is applied here for gradient stability:
        x_adv_rgb = torch.clamp(x_adv_rgb, 0., 255.)
        
        return x_adv_rgb, delta_lab

    
    def __call__(
        self,
        x_rgb: torch.Tensor,
        lab_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the additive manipulation to the input data.

        Parameters
        ----------
        x_rgb : torch.Tensor
            Input data. (in RGB space)
        lab_delta : torch.Tensor
            Perturbation to apply. (in LAB space)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Perturbed data (RGB) and perturbation (LAB) after the
            application of constraints.
        """

        # perturbation constraint in LAB space:        
        lab_delta.data = self._apply_perturbation_constraints(lab_delta)


        # perceptual manipulation in LAB space: (converts x through lab and back)
        x_adv_rgb, lab_delta = self._apply_manipulation(x_rgb, lab_delta)
        # x_adv_rgb: [2200, 1700, 3]
        # lab_delta: [1, 2200, 1700, 3]

        # domain constraint in RGB space:
        print(f'x_adv_rgb quantiles before domain constraint: {ut.num_quantiles(x_adv_rgb)}')

        print(f'Applying PerceptualAdditiveManipulation domain constraints: {self._domain_constraints}')

        x_adv_rgb.data = self._apply_domain_constraints(x_adv_rgb.data)
        print(f'x_adv_rgb quantiles after domain constraint: {ut.num_quantiles(x_adv_rgb)}')

        ### compute delta in rgb space:
        rgb_delta = x_adv_rgb.data - self.x_clean_rgb.data

        
        # get the domain-constrained delta in LAB space to pass back to the optimiser:
        lab_delta.data = self._get_constrained_delta(x_adv_rgb) # x_rgb


        # here we have 388 quantiles and not 256! why??
        
        # print(f'Output of PerceptualManipulation x_adv_rgb has {ut.num_quantiles(x_adv_rgb)} quantiles')
        

        return x_adv_rgb, lab_delta

    def _get_constrained_delta(self, x_adv_rgb, 
                               # x_clean_rgb,
                              ):
        """Get the values of RGB-domain-constrained delta in LAB space
        from the difference between the clean and adversarial inputs"""

        x_adv_lab = srgb_to_oklab(x_adv_rgb, ensure_float=True)
        # x_clean_lab = srgb_to_oklab(x_clean_rgb)

        assert self.x_clean_lab is not None
        actual_delta_lab = x_adv_lab.data - self.x_clean_lab.data

        return actual_delta_lab.unsqueeze(0) # preserve dummy batch dimension

### cloned from secmlt.manipulations.manipulation
class AdditiveManipulation(Manipulation):
    """Additive manipulation for input data."""

    def _apply_manipulation(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x + delta, delta

    def __call__(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the additive manipulation to the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        delta : torch.Tensor
            Perturbation to apply.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Perturbed data and perturbation after the
            application of constraints.
        """
        # x_adv, delta = super().__call__(x, delta)

        # cloned from superclass Manipulation.__call__
        # print(f'\ndelta quantiles before perturbation constraint: {ut.num_quantiles(delta)}')
        delta.data = self._apply_perturbation_constraints(delta.data)
        # print(f'delta quantiles after perturbation constraint: {ut.num_quantiles(delta)}\n')



        x_adv, delta = self._apply_manipulation(x, delta)
        print(f'x_adv quantiles before domain constraint: {ut.num_quantiles(x_adv)}')

        print(f'Applying AdditiveManipulation domain constraints: {self._domain_constraints}')

        x_adv.data = self._apply_domain_constraints(x_adv.data)
        print(f'x_adv quantiles after domain constraint: {ut.num_quantiles(x_adv)}\n')
        
        delta.data = x_adv.data - x.data
        print(f'delta quantiles after domain constraint: {ut.num_quantiles(delta)}')
        return x_adv, delta





class PerPixelSymmetricConstraint(Constraint):
    """Clips the perturbation to have magnitude no greater than 
    the values of the (float) mask. """

    def __init__(self, mask: torch.Tensor, eps:float) -> None:
        """
        Create the PerPixelConstraint as a mask of the same size as the input.

        Parameters
        ----------
        mask : torch.Tensor (must be non-negative)
        """

        if mask.min() < 0.:
            msg = (
                f"PerPixelConstraint mask must be non-negative everywhere."
            )
            raise ValueError(msg)
        elif mask.max() > 1:
            print(f'Warning: Soft mask has values greater than 1, some delta values may exceed eps:{eps}')

        
        self.raw_mask = mask 
        self.eps = eps

        # instead of a global epsilon, we use a per-pixel epsilon based on the mask values
        # (which we assume range between [0,1]
        self.eps_mask = self.raw_mask * eps
        
        super().__init__()

    def _apply_constraint(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enforce the mask constraint.

        Parameters
        ----------
        x : torch.Tensor
            Masked input tensor.

        Returns
        -------
        torch.Tensor
            Input active only on the non-masked components.
        """
        if self.eps_mask.shape != x.squeeze().shape:
            msg = (
                f"Shape of input ({x.shape}) and mask {self.eps_mask.shape} does not match."
            )
            raise ValueError(msg)

        # return torch.clamp_(x, -self.mask, self.mask)

        # pre_l1 = ut.l1(x)
        # pre_q = ut.num_quantiles(x)

        x = x.clamp(min=-self.eps_mask, max=self.eps_mask)

        # post_l1 = ut.l1(x)
        # post_q = ut.num_quantiles(x)

        # print(f'Perturbation l1 before clamp: {pre_l1:.6f} (quantiles: {pre_q})')
        # print(f'Perturbation l1 after clamp:  {post_l1:.6f} (quantiles: {post_q})')

        return x



#### torch functions for LAB space conversion:

def srgb_to_linear(srgb: torch.Tensor, # expected in range 0-1 
                   ensure_float = False):
    if srgb.max() > 1:
        if ensure_float:
            # quietly cast from int to float
            srgb = srgb / 255.0
        else:
            raise Exception(f'srgb_to_linear expects float-valued input but got: {type(srgb)}')
    return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(lrgb):
    return torch.where(lrgb <= 0.0031308, lrgb * 12.92, 1.055 * lrgb ** (1/2.4) - 0.055)

def srgb_to_oklab(srgb: torch.Tensor, # as [B]CHW representation, in range 0-1
                 ensure_float = False,):
    lrgb = srgb_to_linear(srgb, ensure_float)

    if len(lrgb.shape) == 4:
        assert lrgb.shape[0] == 1 # dummy batch dimension
        lrgb = lrgb.squeeze()
        squeezed = True
    else:
        squeezed = False
    
    # r, g, b = [lrgb[:,i,:,:] for i in range(3)] # BCHW
    r, g, b = [lrgb[:,:,i] for i in range(3)] # HWC

    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    
    l_ = l**( 1/3)
    m_ = m** (1/3)
    s_ = s** (1/3)

    okl = torch.stack( [0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_,
                      1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_,
                      0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_], axis=2) # stack across channel dimension (BCHW=1, HWC=2)
    # if squeezed:
    #     okl = okl.unsqueeze(0) # add batch dimension back in
    return okl


def oklab_to_srgb(lab: torch.Tensor, # as BCHW representation
                  clip: bool=False):
    # L, a, b = [lab[:,i,:,:] for i in range(3)] # BHCW

    if len(lab.shape) == 4:
        assert lab.shape[0] == 1 # dummy batch dimension
        lab = lab.squeeze()
        squeezed = True
    else:
        squeezed = False
    
    L, a, b = [lab[:,:,i] for i in range(3)] # HWC
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_*l_*l_
    m = m_*m_*m_
    s = s_*s_*s_

    lrgb = torch.stack([
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], axis=2) # stack across channel dimension (BCHW=1, HWC=2)
    
    srgb = linear_to_srgb(lrgb)
    
    if clip:
        print(f'Clipping from range [{srgb.min().item():.8f}, {srgb.max().item():.8f}]')
        if (srgb.min() < -0.1) or (srgb.max() > 1.1):
            # this shouldn't happen
            print(f'  Warning: SRGB gamut out of bounds with min:{srgb.min():.2f} / max:{srgb.max():.3f}')
        srgb = torch.clamp_(srgb, 0, 1)

    # if squeezed:
    #     srgb = srgb.unsqueeze(0) # add batch dimension back in
    return srgb


def sobel(img: torch.Tensor):
    # img: HWC float tensor, returns HW edge magnitude
    
    assert len(img.shape) == 3 # assume single image
    assert img.shape[2] <= 3   # assume RGB

    if img.max() > 1: 
        img = img / 255. # calculate in float range
    
    gray = img.mean(-1, keepdim=True).permute(2, 0, 1).unsqueeze(0)  # 1,1,H,W (grayscale conversion)

    # sobel kernels:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    ky = kx.transpose(-1, -2)

    # horizontal and vertical edge maps:
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    edges = (gx.square() + gy.square()).sqrt().squeeze()
    # normalise to range [0,1]:
    return edges / (4*(2**0.5))

def blurred_sobel(img: torch.Tensor, constant=256):
    """as above, but performs a small pre-blur to minimise high-frequency edges due to things like jpeg artifacting"""
    # make gaussian kernel, scaled appropriately to image dimensions:
    imh, imw = img.shape[:-1]
    max_dim = max(imh, imw)
    sigma = max_dim / constant  # with 512 it's ~1.0 at 512px, ~3.75 at 1920px
    print(f'Running blurred sobel on img with shape: {img.shape} and gaussian sigma: {sigma:.2f}')
    
    gauss = make_gaussian_kernel(sigma=np.clip(sigma, 0.5, 3.0))
    gray = img.mean(-1, keepdim=True).permute(2, 0, 1).unsqueeze(0)  # 
    img_blur = F.conv2d(gray, gauss, padding=gauss.shape[-1] // 2)
    return sobel(img_blur.squeeze().unsqueeze(-1))

def robust_sobel(img: torch.Tensor, # of shape [h,w,c]
                 scales=[1, 2, 3], # not exclusively powers of 2 to avoid jpeg artifacts (with 8x8 blocks)
                 sqrt=False,
                 aggregation='geometric',
                 return_maps=False,
                ):
    """Returns edges that are consistent across multiple image scales."""
    # img: HWC float tensor
    if len(img.shape) == 2:
        # add missing C dimension if needed
        img = img.unsqueeze(-1)
    assert img.shape[2] in (1,3)
    assert aggregation in ('geometric', 'arithmetic', 'harmonic', 'median', 'min')
    h, w = img.shape[:2]
    edge_maps = []

    print(f'Running robust sobel on img with shape: {img.shape} and scales: {scales}')
    
    for s in scales:
        if s == 1:
            edges = sobel(img)
        else:
            # downsample, run sobel, upsample back
            small = F.interpolate(
                img.permute(2, 0, 1).unsqueeze(0),
                size=(h // s, w // s),
                mode='bilinear',
                align_corners=False,
            )
            edges = sobel(small.squeeze(0).permute(1, 2, 0))
            edges = F.interpolate(
                edges.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            ).squeeze()


        edge_maps.append(edges)
    if return_maps:
        # just give the list of edge maps at each scale
        return edge_maps
    else:
        stacked = torch.stack(edge_maps)

        # aggregate maps according to chosen operator:
        print(f'Using multiscale aggregation={aggregation}')
        if aggregation == 'geometric':
            # smooth and continuous, pretty harsh, edges must be present at all scales to survive
            edges = stacked.prod(dim=0).pow(1.0 / len(scales))
        elif aggregation == 'arithmetic':
            # most forgiving, strongest scale can compensate for all others (prob. sensitive to jpeg artefacts)
            edges = stacked.mean(dim=0)
        elif aggregation == 'harmonic':
            # less harsh than geometric, less lenient than arithmetic, biased toward smaller values
            edges = len(scales) / (1/stacked).sum(dim=0)

        elif aggregation == 'median':
            # robust to outliers and prioritises consensus, but should have an odd number of scales to work properly
            edges = stacked.median(dim=0).values    
        elif aggregation == 'min':
            # harshest consolidation, weakest scale determines everything
            edges = stacked.min(dim=0).values
        
        if sqrt:
            print(f'Applying sqrt scaling')
            # sqrt the values so that chromatic edges don't disappear:
            edges = edges ** 0.5
        return edges 

def make_gaussian_kernel(sigma, size=None):
    if size is None:
        size = int(2 * round(2 * sigma) + 1)  # covers ~2sd each side
        print(f'  and size={size}')
    x = torch.arange(size, dtype=torch.float32) - size // 2
    k = (-x.square() / (2 * sigma**2)).exp()
    k = k / k.sum()
    return (k[:, None] * k[None, :]).view(1, 1, size, size)

def nms_edges(img_channel, sigma=1.0, width=2, low=0.01, high=0.3):
    """img_channel: (H, W) numpy array, single channel"""
    dev = img_channel.device
    img_channel = img_channel.detach().cpu().numpy()
    edges = canny(img_channel, sigma=sigma, low_threshold=low, high_threshold=high)
    dist = distance_transform_edt(~edges)
    return torch.tensor([(dist <= width).astype(float)]).float().squeeze().to(dev)

def robust_canny(img_channel: torch.Tensor, # of shape [h,w]
                 scales=[1,2,3],
                 sigma=1.0,
                 width=1,
                low=0.1,
                high=0.2,
                return_maps=False):
        
    edge_maps = []
    h, w = img_channel.shape[:2]

    print(f'Running robust canny on img with shape: {img_channel.shape}, line width={width}px and scales: {scales}')

    assert len(img_channel.shape) == 2
    # assert img_channel.max() <= 1.0
    # assert img_channel.shape[-1] == 1

    for s in scales:
        if s == 1:
            edges = nms_edges(img_channel, sigma, width, low, high)
        else:
            # downsample, run sobel, upsample back
            small = F.interpolate(
                # img_channel.permute(2, 0, 1).unsqueeze(0),
                img_channel.unsqueeze(0).unsqueeze(0),
                size=(h // s, w // s),
                mode='bilinear',
                align_corners=False,
            ).squeeze()
            edges = nms_edges(small, sigma, width, low, high) #.squeeze(0).permute(1, 2, 0))
            edges = F.interpolate(
                edges.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            ).squeeze()

        edge_maps.append(edges)

    if not return_maps:
        stacked = torch.stack(edge_maps)
        # edges = stacked.min(dim=0).values
        edges = stacked.prod(dim=0).pow(1.0 / len(scales))
     
        return edges    
    else:
        return edge_maps
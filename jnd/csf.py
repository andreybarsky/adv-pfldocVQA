import numpy as np

### Movshon & Kiorpes form (used in Johnson & Fairchild 2003)
def lum_csf(f): 
    """Bandpass luminance CSF. f in cycles/degree."""
    a, b, c = 75, 0.2, 0.8
    return a * f * np.exp(-(b * f) ** c)

### From Mullen (1985)
# def rg_csf(f):
#     """Low-pass red-green CSF. f in cycles/degree."""
#     return 25 * np.exp(-(f / 1.2) ** 1.5)

# def yb_csf(f):
#     """Low-pass yellow-blue CSF. f in cycles/degree."""  
#     return 12 * np.exp(-(f / 0.8) ** 1.5)


# revised constants based on Mantiuk plot:
def rg_csf(f):
    return 25 * np.exp(-(f / 4.0) ** 2)

def yb_csf(f):
    return 12 * np.exp(-(f / 3.5) ** 2)





###### mantiuk-based oklab thresholds from claude:


#     sens = oklab_sensitivity(freq)
#     thresh = oklab_threshold(freq)
#     budget = oklab_relative_budget(freq)
    
def lum_csf_oklab(f):
    return list(oklab_sensitivity(f).values())[0]

def rg_csf_oklab(f):
    return list(oklab_sensitivity(f).values())[1]

def yb_csf_oklab(f):
    return list(oklab_sensitivity(f).values())[2]

def lum_budget_oklab(f):
    return list(oklab_relative_budget(f).values())[0]

def rg_budget_oklab(f):
    return list(oklab_relative_budget(f).values())[1]

def yb_budget_oklab(f):
    return list(oklab_relative_budget(f).values())[2]

"""
oklab_perceptual_threshold — Cross-channel perceptual sensitivity for Oklab perturbations.

Uses castleCSF to compute how visible a unit perturbation in each Oklab axis
(L, a, b) is at a given spatial frequency. Applies a physiologically-motivated
chromatic cutoff above ~10-12 cpd, where the castleCSF training data is sparse
and likely contaminated by luminance artifacts.

Usage:
    from oklab_perceptual_threshold import oklab_threshold, oklab_relative_budget

    thresh = oklab_threshold(4.0)
    # {'L': 0.00093, 'a': 0.0083, 'b': 0.0082}
    # At 4 cpd: a and b have ~9x more budget than L

    thresh = oklab_threshold(20.0)
    # {'L': 0.0042, 'a': very_large, 'b': very_large}
    # At 20 cpd: chromatic channels are essentially invisible

    budget = oklab_relative_budget(4.0)
    # {'L': 1.0, 'a': 8.9, 'b': 8.8}

Depends on castle_csf.py (must be importable).
"""

import numpy as np
from castle_csf import (csf_achromatic, csf_rg, csf_yv,
                         LUMINANCE, T_FREQUENCY, AREA, ECCENTRICITY, VIS_FIELD)

# ============================================================================
# CHROMATIC CUTOFF PARAMETERS
# ============================================================================

# Above this frequency, chromatic sensitivity drops to zero.
# The visual system physically cannot resolve chromatic gratings above ~10-12 cpd
# due to sparse S-cone mosaic and pre-neural optical blur.
# The castleCSF model does not enforce this because its training data contains
# high-frequency chromatic measurements that are likely contaminated by
# luminance artifacts (van der Horst 1969, etc).
#
# Set to None to disable (use raw castleCSF predictions).
CHROMATIC_CUTOFF_CPD = 12.0

# Sharpness of the cutoff (in octaves). Higher = sharper transition.
# 2.0 means sensitivity drops by ~10x over 1 octave above the cutoff.
CHROMATIC_CUTOFF_STEEPNESS = 4.0

# ============================================================================
# COLOR SPACE TRANSFORMS
# ============================================================================

_M_OKLAB_TO_LMS_CBRT = np.array([
    [1.0,  0.3963377774,  0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
])

_M_SRGB_TO_LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])

# castleCSF mechanism matrix: LMS -> [Achromatic, RG, YV]
_M_LMS2ACC = np.array([
    [ 1.0,  1.0,  2.3112],
    [ 1.0, -1.0,  0.0   ],
    [-1.0, -1.0, 50.9875],
])

# ============================================================================
# BACKGROUND COLOR
# ============================================================================

BACKGROUND_SRGB = np.array([0.5, 0.5, 0.5])


def _srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _get_background_lms(srgb=None):
    if srgb is None:
        srgb = BACKGROUND_SRGB
    return _M_SRGB_TO_LMS @ _srgb_to_linear(srgb)


def _oklab_to_lms_jacobian(lms_bkg):
    """Jacobian d(LMS)/d(Oklab) at a given background."""
    cbrt = np.cbrt(lms_bkg)
    return np.diag(3.0 * cbrt ** 2) @ _M_OKLAB_TO_LMS_CBRT


# ============================================================================
# CHROMATIC CUTOFF
# ============================================================================

def _chromatic_cutoff_factor(freq):
    """Multiplicative factor that suppresses chromatic sensitivity above cutoff.

    Returns 1.0 below cutoff, decays to 0 above. Applied to chromatic
    channel sensitivities (S_R, S_Y) before combining.
    """
    if CHROMATIC_CUTOFF_CPD is None:
        return np.ones_like(np.asarray(freq, dtype=np.float64))

    freq = np.asarray(freq, dtype=np.float64)
    # Half-Gaussian rolloff above cutoff in log-frequency
    log_ratio = np.log2(np.maximum(freq, 1e-10) / CHROMATIC_CUTOFF_CPD)
    factor = np.where(
        log_ratio <= 0,
        1.0,
        np.exp(-0.5 * (log_ratio * CHROMATIC_CUTOFF_STEEPNESS) ** 2)
    )
    return factor


# ============================================================================
# MECHANISM CONTRAST + THRESHOLD COMPUTATION
# ============================================================================

def _csf_chrom_directions(lms_bkg, lms_delta):
    """Compute mechanism contrasts C_A, C_R, C_Y."""
    ACC_mean = np.abs(_M_LMS2ACC @ lms_bkg)
    ACC_delta = np.abs(_M_LMS2ACC @ lms_delta)

    C_A = ACC_delta[0] / ACC_mean[0]
    C_R = ACC_delta[1] / ACC_mean[0]
    C_Y = ACC_delta[2] / ACC_mean[0]

    return C_A, C_R, C_Y


def _csf_chrom_directions_freq(lms_bkg, lms_delta, freq):
    """Compute frequency-dependent mechanism contrasts.

    Below the chromatic cutoff, uses the full castleCSF mechanism matrix
    (L+M+2.31S achromatic). Above the cutoff, fades the S-cone weight
    in the achromatic mechanism to zero, since S-cones do not contribute
    to luminance detection at high spatial frequencies (the S-cone mosaic
    is too sparse to resolve high-frequency luminance gratings).

    This fixes the issue where the castleCSF's fitted S-cone weight in the
    achromatic mechanism (2.3112) causes Oklab-b perturbations to appear
    highly visible at all frequencies via the achromatic pathway.
    """
    freq = np.asarray(freq, dtype=np.float64)
    chrom_factor = _chromatic_cutoff_factor(freq)

    # Interpolate S-cone weight: 2.3112 at low freq -> 0 above cutoff
    s_weight = 2.3112 * chrom_factor

    # Build per-frequency mechanism matrix (only achromatic row changes)
    # Achromatic: L + M + s_weight*S
    ACC_mean_ach = np.abs(lms_bkg[0] + lms_bkg[1] + s_weight * lms_bkg[2])
    ACC_delta_ach = np.abs(lms_delta[0] + lms_delta[1] + s_weight * lms_delta[2])

    # RG and YV are unchanged
    ACC_mean_static = np.abs(_M_LMS2ACC @ lms_bkg)
    ACC_delta_static = np.abs(_M_LMS2ACC @ lms_delta)

    C_A = ACC_delta_ach / np.maximum(ACC_mean_ach, 1e-30)
    C_R = ACC_delta_static[1] / np.maximum(ACC_mean_ach, 1e-30)
    C_Y = ACC_delta_static[2] / np.maximum(ACC_mean_ach, 1e-30)

    return C_A, C_R, C_Y


def _det_threshold(lms_bkg, lms_delta, freq, **kwargs):
    """Detection threshold k such that k * lms_delta is just visible.

    Applies chromatic cutoff to RG/YV sensitivities AND fades S-cone
    weight in the achromatic mechanism above the cutoff frequency.
    """
    C_A, C_R, C_Y = _csf_chrom_directions_freq(lms_bkg, lms_delta, freq)

    S_A = csf_achromatic(freq, **kwargs)
    S_R = csf_rg(freq, **kwargs)
    S_Y = csf_yv(freq, **kwargs)

    # Apply chromatic cutoff to channel sensitivities
    chrom_factor = _chromatic_cutoff_factor(freq)
    S_R = S_R * chrom_factor
    S_Y = S_Y * chrom_factor

    beta = 2.0
    C = (np.abs(C_A * S_A) ** beta +
         np.abs(C_R * S_R) ** beta +
         np.abs(C_Y * S_Y) ** beta) ** (1.0 / beta)

    # Avoid division by zero
    C = np.maximum(C, 1e-30)
    k = 1.0 / C
    return k


# ============================================================================
# PUBLIC API
# ============================================================================

def oklab_sensitivity(freq, background_srgb=None, luminance=None,
                      t_frequency=None, area=None, eccentricity=None,
                      vis_field=None):
    """Perceptual sensitivity to unit perturbations in each Oklab axis.

    Higher value = more visible = need smaller perturbation budget.
    Values are directly comparable across channels at any given frequency.
    """
    lms_bkg = _get_background_lms(background_srgb)
    J = _oklab_to_lms_jacobian(lms_bkg)

    kwargs = {}
    if luminance is not None: kwargs['luminance'] = luminance
    if t_frequency is not None: kwargs['t_frequency'] = t_frequency
    if area is not None: kwargs['area'] = area
    if eccentricity is not None: kwargs['eccentricity'] = eccentricity
    if vis_field is not None: kwargs['vis_field'] = vis_field

    k_L = _det_threshold(lms_bkg, J[:, 0], freq, **kwargs)
    k_a = _det_threshold(lms_bkg, J[:, 1], freq, **kwargs)
    k_b = _det_threshold(lms_bkg, J[:, 2], freq, **kwargs)

    return {'L': 1.0 / k_L, 'a': 1.0 / k_a, 'b': 1.0 / k_b}


def oklab_threshold(freq, background_srgb=None, **kwargs):
    """Detection threshold in Oklab units per axis.

    Higher value = less visible = more perturbation budget.
    Directly usable as per-channel epsilon.
    """
    lms_bkg = _get_background_lms(background_srgb)
    J = _oklab_to_lms_jacobian(lms_bkg)

    csf_kwargs = {k: v for k, v in kwargs.items()
                  if k in ('luminance', 't_frequency', 'area', 'eccentricity', 'vis_field')}

    k_L = _det_threshold(lms_bkg, J[:, 0], freq, **csf_kwargs)
    k_a = _det_threshold(lms_bkg, J[:, 1], freq, **csf_kwargs)
    k_b = _det_threshold(lms_bkg, J[:, 2], freq, **csf_kwargs)

    return {'L': k_L, 'a': k_a, 'b': k_b}


def oklab_relative_budget(freq, background_srgb=None, **kwargs):
    """Relative perturbation budget per Oklab channel, normalized so L=1."""
    thresh = oklab_threshold(freq, background_srgb, **kwargs)
    t_L = np.maximum(thresh['L'], 1e-30)
    return {
        'L': np.ones_like(np.asarray(freq, dtype=np.float64)),
        'a': thresh['a'] / t_L,
        'b': thresh['b'] / t_L,
    }


# ============================================================================
# DEMO
# ============================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    freq = np.logspace(np.log10(0.125), np.log10(50), 500)

    sens = oklab_sensitivity(freq)
    thresh = oklab_threshold(freq)
    budget = oklab_relative_budget(freq)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    ax.loglog(freq, sens['L'], 'k-', lw=2, label='Oklab L (lightness)')
    ax.loglog(freq, sens['a'], 'r-', lw=2, label='Oklab a (red-green)')
    ax.loglog(freq, sens['b'], 'b-', lw=2, label='Oklab b (yellow-blue)')
    ax.axvline(x=CHROMATIC_CUTOFF_CPD, color='gray', ls=':', alpha=0.6,
               label=f'chromatic cutoff ({CHROMATIC_CUTOFF_CPD} cpd)')
    ax.set_xlabel('Spatial frequency (cpd)')
    ax.set_ylabel('Sensitivity (1/threshold)')
    ax.set_title('Perceptual sensitivity per Oklab axis')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    ax.loglog(freq, thresh['L'], 'k-', lw=2, label='Oklab L')
    ax.loglog(freq, thresh['a'], 'r-', lw=2, label='Oklab a')
    ax.loglog(freq, thresh['b'], 'b-', lw=2, label='Oklab b')
    ax.axvline(x=CHROMATIC_CUTOFF_CPD, color='gray', ls=':', alpha=0.6)
    ax.set_xlabel('Spatial frequency (cpd)')
    ax.set_ylabel('Detection threshold (Oklab units)')
    ax.set_title('Per-axis detection threshold\n(= max invisible perturbation)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[2]
    ax.semilogy(freq, budget['L'], 'k-', lw=2, label='Oklab L')
    ax.semilogy(freq, budget['a'], 'r-', lw=2, label='Oklab a')
    ax.semilogy(freq, budget['b'], 'b-', lw=2, label='Oklab b')
    ax.axvline(x=CHROMATIC_CUTOFF_CPD, color='gray', ls=':', alpha=0.6)
    ax.axhline(y=1, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel('Spatial frequency (cpd)')
    ax.set_ylabel('Relative budget (L = 1)')
    ax.set_title('Relative perturbation budget\n(higher = more room to perturb)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('oklab_perceptual_budget.png', dpi=150)
    plt.close()

    # Table
    print(f"{'freq':>6s}  {'thresh_L':>10s}  {'thresh_a':>10s}  {'thresh_b':>10s}  | {'bgt_a':>7s}  {'bgt_b':>7s}")
    print('-' * 68)
    test_freqs = [0.5, 1, 2, 4, 8, 12, 16, 20, 30, 40]
    for f in test_freqs:
        t = oklab_threshold(f)
        b = oklab_relative_budget(f)
        print(f"{f:6.1f}  {t['L']:10.5f}  {t['a']:10.5f}  {t['b']:10.5f}  | {b['a']:7.1f}  {b['b']:7.1f}")
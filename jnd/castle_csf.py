"""
castleCSF — Python port of the Colour, Area, Spatial frequency, Temporal frequency,
Luminance, Eccentricity dependent Contrast Sensitivity Function.

Ported from the MATLAB implementation at:
    https://github.com/gfxdisp/castleCSF

Reference:
    Ashraf, M., Mantiuk, R. K., Chapiro, A., & Wuerger, S. (2024).
    castleCSF — A contrast sensitivity function of color, area, spatiotemporal
    frequency, luminance and eccentricity. Journal of Vision, 24(4), 5–5.

Usage:
    import numpy as np
    from castle_csf import csf_achromatic, csf_rg, csf_yv

    freq = np.logspace(-1, np.log10(60), 200)  # cpd
    S_ach = csf_achromatic(freq)
    S_rg  = csf_rg(freq)
    S_yv  = csf_yv(freq)
"""

import numpy as np

# ============================================================================
# VIEWING CONDITIONS — change these as needed
# ============================================================================

LUMINANCE = 100.0        # Background luminance in cd/m^2
T_FREQUENCY = 0.0        # Temporal frequency in Hz (0 = static image)
AREA = 1.0               # Stimulus area in deg^2 (1 deg^2 ≈ Gabor with sigma=0.564°)
ECCENTRICITY = 0.0       # Retinal eccentricity in degrees (0 = foveal)
VIS_FIELD = 0.0          # Visual field orientation in degrees (0 = temporal)


# ============================================================================
# LUMINANCE-DEPENDENT PARAMETER FUNCTION
# ============================================================================

def _get_lum_dep(pars, L):
    """Compute luminance-dependent parameter value.

    Dispatches based on the number of parameters (matches MATLAB CSF_base.get_lum_dep).
        1 param:  constant
        2 params: power law  p[1] * L^p[0]
        3 params: single hyperbolic  p[0] * (1 + p[1]/L)^(-p[2])
        5 params: two hyperbolic  p[0]*(1+p[1]/L)^(-p[2]) * (1-(1+p[3]/L)^(-p[4]))
    """
    L = np.asarray(L, dtype=np.float64)
    n = len(pars)
    if n == 1:
        return np.full_like(L, pars[0])
    elif n == 2:
        return pars[1] * L ** pars[0]
    elif n == 3:
        return pars[0] * (1.0 + pars[1] / L) ** (-pars[2])
    elif n == 5:
        return (pars[0] * (1.0 + pars[1] / L) ** (-pars[2])
                * (1.0 - (1.0 + pars[3] / L) ** (-pars[4])))
    else:
        raise ValueError(f"get_lum_dep: unsupported parameter count {n}")


# ============================================================================
# FITTED PARAMETERS (from CSF_castleCSF.get_default_par)
# ============================================================================

# --- Achromatic: sustained channel ---
_ACH_SUST = dict(
    S_max=[56.4947, 7.54726, 0.144532, 5.58341e-07, 9.66862e+09],
    f_max=[1.78119, 91.5718, 0.256682],
    bw=0.000213047,
    a=0.100207,
    A_0=157.103,
    f_0=0.702338,
)

# --- Achromatic: transient channel ---
_ACH_TRANS = dict(
    S_max=[0.193434, 2748.09],
    f_max=0.000316696,  # scalar → constant across luminance
    bw=2.6761,
    a=0.000241177,
    A_0=3.81611,
    f_0=3.01389,
)

# --- Achromatic: temporal & eccentricity ---
_ACH_TEMP = dict(
    sigma_sust=10.5795,
    sigma_trans=0.0844836,
    omega_trans_sl=2.41482,
    omega_trans_c=4.7036,
    ecc_drop=0.0239853,
    ecc_drop_nasal=0.0400662,
    ecc_drop_f=0.0189038,
    ecc_drop_f_nasal=0.00813619,
)

# --- Red-Green chromatic ---
_RG = dict(
    S_max=[681.434, 38.0038, 0.480386],
    f_max=0.0178364,
    bw=2.42104,
    A_0=2816.44,
    f_0=0.0711058,
    sigma_sust=16.4325,
    beta_sust=1.15591,
    ecc_drop=0.0591402,
    ecc_drop_nasal=2.89615e-05,
    ecc_drop_f=2.04986e-69,
    ecc_drop_f_nasal=0.18108,
)

# --- Yellow-Violet chromatic ---
_YV = dict(
    S_max=[166.683, 62.8974, 0.41193],
    f_max=0.00425753,
    bw=2.68197,
    A_0=2.82789e+07,
    f_0=0.000635093,
    sigma_sust=7.15012,
    beta_sust=0.969123,
    ecc_drop=0.00356865,
    ecc_drop_nasal=5.85804e-141,
    ecc_drop_f=0.00806631,
    ecc_drop_f_nasal=0.0110662,
)


# ============================================================================
# CORE CSF EQUATIONS
# ============================================================================

def _csf_achrom_channel(freq, area, lum, ch_pars):
    """Achromatic CSF for one channel (sustained or transient).

    Truncated log-parabola × Rovamo area summation model.
    """
    freq = np.asarray(freq, dtype=np.float64)
    lum = np.asarray(lum, dtype=np.float64)

    S_max = _get_lum_dep(ch_pars['S_max'], lum)
    f_max_pars = ch_pars['f_max']
    if np.isscalar(f_max_pars) or (isinstance(f_max_pars, (list, np.ndarray)) and len(f_max_pars) == 1):
        f_max = np.atleast_1d(f_max_pars)[0] * np.ones_like(lum)
    else:
        f_max = _get_lum_dep(f_max_pars, lum)
    bw = ch_pars['bw']
    a = ch_pars['a']

    # Truncated log-parabola (achromatic version: flat plateau = 1-a below f_max)
    S_LP = 10.0 ** (-(np.log10(freq) - np.log10(f_max)) ** 2 / (2.0 ** bw))
    S_LP = np.where((freq < f_max) & (S_LP < (1.0 - a)), 1.0 - a, S_LP)

    S_peak = S_max * S_LP

    # Rovamo et al. (1993) area summation
    f0 = ch_pars['f_0']
    A0 = ch_pars['A_0']
    Ac = A0 / (1.0 + (freq / f0) ** 2)
    S = S_peak * np.sqrt(Ac / (1.0 + Ac / area)) * freq

    return S


def _csf_chrom_channel(freq, area, lum, ch_pars, f_0, A_0):
    """Chromatic CSF (RG or YV).

    Truncated log-parabola (low-pass: flat below f_max) × area summation.
    """
    freq = np.asarray(freq, dtype=np.float64)
    lum = np.asarray(lum, dtype=np.float64)

    S_max = _get_lum_dep(ch_pars['S_max'], lum)
    f_max_pars = ch_pars['f_max']
    if np.isscalar(f_max_pars):
        f_max = f_max_pars * np.ones_like(lum)
    else:
        f_max = _get_lum_dep(f_max_pars, lum)
    bw = ch_pars['bw']

    # Truncated log-parabola (chromatic version: flat = 1.0 below f_max)
    S_LP = 10.0 ** (-np.abs(np.log10(freq) - np.log10(f_max)) ** 2 / (2.0 ** bw))
    S_LP = np.where(freq < f_max, 1.0, S_LP)

    S_peak = S_max * S_LP

    # Area summation
    Ac = A_0 / (1.0 + (freq / f_0) ** 2)
    S = S_peak * np.sqrt(Ac / (1.0 + Ac / area)) * freq

    return S


# ============================================================================
# TEMPORAL RESPONSE FUNCTIONS
# ============================================================================

def _ach_temporal_response(omega, lum):
    """Achromatic sustained + transient temporal response weights."""
    omega = np.asarray(omega, dtype=np.float64)
    lum = np.asarray(lum, dtype=np.float64)

    # Sustained
    sigma_sust = _ACH_TEMP['sigma_sust']
    beta_sust = 1.3314  # hardcoded in MATLAB source
    R_sust = np.exp(-omega ** beta_sust / sigma_sust)

    # Transient (bandpass, luminance-dependent peak)
    sigma_trans = _ACH_TEMP['sigma_trans']
    beta_trans = 0.1898  # hardcoded in MATLAB source
    omega_0 = np.log10(lum) * _ACH_TEMP['omega_trans_sl'] + _ACH_TEMP['omega_trans_c']
    R_trans = np.exp(-np.abs(omega ** beta_trans - omega_0 ** beta_trans) ** 2 / sigma_trans)

    return R_sust, R_trans


def _chrom_temporal_response(omega, sigma_sust, beta_sust):
    """Chromatic sustained-only temporal response."""
    omega = np.asarray(omega, dtype=np.float64)
    return np.exp(-omega ** beta_sust / sigma_sust)


# ============================================================================
# ECCENTRICITY DROP
# ============================================================================

def _apply_eccentricity(S, freq, ecc, vis_field, ecc_drop, ecc_drop_nasal,
                         ecc_drop_f, ecc_drop_f_nasal):
    """Apply eccentricity-dependent sensitivity drop."""
    if ecc == 0.0:
        return S
    alpha = min(1.0, abs(vis_field - 180.0) / 90.0)
    ed = alpha * ecc_drop + (1.0 - alpha) * ecc_drop_nasal
    edf = alpha * ecc_drop_f + (1.0 - alpha) * ecc_drop_f_nasal
    a = ed + freq * edf
    return S * 10.0 ** (-a * ecc)


# ============================================================================
# PUBLIC API
# ============================================================================

def csf_achromatic(freq, luminance=None, t_frequency=None, area=None,
                   eccentricity=None, vis_field=None):
    """Achromatic (luminance) contrast sensitivity.

    Args:
        freq: Spatial frequency in cycles per degree (cpd). Scalar or array.
        luminance: Background luminance in cd/m^2 (default: LUMINANCE).
        t_frequency: Temporal frequency in Hz (default: T_FREQUENCY).
        area: Stimulus area in deg^2 (default: AREA).
        eccentricity: Retinal eccentricity in deg (default: ECCENTRICITY).
        vis_field: Visual field orientation in deg (default: VIS_FIELD).

    Returns:
        Contrast sensitivity (1/threshold). Same shape as freq.
    """
    lum = luminance if luminance is not None else LUMINANCE
    omega = t_frequency if t_frequency is not None else T_FREQUENCY
    A = area if area is not None else AREA
    ecc = eccentricity if eccentricity is not None else ECCENTRICITY
    vf = vis_field if vis_field is not None else VIS_FIELD

    freq = np.asarray(freq, dtype=np.float64)

    R_sust, R_trans = _ach_temporal_response(omega, lum)

    S_sust = _csf_achrom_channel(freq, A, lum, _ACH_SUST)
    S_trans = _csf_achrom_channel(freq, A, lum, _ACH_TRANS)

    # ps_beta=1 path (linear summation, pm_ratio=1)
    S = R_sust * S_sust + R_trans * S_trans

    S = _apply_eccentricity(S, freq, ecc, vf,
                            _ACH_TEMP['ecc_drop'], _ACH_TEMP['ecc_drop_nasal'],
                            _ACH_TEMP['ecc_drop_f'], _ACH_TEMP['ecc_drop_f_nasal'])

    return np.maximum(S, 0.0)


def csf_rg(freq, luminance=None, t_frequency=None, area=None,
           eccentricity=None, vis_field=None):
    """Red-Green chromatic contrast sensitivity.

    Args: same as csf_achromatic.
    Returns: Contrast sensitivity (1/threshold).
    """
    lum = luminance if luminance is not None else LUMINANCE
    omega = t_frequency if t_frequency is not None else T_FREQUENCY
    A = area if area is not None else AREA
    ecc = eccentricity if eccentricity is not None else ECCENTRICITY
    vf = vis_field if vis_field is not None else VIS_FIELD

    freq = np.asarray(freq, dtype=np.float64)

    R_sust = _chrom_temporal_response(omega, _RG['sigma_sust'], _RG['beta_sust'])

    ch_pars = dict(S_max=_RG['S_max'], f_max=_RG['f_max'], bw=_RG['bw'])
    S = R_sust * _csf_chrom_channel(freq, A, lum, ch_pars, _RG['f_0'], _RG['A_0'])

    S = _apply_eccentricity(S, freq, ecc, vf,
                            _RG['ecc_drop'], _RG['ecc_drop_nasal'],
                            _RG['ecc_drop_f'], _RG['ecc_drop_f_nasal'])

    return np.maximum(S, 0.0)


def csf_yv(freq, luminance=None, t_frequency=None, area=None,
           eccentricity=None, vis_field=None):
    """Yellow-Violet chromatic contrast sensitivity.

    Args: same as csf_achromatic.
    Returns: Contrast sensitivity (1/threshold).
    """
    lum = luminance if luminance is not None else LUMINANCE
    omega = t_frequency if t_frequency is not None else T_FREQUENCY
    A = area if area is not None else AREA
    ecc = eccentricity if eccentricity is not None else ECCENTRICITY
    vf = vis_field if vis_field is not None else VIS_FIELD

    freq = np.asarray(freq, dtype=np.float64)

    R_sust = _chrom_temporal_response(omega, _YV['sigma_sust'], _YV['beta_sust'])

    ch_pars = dict(S_max=_YV['S_max'], f_max=_YV['f_max'], bw=_YV['bw'])
    S = R_sust * _csf_chrom_channel(freq, A, lum, ch_pars, _YV['f_0'], _YV['A_0'])

    S = _apply_eccentricity(S, freq, ecc, vf,
                            _YV['ecc_drop'], _YV['ecc_drop_nasal'],
                            _YV['ecc_drop_f'], _YV['ecc_drop_f_nasal'])

    return np.maximum(S, 0.0)


# ============================================================================
# CONVENIENCE
# ============================================================================

def csf_all(freq, **kwargs):
    """Return all three channels as a dict."""
    return {
        'achromatic': csf_achromatic(freq, **kwargs),
        'rg': csf_rg(freq, **kwargs),
        'yv': csf_yv(freq, **kwargs),
    }


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    freq = np.logspace(np.log10(0.0625), np.log10(64), 500)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Default viewing conditions ---
    S = csf_all(freq)
    ax = axes[0]
    ax.loglog(freq, S['achromatic'], 'k-', linewidth=2, label='Achromatic')
    ax.loglog(freq, S['rg'], 'r-', linewidth=2, label='Red-Green')
    ax.loglog(freq, S['yv'], 'b-', linewidth=2, label='Yellow-Violet')
    ax.set_xlabel('Spatial frequency (cpd)')
    ax.set_ylabel('Contrast sensitivity')
    ax.set_title(f'castleCSF  (L={LUMINANCE} cd/m², ω={T_FREQUENCY} Hz, A={AREA} deg²)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(0.0625, 64)
    ax.set_ylim(0.1, 1e4)

    # --- Plot 2: Luminance dependence ---
    ax = axes[1]
    lums = [0.1, 1, 10, 100, 1000]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lums)))
    for lum, c in zip(lums, colors):
        S_ach = csf_achromatic(freq, luminance=lum)
        ax.loglog(freq, S_ach, color=c, linewidth=1.5, label=f'{lum} cd/m²')
    ax.set_xlabel('Spatial frequency (cpd)')
    ax.set_ylabel('Contrast sensitivity')
    ax.set_title('Achromatic CSF — luminance dependence')
    ax.legend(title='Luminance', fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(0.0625, 64)
    ax.set_ylim(0.1, 1e4)

    plt.tight_layout()
    plt.savefig('castle_csf_curves.png', dpi=150)
    plt.close()
    print("Saved castle_csf_curves.png")

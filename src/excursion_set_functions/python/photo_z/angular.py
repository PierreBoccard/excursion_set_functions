"""
Angular factor G(a) for photo-z damping (eq. 87).

This module implements the line-of-sight angular damping factor due to 
photo-z uncertainty.
"""

import numpy as np
from numba import jit
from scipy import special


def G_photo_z(a):
    """
    Angular factor for photo-z damping: G(a) = sqrt(pi)/(2a) * erf(a)
    
    This factor arises from the LOS (line-of-sight) damping due to photo-z 
    uncertainty. G(0) = 1.
    
    From eq. (87):
    G(a) = sqrt(pi)/(2a) * erf(a), where a = k*sigma_chi
    
    Parameters
    ----------
    a : float or array
        Argument k*sigma_chi where sigma_chi is the photo-z scatter in 
        comoving distance units
        
    Returns
    -------
    G : float or array
        Angular damping factor
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    result = np.ones_like(a)
    
    # For small a, use Taylor expansion: G(a) ≈ 1 - a^2/3 + a^4/10 - ...
    small = np.abs(a) < 1e-6
    large = ~small
    
    a_small = a[small]
    a2 = a_small ** 2
    result[small] = 1.0 - a2 / 3.0 + a2 ** 2 / 10.0
    
    # For larger a, use the full expression
    a_large = a[large]
    result[large] = np.sqrt(np.pi) / (2.0 * a_large) * special.erf(a_large)
    
    if result.size == 1:
        return float(result[0])
    return result


@jit(nopython=True, cache=True)
def _G_photo_z_numba(a):
    """
    Numba-compatible angular factor G(a) for photo-z.
    
    Uses polynomial approximation for erf.
    """
    if np.abs(a) < 1e-6:
        # Taylor expansion
        a2 = a * a
        return 1.0 - a2 / 3.0 + a2 * a2 / 10.0
    else:
        # Approximation for erf using polynomial
        # erf(x) ≈ 1 - exp(-x^2) * (a1*t + a2*t^2 + ...) where t = 1/(1+p*x)
        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        
        sign = 1.0 if a >= 0 else -1.0
        x = np.abs(a)
        t = 1.0 / (1.0 + p * x)
        erf_val = sign * (1.0 - (a1 * t + a2 * t**2 + a3 * t**3 + 
                                  a4 * t**4 + a5 * t**5) * np.exp(-x * x))
        
        return np.sqrt(np.pi) / (2.0 * a) * erf_val

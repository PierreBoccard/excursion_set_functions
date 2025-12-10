"""
Window functions and photo-z damping for VSF computation.

Implements Fourier-space top-hat window function and the photo-z damping
factor G(k, σ_χ) from angular integration.
"""

import numpy as np
from scipy.special import erf


def tophat_window_fourier(x):
    """
    Fourier transform of real-space top-hat window.
    
    W_T(x) = 3(sin(x) - x*cos(x))/x³
    
    Parameters:
    -----------
    x : float or array
        Argument k*R
        
    Returns:  
    --------
    W :   float or array
        Window function value
        
    Notes:
    ------
    Uses Taylor expansion near x=0 for numerical stability. 
    """
    x = np.asarray(x)
    
    # Handle x=0 case
    result = np.zeros_like(x, dtype=float)
    mask = x != 0
    
    x_nonzero = x[mask]
    result[mask] = 3.0 * (np.sin(x_nonzero) - x_nonzero * np.cos(x_nonzero)) / (x_nonzero**3)
    result[~mask] = 1.0  # limit as x->0
    
    return result


def G_function(k, sigma_chi):
    """
    Photo-z damping factor from angular integration.  
    
    G(a) = (√π / 2a) * erf(a) where a = k*sigma_chi
    
    This comes from integrating exp(-k_∥² σ_χ²) over the angular direction,
    representing the smoothing effect of photo-z uncertainties along the
    line of sight.
    
    Parameters:
    -----------
    k :  array
        Wavenumber in h/Mpc
    sigma_chi : float
        Comoving radial uncertainty in Mpc/h
        
    Returns:   
    --------
    G :   array
        Damping factor (1 when sigma_chi=0, <1 otherwise)
        
    Notes: 
    ------
    - Uses Taylor expansion for small k*σ_χ to avoid numerical issues
    - For σ_χ = 0 (no photo-z), returns 1 (no damping)
    - Damping increases with k (smaller scales more affected)
    """
    if sigma_chi == 0:
        return np.ones_like(k)
    
    a = k * sigma_chi
    
    # Handle small a with Taylor expansion for numerical stability
    # G(a) ≈ 1 - a²/3 + a⁴/10 - ...
    mask_small = a < 0.1
    result = np.zeros_like(a, dtype=float)
    
    # Small a:   Taylor series
    a_small = a[mask_small]
    result[mask_small] = 1.0 - (a_small**2) / 3.0 + (a_small**4) / 10.0
    
    # Large a:  exact formula
    a_large = a[~mask_small]
    result[~mask_small] = (np.sqrt(np.pi) / (2.0 * a_large)) * erf(a_large)
    
    return result
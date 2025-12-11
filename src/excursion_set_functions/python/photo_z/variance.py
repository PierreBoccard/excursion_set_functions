"""
Variance and covariance computations for filtered density fields.

Implements computation of variance S(R), derivatives dS/dR, and the W parameter
needed for moving barrier excursion set theory with photo-z effects.
"""

import numpy as np
from . window_functions import tophat_window_fourier, G_function


def compute_effective_variance_photoz(Pk_interp, kh, R, sigma_chi):
    """
    Compute effective variance with photo-z using the analytical G-function.
    
    S_eff(R,z) = (1/2π²) ∫ dk k² P(k) |W_L(kR)|² G(k σ_χ)
    
    Parameters:  
    -----------
    Pk_interp : callable
        Interpolated power spectrum function P(k)
    kh : array
        Wavenumber grid in h/Mpc
    R : float
        Lagrangian radius in Mpc/h
    sigma_chi : float
        Comoving radial uncertainty in Mpc/h (if 0, reduces to isotropic)
        
    Returns:
    --------
    S_eff : float
        Effective variance
        
    Notes: 
    ------
    When sigma_chi = 0, this reduces to the standard isotropic variance.
    """
    k = kh
    Pk = Pk_interp(k)
    W = tophat_window_fourier(k * R)
    G = G_function(k, sigma_chi)
    
    # Integrand:  k² P(k) |W|² G(k σ_χ)
    integrand = k**2 * Pk * W**2 * G
    
    # Integrate
    S_eff = np. trapz(integrand, k) / (2.0 * np. pi**2)
    
    return S_eff


def compute_cross_covariance_photoz(Pk_interp, kh, R1, R2, sigma_chi):
    """
    Compute cross-covariance Ξ(R1, R2) with photo-z damping.
    
    Ξ(R1, R2) = (1/2π²) ∫ dk k² P(k) W_T(k R1) W_T(k R2) G(k σ_χ)
    
    Parameters: 
    -----------
    Pk_interp : callable
        Interpolated power spectrum
    kh : array
        Wavenumber grid
    R1, R2 : float
        Filter radii
    sigma_chi :   float
        Comoving radial uncertainty
        
    Returns:  
    --------
    Xi_12 : float
        Cross-covariance value
        
    Notes:
    ------
    Used to compute finite-difference derivatives for the W parameter.
    """
    k = kh
    Pk = Pk_interp(k)
    
    W1 = tophat_window_fourier(k * R1)
    W2 = tophat_window_fourier(k * R2)
    G = G_function(k, sigma_chi)
    
    integrand = k**2 * Pk * W1 * W2 * G
    Xi_12 = np.trapz(integrand, k) / (2.0 * np.pi**2)
    
    return Xi_12


def compute_W_reference_method(Pk_interp, kh, R_array, sigma_chi, dRperc=5e-3, verbose=True):
    """
    Compute W parameter using the reference covariance matrix method.
    
    Matches the integration. py:: sigma2_2_TopHat_numdiff approach:  
    1. Compute cross-covariances Ξ(R±ε, R±ε) for small ε
    2. Compute mixed second derivative:  ∂²Ξ/∂R₁∂R₂|_{R₁=R₂}
    3. Normalize by (dS/dR)² to get W = sigma2_2 / (dS/dR)²
    
    This W parameter is what enters the moving barrier formula f_S_MB_approx.
    
    Parameters:
    -----------
    Pk_interp : callable
        Interpolated power spectrum
    kh :  array
        Wavenumber grid
    R_array : array
        Array of Lagrangian radii
    sigma_chi : float
        Comoving radial uncertainty
    dRperc : float, optional
        Relative step size for finite differences (default 5e-3 = 0.5%)
    verbose : bool, optional
        Print progress messages (default True)
        
    Returns:
    --------
    S :  array
        Variance at each radius
    dS_dR : array
        First derivative dS/dR
    W :   array
        W = sigma2_2 / (dS/dR)²
        
    Notes:
    ------
    The W parameter encodes information about correlations in the random walk
    and is essential for the moving barrier approximation to be accurate.
    """
    N = len(R_array)
    S = np.zeros(N)
    dS_dR = np.zeros(N)
    W = np.zeros(N)
    
    for i, R in enumerate(R_array):
        # Grid of perturbations:   [R(1-ε), R, R(1+ε)]
        R_minus = R * (1.0 - dRperc)
        R_plus = R * (1.0 + dRperc)
        
        # Compute covariance matrix elements
        # [0]:   Ξ(R-, R-)  [-- case]
        # [1]:  Ξ(R+, R-)  [+- case]
        # [2]:   Ξ(R+, R+)  [++ case]
        Xi_mm = compute_effective_variance_photoz(Pk_interp, kh, R_minus, sigma_chi)
        Xi_pm = compute_cross_covariance_photoz(Pk_interp, kh, R_plus, R_minus, sigma_chi)
        Xi_pp = compute_effective_variance_photoz(Pk_interp, kh, R_plus, sigma_chi)
        
        # Main variance (at R)
        S[i] = compute_effective_variance_photoz(Pk_interp, kh, R, sigma_chi)
        
        # Compute sigma2_2 (mixed second derivative of covariance)
        # This is:   ∂²Ξ(R1,R2)/∂R1∂R2 |_{R1=R2=R}
        sigma2_2 = 0.25 * (Xi_pp - 2.0 * Xi_pm + Xi_mm) / (R * dRperc)**2
        
        # Compute dS/dR using central difference
        dS_dR[i] = (Xi_pp - Xi_mm) / (2.0 * R * dRperc)
        
        # W parameter
        W[i] = sigma2_2 / (dS_dR[i]**2)
    
    return S, dS_dR, W
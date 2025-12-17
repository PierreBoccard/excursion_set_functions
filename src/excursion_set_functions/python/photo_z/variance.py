"""
Variance and covariance computations for filtered density fields.

Implements computation of variance S(R), derivatives dS/dR, and the W parameter
needed for moving barrier excursion set theory with photo-z effects.
"""

import numpy as np
from . window_functions import tophat_window_fourier, G_function


def tophat_window_derivative(x):
    """
    Derivative of Fourier top-hat window: dW_T/dx
    
    dW_T/dx = 3[sin(x)/x² - 3(sin(x) - x*cos(x))/x⁴]
    
    Parameters:
    -----------
    x : float or array
        Argument k*R
        
    Returns:
    --------
    dW_dx : float or array
        Derivative value
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    mask = x != 0
    
    x_nonzero = x[mask]
    sin_x = np.sin(x_nonzero)
    cos_x = np.cos(x_nonzero)
    
    result[mask] = 3.0 * (sin_x / x_nonzero**2 - 
                          3.0 * (sin_x - x_nonzero * cos_x) / x_nonzero**4)
    result[~mask] = 0.0  # limit as x->0
    
    return result


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


def compute_dS_dR_photoz(Pk_interp, kh, R, sigma_chi):
    """
    Compute true derivative dS/dR with photo-z effects.
    
    dS/dR = (1/π²) ∫ dk k³ P(k) W_T(kR) dW_T/d(kR) G(k σ_χ)
    
    This is the analytical derivative, not a finite difference approximation.
    
    Parameters:
    -----------
    Pk_interp : callable
        Interpolated power spectrum function P(k)
    kh : array
        Wavenumber grid in h/Mpc
    R : float
        Lagrangian radius in Mpc/h
    sigma_chi : float
        Comoving radial uncertainty in Mpc/h
        
    Returns:
    --------
    dS_dR : float
        True derivative of variance with respect to R
        
    Notes:
    ------
    Uses the chain rule: dS/dR = ∫ k² P(k) d[W²]/dR G dk
                                = ∫ k² P(k) 2W dW/d(kR) * k G dk
                                = (2/2π²) ∫ k³ P(k) W dW/dx G dk
    """
    k = kh
    Pk = Pk_interp(k)
    x = k * R
    
    W = tophat_window_fourier(x)
    dW_dx = tophat_window_derivative(x)
    G = G_function(k, sigma_chi)
    
    # Integrand: k³ P(k) W(kR) dW/d(kR) G(k σ_χ)
    integrand = k**3 * Pk * W * dW_dx * G
    
    # Factor of 2 from derivative of W², factor of 1/(2π²) from normalization
    dS_dR = 2.0 * np.trapz(integrand, k) / (2.0 * np.pi**2)
    
    return dS_dR


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


def compute_d2Xi_dR1dR2_photoz(Pk_interp, kh, R, sigma_chi):
    """
    Compute mixed second derivative ∂²Ξ/∂R₁∂R₂|_{R₁=R₂=R} analytically.
    
    This is sigma2_2, needed for the W parameter computation.
    
    At R₁ = R₂ = R:
    ∂²Ξ/∂R₁∂R₂ = (1/2π²) ∫ dk k⁴ P(k) [dW/dx]² G(k σ_χ)
    
    where x = kR and dW/dx is the derivative of the window function.
    
    Parameters:
    -----------
    Pk_interp : callable
        Interpolated power spectrum function P(k)
    kh : array
        Wavenumber grid in h/Mpc
    R : float
        Lagrangian radius in Mpc/h
    sigma_chi : float
        Comoving radial uncertainty in Mpc/h
        
    Returns:
    --------
    sigma2_2 : float
        Mixed second derivative (variance of the derivative)
        
    Notes:
    ------
    This is derived from:
    ∂Ξ/∂R₁ = (1/2π²) ∫ k³ P(k) W₂ dW₁/dx G dk
    ∂²Ξ/∂R₁∂R₂ = (1/2π²) ∫ k⁴ P(k) dW₁/dx dW₂/dx G dk
    At R₁=R₂=R: both derivatives are the same, so we get [dW/dx]²
    """
    k = kh
    Pk = Pk_interp(k)
    x = k * R
    
    dW_dx = tophat_window_derivative(x)
    G = G_function(k, sigma_chi)
    
    # Integrand: k⁴ P(k) [dW/dx]² G(k σ_χ)
    integrand = k**4 * Pk * dW_dx**2 * G
    
    sigma2_2 = np.trapz(integrand, k) / (2.0 * np.pi**2)
    
    return sigma2_2


def compute_W_reference_method(Pk_interp, kh, R_array, sigma_chi):
    """
    Compute W parameter using analytical derivatives (not finite differences).
    
    Computes:
    1. Variance S(R) = (1/2π²) ∫ k² P(k) W² G dk
    2. First derivative dS/dR = (1/π²) ∫ k³ P(k) W dW/dx G dk
    3. Second derivative sigma2_2 = (1/2π²) ∫ k⁴ P(k) [dW/dx]² G dk
    4. W parameter: W = sigma2_2 / (dS/dR)²
    
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
        DEPRECATED - kept for backward compatibility, not used
    verbose : bool, optional
        Print progress messages (default True)
        
    Returns:
    --------
    S :  array
        Variance at each radius
    dS_dR : array
        Analytical first derivative dS/dR
    W :   array
        W = sigma2_2 / (dS/dR)²
        
    Notes:
    ------
    The W parameter encodes information about correlations in the random walk
    and is essential for the moving barrier approximation to be accurate.
    Now uses analytical derivatives instead of finite differences for better accuracy.
    """
    N = len(R_array)
    S = np.zeros(N)
    dS_dR = np.zeros(N)
    W = np.zeros(N)
    
    for i, R in enumerate(R_array):
        # Compute variance analytically
        S[i] = compute_effective_variance_photoz(Pk_interp, kh, R, sigma_chi)
        
        # Compute first derivative analytically (TRUE derivative, not finite difference)
        dS_dR[i] = compute_dS_dR_photoz(Pk_interp, kh, R, sigma_chi)
        
        # Compute second derivative (sigma2_2) analytically
        sigma2_2 = compute_d2Xi_dR1dR2_photoz(Pk_interp, kh, R, sigma_chi)
        
        # W parameter
        if dS_dR[i] != 0:
            W[i] = sigma2_2 / (dS_dR[i]**2)
        else:
            W[i] = 0.0
    
    return S, dS_dR, W
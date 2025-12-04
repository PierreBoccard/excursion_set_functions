"""
Photo-z extension module for excursion-set functions.

This module implements the photometric redshift error formalism for 
excursion-set calculations as described in Photo_z_2.pdf.

Key equations implemented:
- G(a): Angular factor (eq. 87)
- Seff: Effective variance with photo-z (eq. 88)
- dSeff/dR: First derivative (eq. 89)
- d2Seff/dR2: Second derivative (eq. 90)
- DW: Diffusion coefficient (eq. 91)
- Xi_eff: Cross-covariance (eq. 92)
- B_photo_z: Barrier with photo-z (eq. 93)
- f_photo_z: Multiplicity with photo-z (eqs. 95-98)
"""

import numpy as np
from numba import jit
from scipy import special


# =============================================================================
# Tophat window function and derivatives (eqs. 85-86)
# =============================================================================

@jit(nopython=True, cache=True)
def tophat_window(x):
    """
    Tophat window function in Fourier space: W_T(x) = 3(sin(x) - x*cos(x))/x^3
    
    Parameters
    ----------
    x : float or array
        Argument kR
        
    Returns
    -------
    W : float or array
        Tophat window function value
    """
    result = np.zeros_like(x)
    small = np.abs(x) < 1e-2
    large = ~small
    
    # Taylor expansion for small x
    x2 = x[small] ** 2
    x4 = x2 ** 2
    x6 = x4 * x2
    x8 = x6 * x2
    result[small] = 1.0 - x2 / 10.0 + x4 / 280.0 - x6 / 15120.0 + x8 / 1330560.0
    
    # Full expression for large x
    xl = x[large]
    result[large] = 3.0 * (np.sin(xl) - xl * np.cos(xl)) / (xl ** 3)
    
    return result


@jit(nopython=True, cache=True)
def tophat_window_derivative(x):
    """
    First derivative of tophat window function: W'_T(x) = dW_T/dx
    
    From eq. (85):
    W'_T(x) = 3 * [sin(x)/x^2 - 3(sin(x) - x*cos(x))/x^4]
    
    Parameters
    ----------
    x : float or array
        Argument kR
        
    Returns
    -------
    dW : float or array
        First derivative of tophat window function
    """
    result = np.zeros_like(x)
    small = np.abs(x) < 1e-2
    large = ~small
    
    # Taylor expansion for small x: dW/dx = -x/5 + x^3/70 - x^5/2520 + ...
    xs = x[small]
    x2 = xs ** 2
    x4 = x2 ** 2
    x6 = x4 * x2
    result[small] = -xs / 5.0 + xs ** 3 / 70.0 - xs ** 5 / 2520.0 + xs ** 7 / 166320.0
    
    # Full expression for large x
    xl = x[large]
    sin_x = np.sin(xl)
    cos_x = np.cos(xl)
    result[large] = 3.0 * (sin_x / (xl ** 2) - 3.0 * (sin_x - xl * cos_x) / (xl ** 4))
    
    return result


@jit(nopython=True, cache=True)
def tophat_window_second_derivative(x):
    """
    Second derivative of tophat window function: W''_T(x) = d^2W_T/dx^2
    
    Computed as d/dx of dW/dx, where:
    dW/dx = 3 * [sin(x)/x^2 - 3*(sin(x) - x*cos(x))/x^4]
    
    Resulting in:
    d^2W/dx^2 = 3 * [cos(x)/x^2 - 5*sin(x)/x^3 + 12*(sin(x) - x*cos(x))/x^5]
    
    Parameters
    ----------
    x : float or array
        Argument kR
        
    Returns
    -------
    d2W : float or array
        Second derivative of tophat window function
    """
    result = np.zeros_like(x)
    small = np.abs(x) < 1e-2
    large = ~small
    
    # Taylor expansion for small x: d^2W/dx^2 = -1/5 + 3*x^2/70 - x^4/504 + ...
    xs = x[small]
    x2 = xs ** 2
    x4 = x2 ** 2
    result[small] = -1.0 / 5.0 + 3.0 * x2 / 70.0 - x4 / 504.0 + x2 ** 3 / 23760.0
    
    # Full expression for large x
    xl = x[large]
    sin_x = np.sin(xl)
    cos_x = np.cos(xl)
    result[large] = 3.0 * (cos_x / (xl ** 2) - 5.0 * sin_x / (xl ** 3) 
                           + 12.0 * (sin_x - xl * cos_x) / (xl ** 5))
    
    return result


# =============================================================================
# Angular factor G(a) for photo-z (eq. 87)
# =============================================================================

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


# =============================================================================
# Effective variance Seff and derivatives with photo-z (eqs. 88-91)
# =============================================================================

def Seff_photo_z(Pk, k, R, sigma_chi):
    """
    Compute effective variance Seff(R) with photo-z damping.
    
    From eq. (88):
    S_eff(R) = 1/(2*pi^2) * integral dk k^2 P(k) W_T^2(kR) G(k*sigma_chi)
    
    Parameters
    ----------
    Pk : array
        Power spectrum P(k)
    k : array
        Wavenumber array in h/Mpc
    R : float or array
        Smoothing radius in Mpc/h
    sigma_chi : float
        Photo-z scatter in comoving distance units (Mpc/h)
        
    Returns
    -------
    Seff : float or array
        Effective variance at radius R
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    k = np.asarray(k, dtype=float)
    Pk = np.asarray(Pk, dtype=float)
    
    # Compute G(k*sigma_chi) once
    G = G_photo_z(k * sigma_chi)
    
    Seff = np.zeros(len(R))
    for i, r in enumerate(R):
        x = k * r
        W2 = tophat_window(x) ** 2
        integrand = k ** 2 * Pk * W2 * G
        Seff[i] = np.trapz(integrand, k) / (2.0 * np.pi ** 2)
    
    if len(Seff) == 1:
        return float(Seff[0])
    return Seff


def dSeff_dR_photo_z(Pk, k, R, sigma_chi):
    """
    Compute first derivative dSeff/dR with photo-z damping.
    
    From eq. (89):
    dS_eff/dR = 1/(pi^2) * integral dk k^3 P(k) W_T(kR) W'_T(kR) G(k*sigma_chi)
    
    Parameters
    ----------
    Pk : array
        Power spectrum P(k)
    k : array
        Wavenumber array in h/Mpc
    R : float or array
        Smoothing radius in Mpc/h
    sigma_chi : float
        Photo-z scatter in comoving distance units (Mpc/h)
        
    Returns
    -------
    dSeff_dR : float or array
        First derivative of effective variance at radius R
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    k = np.asarray(k, dtype=float)
    Pk = np.asarray(Pk, dtype=float)
    
    # Compute G(k*sigma_chi) once
    G = G_photo_z(k * sigma_chi)
    
    dSeff = np.zeros(len(R))
    for i, r in enumerate(R):
        x = k * r
        W = tophat_window(x)
        dW = tophat_window_derivative(x)
        # Note: dW/dR = dW/dx * dx/dR = dW/dx * k
        integrand = k ** 3 * Pk * W * dW * G
        dSeff[i] = np.trapz(integrand, k) / (np.pi ** 2)
    
    if len(dSeff) == 1:
        return float(dSeff[0])
    return dSeff


def d2Seff_dR2_photo_z(Pk, k, R, sigma_chi):
    """
    Compute second derivative d^2Seff/dR^2 with photo-z damping.
    
    From eq. (90):
    d^2S_eff/dR^2 = 1/(pi^2) * integral dk k^2 P(k) [k*W'_T(kR)]^2 + W_T(kR)*[k^2*W''_T(kR)]} G(k*sigma_chi)
    
    Parameters
    ----------
    Pk : array
        Power spectrum P(k)
    k : array
        Wavenumber array in h/Mpc
    R : float or array
        Smoothing radius in Mpc/h
    sigma_chi : float
        Photo-z scatter in comoving distance units (Mpc/h)
        
    Returns
    -------
    d2Seff_dR2 : float or array
        Second derivative of effective variance at radius R
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    k = np.asarray(k, dtype=float)
    Pk = np.asarray(Pk, dtype=float)
    
    # Compute G(k*sigma_chi) once
    G = G_photo_z(k * sigma_chi)
    
    d2Seff = np.zeros(len(R))
    for i, r in enumerate(R):
        x = k * r
        W = tophat_window(x)
        dW = tophat_window_derivative(x)
        d2W = tophat_window_second_derivative(x)
        # d^2(W^2)/dR^2 = 2*(dW/dR)^2 + 2*W*d^2W/dR^2
        # = 2*k^2*dW^2 + 2*W*k^2*d2W
        integrand = k ** 2 * Pk * ((k * dW) ** 2 + W * k ** 2 * d2W) * G
        d2Seff[i] = np.trapz(integrand, k) / (np.pi ** 2)
    
    if len(d2Seff) == 1:
        return float(d2Seff[0])
    return d2Seff


def DW_photo_z(Pk, k, R, sigma_chi):
    """
    Compute diffusion coefficient DW(R) with photo-z.
    
    From eq. (91):
    D_W(R) = (d^2S_eff/dR^2) / (dS_eff/dR)^2
    
    Parameters
    ----------
    Pk : array
        Power spectrum P(k)
    k : array
        Wavenumber array in h/Mpc
    R : float or array
        Smoothing radius in Mpc/h
    sigma_chi : float
        Photo-z scatter in comoving distance units (Mpc/h)
        
    Returns
    -------
    DW : float or array
        Diffusion coefficient at radius R
    """
    dSeff = dSeff_dR_photo_z(Pk, k, R, sigma_chi)
    d2Seff = d2Seff_dR2_photo_z(Pk, k, R, sigma_chi)
    
    return d2Seff / (dSeff ** 2)


# =============================================================================
# Cross-covariance Xi_eff with photo-z (eq. 92)
# =============================================================================

def Xi_eff_photo_z(Pk, k, R, sigma_chi):
    """
    Compute cross-covariance matrix Xi_eff(R_i, R_j) with photo-z.
    
    From eq. (92):
    Xi_ij = 1/(2*pi^2) * integral dk k^2 P(k) W_T(kR_i) W_T(kR_j) G(k*sigma_chi)
    
    Note: The diagonal elements Xi_ii = Seff(R_i)
    
    Parameters
    ----------
    Pk : array
        Power spectrum P(k)
    k : array
        Wavenumber array in h/Mpc
    R : array
        Array of smoothing radii in Mpc/h
    sigma_chi : float
        Photo-z scatter in comoving distance units (Mpc/h)
        
    Returns
    -------
    Xi_eff : 2D array
        Cross-covariance matrix of shape (len(R), len(R))
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    k = np.asarray(k, dtype=float)
    Pk = np.asarray(Pk, dtype=float)
    
    len_R = len(R)
    len_k = len(k)
    
    # Compute G(k*sigma_chi) once
    G = G_photo_z(k * sigma_chi)
    
    # Pre-compute all tophat windows using broadcasting
    # W_all[i, j] = W_T(k[j] * R[i])
    W_all = np.zeros((len_R, len_k))
    for i, r in enumerate(R):
        W_all[i] = tophat_window(k * r)
    
    # Compute integrand base: k^2 * P(k) * G
    integrand_base = k ** 2 * Pk * G  # shape: (len_k,)
    
    # Compute cross-covariance matrix using vectorized outer products
    # Xi[i,j] = trapz(integrand_base * W_all[i] * W_all[j], k) / (2*pi^2)
    Xi = np.zeros((len_R, len_R))
    for i in range(len_R):
        for j in range(i + 1):
            integrand = integrand_base * W_all[i] * W_all[j]
            Xi[i, j] = np.trapz(integrand, k) / (2.0 * np.pi ** 2)
            Xi[j, i] = Xi[i, j]  # Symmetric
    
    return Xi


# =============================================================================
# Barrier with photo-z (eq. 93)
# =============================================================================

def B_photo_z(B_spectro, S, Seff, alpha_B=1.0, beta_B=0.0):
    """
    Compute barrier with photo-z adjustment.
    
    From eq. (93):
    B_ph(R) = alpha_B * sqrt(S_eff(R)/S(R)) * B(S(R)) + beta_B
    
    where B(S) is the spectroscopic barrier and (alpha_B, beta_B) = (1, 0)
    for a parameter-free theory prediction.
    
    Parameters
    ----------
    B_spectro : float or array
        Spectroscopic barrier B(S)
    S : float or array
        Spectroscopic variance S(R)
    Seff : float or array
        Effective (photo-z) variance Seff(R)
    alpha_B : float, optional
        Barrier amplitude calibration parameter, default 1.0
    beta_B : float, optional
        Barrier offset calibration parameter, default 0.0
        
    Returns
    -------
    Bph : float or array
        Photo-z adjusted barrier
    """
    return alpha_B * np.sqrt(Seff / S) * B_spectro + beta_B


def dB_photo_z_dSeff(B_spectro, S, Seff, dS_dR, dSeff_dR, dB_dS, alpha_B=1.0):
    """
    Compute derivative dB_ph/dS_eff for photo-z barrier.
    
    From eqs. (72-73):
    dB_ph/dS_eff = (dB_ph/dR) / (dS_eff/dR)
    
    dB_ph/dR = alpha_B * [d/dR(sqrt(S_eff/S)) * B(S) + sqrt(S_eff/S) * dB/dS * dS/dR]
    
    d/dR(sqrt(S_eff/S)) = 1/2 * sqrt(S/S_eff) * [1/S_eff * dS_eff/dR - 1/S * dS/dR]
    
    Parameters
    ----------
    B_spectro : float or array
        Spectroscopic barrier B(S)
    S : float or array
        Spectroscopic variance S(R)
    Seff : float or array
        Effective (photo-z) variance Seff(R)
    dS_dR : float or array
        Derivative dS/dR (spectroscopic)
    dSeff_dR : float or array
        Derivative dSeff/dR (photo-z)
    dB_dS : float or array
        Derivative dB/dS of spectroscopic barrier
    alpha_B : float, optional
        Barrier amplitude calibration parameter, default 1.0
        
    Returns
    -------
    dBph_dSeff : float or array
        Derivative of photo-z barrier with respect to Seff
    """
    sqrt_ratio = np.sqrt(Seff / S)
    sqrt_inv = np.sqrt(S / Seff)
    
    # d/dR(sqrt(S_eff/S))
    d_sqrt_dR = 0.5 * sqrt_inv * (dSeff_dR / Seff - dS_dR / S)
    
    # dB_ph/dR
    dBph_dR = alpha_B * (d_sqrt_dR * B_spectro + sqrt_ratio * dB_dS * dS_dR)
    
    # dB_ph/dS_eff = (dB_ph/dR) / (dS_eff/dR)
    return dBph_dR / dSeff_dR


# =============================================================================
# Multiplicity functions with photo-z (eqs. 95-98)
# =============================================================================

def f_photo_z_MB(Seff, B_ph, dB_ph_dSeff, D_tot=0.0, N_taylor=5):
    """
    Compute photo-z multiplicity using MB (Musso-Sheth) form.
    
    From eqs. (95-96):
    f_ph(S_eff) ≈ |T_ph(S_eff)| / sqrt(2*pi*S_eff^3) * exp(-B_ph^2 / (2*(1+D_tot)*S_eff))
    
    T_ph(S_eff) = sum_{n=0}^{N} (-S_eff)^n / n! * d^n B_ph / dS_eff^n
    
    Parameters
    ----------
    Seff : float or array
        Effective variance Seff(R)
    B_ph : float or array
        Photo-z barrier Bph(R)
    dB_ph_dSeff : float or array
        First derivative dBph/dSeff
    D_tot : float, optional
        Total diffusion coefficient D_tot = D_B + D_ph, default 0.0
    N_taylor : int, optional
        Number of Taylor terms in T_ph, default 5
        
    Returns
    -------
    f_ph : float or array
        Photo-z multiplicity
    """
    Seff = np.atleast_1d(np.asarray(Seff, dtype=float))
    B_ph = np.atleast_1d(np.asarray(B_ph, dtype=float))
    dB_ph_dSeff = np.atleast_1d(np.asarray(dB_ph_dSeff, dtype=float))
    
    # Compute T_ph (simplified: using first few terms)
    # For a linear barrier B = delta_c (constant), only n=0 term is non-zero: T = B
    # For general case, we use the first-order approximation
    T_ph = B_ph - Seff * dB_ph_dSeff
    
    # Exponential suppression
    exp_term = np.exp(-B_ph ** 2 / (2.0 * (1.0 + D_tot) * Seff))
    
    # Prefactor
    prefactor = np.abs(T_ph) / np.sqrt(2.0 * np.pi * Seff ** 3)
    
    f_ph = prefactor * exp_term
    
    if len(f_ph) == 1:
        return float(f_ph[0])
    return f_ph


def f_photo_z_upcrossing(Seff, B_ph, dB_ph_dSeff, Gamma_eff_dd):
    """
    Compute photo-z multiplicity using upcrossing form.
    
    From eq. (98):
    f_ph(S_eff) ≈ exp(-B_ph^2/(2*S_eff)) / sqrt(2*pi*S_eff) *
        [sqrt(Gamma_eff/(2*pi*S_eff)) * exp(-S_eff/(2*Gamma_eff)*(B_ph/(2*S_eff) - dB_ph/dS_eff)^2)
         + 1/2 * (B_ph/(2*S_eff) - dB_ph/dS_eff) * (erf(...) + 1)]
    
    Parameters
    ----------
    Seff : float or array
        Effective variance Seff(R)
    B_ph : float or array
        Photo-z barrier Bph(R)
    dB_ph_dSeff : float or array
        Derivative dBph/dSeff
    Gamma_eff_dd : float or array
        Value-slope covariance Gamma_eff from photo-z covariance
        
    Returns
    -------
    f_ph : float or array
        Photo-z multiplicity (upcrossing form)
    """
    Seff = np.atleast_1d(np.asarray(Seff, dtype=float))
    B_ph = np.atleast_1d(np.asarray(B_ph, dtype=float))
    dB_ph_dSeff = np.atleast_1d(np.asarray(dB_ph_dSeff, dtype=float))
    Gamma_eff_dd = np.atleast_1d(np.asarray(Gamma_eff_dd, dtype=float))
    
    # beta_star = B_ph/(2*S_eff) - dB_ph/dS_eff
    beta_star = B_ph / (2.0 * Seff) - dB_ph_dSeff
    
    # Argument for the Gaussian and erf
    arg = np.sqrt(Seff / (2.0 * Gamma_eff_dd)) * beta_star
    
    # Term 1: Gaussian part
    term1 = np.sqrt(Gamma_eff_dd / (2.0 * np.pi * Seff)) * np.exp(-Seff / (2.0 * Gamma_eff_dd) * beta_star ** 2)
    
    # Term 2: erf part
    term2 = 0.5 * beta_star * (special.erf(arg) + 1.0)
    
    # Prefactor
    prefactor = np.exp(-B_ph ** 2 / (2.0 * Seff)) / np.sqrt(2.0 * np.pi * Seff)
    
    f_ph = prefactor * (term1 + term2)
    
    if len(f_ph) == 1:
        return float(f_ph[0])
    return f_ph


def Gamma_eff_from_Xi(Xi_eff, Seff):
    """
    Compute Gamma_eff_dd from cross-covariance matrix.
    
    From eqs. (76-78):
    Gamma_delta_delta is obtained from the value-slope moments:
    <delta^2> = S
    <delta * dot_delta> = dC(S,S')/dS' |_{S'=S}
    <dot_delta^2> = d^2C(S,S')/(dS*dS') |_{S'=S}
    
    Parameters
    ----------
    Xi_eff : 2D array
        Cross-covariance matrix from Xi_eff_photo_z
    Seff : array
        Diagonal of Xi_eff (effective variances)
        
    Returns
    -------
    Gamma_eff_dd : array
        Value-slope covariance for each radius
    """
    n = len(Seff)
    if n < 3:
        raise ValueError("Need at least 3 radii for finite differences")
    
    Gamma_eff = np.zeros(n)
    
    # Use finite differences to estimate derivatives
    for i in range(1, n - 1):
        # Central difference for d<delta*dot_delta>/dS
        dS = (Seff[i + 1] - Seff[i - 1]) / 2.0
        
        # <dot_delta^2> ≈ d^2 Xi / dS_i dS_j evaluated at i=j
        # Using second-order finite difference
        d2Xi = (Xi_eff[i + 1, i + 1] - 2 * Xi_eff[i, i] + Xi_eff[i - 1, i - 1]) / (dS ** 2)
        
        # Gamma = <dot_delta^2> - <delta*dot_delta>^2/<delta^2>
        # For a Markov walk, <delta*dot_delta>/<delta^2> = 1/(2S)
        # So Gamma ≈ <dot_delta^2> - 1/(4S)
        Gamma_eff[i] = max(d2Xi - 1.0 / (4.0 * Seff[i]), 1e-10)
    
    # Extrapolate to boundaries
    Gamma_eff[0] = Gamma_eff[1]
    Gamma_eff[-1] = Gamma_eff[-2]
    
    return Gamma_eff


# =============================================================================
# Eulerian VSF with photo-z (eq. 97, 99)
# =============================================================================

def dnE_dRE_photo_z(f_ph, Seff, dSeff_dR, R, RE, dR_dRE):
    """
    Compute Eulerian void size function with photo-z.
    
    From eq. (97) or (99):
    dn_E/dR_E = 3/(4*pi*R_E^3) * f_ph(S_eff) * |dS_eff/dR| * dR/dR_E
    
    Parameters
    ----------
    f_ph : array
        Photo-z multiplicity
    Seff : array
        Effective variance
    dSeff_dR : array
        Derivative dSeff/dR
    R : array
        Lagrangian radii
    RE : array
        Eulerian radii (typically RE = q*R with q ≈ 1.7)
    dR_dRE : array or float
        Jacobian dR/dR_E (= 1/q if RE = q*R)
        
    Returns
    -------
    dn_dRE : array
        Eulerian void size function
    """
    return 3.0 / (4.0 * np.pi * RE ** 3) * f_ph * np.abs(dSeff_dR) * dR_dRE


# =============================================================================
# Complete photo-z VSF pipeline
# =============================================================================

def compute_photo_z_vsf(Pk, k, R, sigma_chi, barrier_params=None, 
                        D_B=0.0, D_ph=0.0, alpha_B=1.0, beta_B=0.0,
                        method='MB', q=1.7):
    """
    Complete pipeline to compute void size function with photo-z.
    
    Parameters
    ----------
    Pk : array
        Power spectrum P(k)
    k : array
        Wavenumber array in h/Mpc
    R : array
        Lagrangian radii in Mpc/h
    sigma_chi : float
        Photo-z scatter in comoving distance units (Mpc/h).
        Set to 0 for spectroscopic limit.
    barrier_params : dict, optional
        Parameters for the barrier function. Should contain:
        - 'alpha': barrier amplitude (default 1.0 for voids: delta_v = -2.7)
        - 'beta': barrier slope parameter (default 0.0)
        - 'gamma': barrier power (default 0.0 for constant barrier)
        Default is a constant void barrier with delta_v = -2.7
    D_B : float, optional
        Baseline diffusion coefficient, default 0.0
    D_ph : float, optional
        Photo-z diffusion coefficient, default 0.0
    alpha_B : float, optional
        Barrier calibration amplitude, default 1.0
    beta_B : float, optional
        Barrier calibration offset, default 0.0
    method : str, optional
        Multiplicity method: 'MB' or 'upcrossing', default 'MB'
    q : float, optional
        Lagrangian to Eulerian radius ratio, default 1.7
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'R': Lagrangian radii
        - 'RE': Eulerian radii
        - 'Seff': Effective variances
        - 'S': Spectroscopic variances
        - 'Bph': Photo-z barriers
        - 'fph': Photo-z multiplicities
        - 'dn_dRE': Eulerian void size function
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    k = np.asarray(k, dtype=float)
    Pk = np.asarray(Pk, dtype=float)
    
    # Default barrier: constant void barrier
    if barrier_params is None:
        barrier_params = {'alpha': -2.7, 'beta': 0.0, 'gamma': 0.0}
    
    alpha = barrier_params.get('alpha', -2.7)
    beta = barrier_params.get('beta', 0.0)
    gamma = barrier_params.get('gamma', 0.0)
    
    # Compute spectroscopic variance (sigma_chi = 0)
    S = Seff_photo_z(Pk, k, R, sigma_chi=0.0)
    dS_dR = dSeff_dR_photo_z(Pk, k, R, sigma_chi=0.0)
    
    # Compute photo-z effective variance
    if sigma_chi > 0:
        Seff = Seff_photo_z(Pk, k, R, sigma_chi)
        dSeff_dR = dSeff_dR_photo_z(Pk, k, R, sigma_chi)
    else:
        Seff = S.copy()
        dSeff_dR = dS_dR.copy()
    
    # Spectroscopic barrier: B(S) = alpha * (1 + beta / S^gamma)
    # Handle cases: gamma=0 gives constant barrier, gamma>0 with beta!=0 adds S-dependence
    if gamma > 0 and beta != 0:
        B_spectro = alpha * (1.0 + beta / S ** gamma)
        dB_dS = -alpha * beta * gamma * S ** (-gamma - 1)
    else:
        # Constant barrier: B = alpha (when gamma=0 or beta=0)
        B_spectro = alpha * np.ones_like(S)
        dB_dS = np.zeros_like(S)
    
    # Photo-z barrier
    Bph = B_photo_z(B_spectro, S, Seff, alpha_B, beta_B)
    
    # Derivative dBph/dSeff
    dBph_dSeff = dB_photo_z_dSeff(B_spectro, S, Seff, dS_dR, dSeff_dR, dB_dS, alpha_B)
    
    # Total diffusion
    D_tot = D_B + D_ph
    
    # Compute multiplicity
    if method == 'MB':
        fph = f_photo_z_MB(Seff, Bph, dBph_dSeff, D_tot)
    elif method == 'upcrossing':
        # Need to compute Gamma_eff from cross-covariance
        Xi_eff = Xi_eff_photo_z(Pk, k, R, sigma_chi)
        Gamma_eff = Gamma_eff_from_Xi(Xi_eff, Seff)
        fph = f_photo_z_upcrossing(Seff, Bph, dBph_dSeff, Gamma_eff)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'MB' or 'upcrossing'")
    
    # Eulerian radii and VSF
    RE = q * R
    dR_dRE = 1.0 / q
    dn_dRE = dnE_dRE_photo_z(fph, Seff, dSeff_dR, R, RE, dR_dRE)
    
    return {
        'R': R,
        'RE': RE,
        'S': S,
        'Seff': Seff,
        'dSeff_dR': dSeff_dR,
        'Bph': Bph,
        'dBph_dSeff': dBph_dSeff,
        'fph': fph,
        'dn_dRE': dn_dRE,
        'sigma_chi': sigma_chi,
        'D_tot': D_tot,
    }

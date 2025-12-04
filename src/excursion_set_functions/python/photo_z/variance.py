"""
Effective variance Seff and derivatives with photo-z (eqs. 88-92).

This module implements the effective variance and related quantities
that incorporate photo-z damping.
"""

import numpy as np
from .window import tophat_window, tophat_window_derivative, tophat_window_second_derivative
from .angular import G_photo_z


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

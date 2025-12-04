"""
Barrier functions with photo-z (eq. 93).

This module implements the barrier function modifications
for photometric redshift uncertainties.
"""

import numpy as np


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

"""
Multiplicity functions with photo-z (eqs. 95-98).

This module implements the multiplicity functions that account for
photometric redshift uncertainties.
"""

import numpy as np
from scipy import special


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

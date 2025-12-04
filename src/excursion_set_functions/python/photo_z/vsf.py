"""
Void Size Function (VSF) computation with photo-z (eqs. 97, 99).

This module provides the complete pipeline to compute void size functions
with photometric redshift uncertainties.
"""

import numpy as np
from .variance import Seff_photo_z, dSeff_dR_photo_z, Xi_eff_photo_z, Gamma_eff_from_Xi
from .barrier import B_photo_z, dB_photo_z_dSeff
from .multiplicity import f_photo_z_MB, f_photo_z_upcrossing


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

"""
Cosmology utilities for photo-z VSF computation. 

Functions for converting photo-z uncertainties to comoving radial uncertainties.
"""

import numpy as np


def compute_sigma_chi(z, sigma_z, cosmo):
    """
    Compute comoving radial uncertainty from photo-z scatter.  
    
    The photo-z uncertainty σ_z(1+z) maps to a comoving radial uncertainty
    through the comoving distance-redshift relation.  
    
    Parameters:   
    -----------
    z :   float
        Redshift
    sigma_z :  float
        Photo-z scatter coefficient (dimensionless, σ_z/(1+z))
    cosmo : astropy.cosmology.Cosmology
        Cosmology object (e.g., FlatwCDM)
        
    Returns:  
    --------
    sigma_chi : float
        Comoving radial uncertainty in Mpc/h
        
    Examples:
    ---------
    >>> from astropy.cosmology import FlatwCDM
    >>> cosmo = FlatwCDM(H0=71, Om0=0.3, w0=-1.0, Ob0=0.044)
    >>> sigma_chi = compute_sigma_chi(z=1.0, sigma_z=0.01, cosmo=cosmo)
    """
    # Compute dχ/dz = c/H(z) in units of Mpc/h
    h = cosmo.H0. value / 100.0
    
    # Comoving distance derivative:    dχ/dz = c/H(z)
    Hz = cosmo.H(z).value  # in km/s/Mpc
    c_km_s = 299792.458  # speed of light in km/s
    
    dchi_dz = c_km_s / Hz  # in Mpc
    dchi_dz_h = dchi_dz * h  # in Mpc/h
    
    # Photo-z uncertainty in comoving space
    sigma_chi = np.abs(dchi_dz_h) * sigma_z * (1.0 + z)
    
    return sigma_chi
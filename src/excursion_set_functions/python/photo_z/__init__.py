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

Submodules:
- window: Tophat window functions and derivatives
- angular: Angular damping factor G(a)
- variance: Effective variance and related quantities
- barrier: Barrier function modifications
- multiplicity: Multiplicity functions
- vsf: Complete VSF pipeline
"""

# Window functions
from .window import (
    tophat_window,
    tophat_window_derivative,
    tophat_window_second_derivative,
)

# Angular factor
from .angular import (
    G_photo_z,
    _G_photo_z_numba,
)

# Variance functions
from .variance import (
    Seff_photo_z,
    dSeff_dR_photo_z,
    d2Seff_dR2_photo_z,
    DW_photo_z,
    Xi_eff_photo_z,
    Gamma_eff_from_Xi,
)

# Barrier functions
from .barrier import (
    B_photo_z,
    dB_photo_z_dSeff,
)

# Multiplicity functions
from .multiplicity import (
    f_photo_z_MB,
    f_photo_z_upcrossing,
)

# VSF pipeline
from .vsf import (
    dnE_dRE_photo_z,
    compute_photo_z_vsf,
)

__all__ = [
    # Window functions
    "tophat_window",
    "tophat_window_derivative",
    "tophat_window_second_derivative",
    # Angular factor
    "G_photo_z",
    "_G_photo_z_numba",
    # Variance functions
    "Seff_photo_z",
    "dSeff_dR_photo_z",
    "d2Seff_dR2_photo_z",
    "DW_photo_z",
    "Xi_eff_photo_z",
    "Gamma_eff_from_Xi",
    # Barrier functions
    "B_photo_z",
    "dB_photo_z_dSeff",
    # Multiplicity functions
    "f_photo_z_MB",
    "f_photo_z_upcrossing",
    # VSF pipeline
    "dnE_dRE_photo_z",
    "compute_photo_z_vsf",
]

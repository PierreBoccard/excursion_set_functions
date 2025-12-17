"""
Photo-z Void Size Function (VSF) Package

...  [previous docstring] ... 
"""

from .cosmology import compute_sigma_chi
from .window_functions import tophat_window_fourier, G_function
from .variance import (
    compute_effective_variance_photoz,
    compute_dS_dR_photoz,
    compute_d2Xi_dR1dR2_photoz,
    compute_cross_covariance_photoz,
    compute_W_reference_method,
    tophat_window_derivative
)
from .barriers import (
    compute_barrier_standard,
    compute_barrier_derivative_standard,
    compute_barrier_parameters_voids,
    compute_barrier_and_derivative,
    compute_constant_barrier,
    compute_linear_barrier,
    check_barrier_validity,
    peak_height,
    print_barrier_diagnostics
)
from .multiplicity import compute_vsf_eulerian_with_photoz

__version__ = "1.0.0"

__all__ = [
    # Cosmology
    "compute_sigma_chi",
    
    # Window functions
    "tophat_window_fourier",
    "G_function",
    
    # Variance
    "compute_effective_variance_photoz",
    "compute_dS_dR_photoz",
    "compute_d2Xi_dR1dR2_photoz",
    "compute_cross_covariance_photoz",
    "compute_W_reference_method",
    "tophat_window_derivative",
    
    # Barriers
    "compute_barrier_standard",
    "compute_barrier_derivative_standard",
    "compute_barrier_parameters_voids",
    "compute_barrier_and_derivative",
    "compute_constant_barrier",
    "compute_linear_barrier",
    "check_barrier_validity",
    "peak_height",
    "print_barrier_diagnostics",
    
    # Main VSF computation
    "compute_vsf_eulerian_with_photoz",
]
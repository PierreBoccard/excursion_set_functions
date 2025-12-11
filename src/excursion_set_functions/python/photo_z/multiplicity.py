"""
Void size function (VSF) computation with photo-z effects.

Main module implementing the complete VSF calculation including:
- Power spectrum setup
- Variance and W parameter computation  
- Barrier evaluation
- Multiplicity function computation
- Eulerian transformation
"""

import numpy as np
from scipy.interpolate import interp1d
from astropy.cosmology import FlatwCDM
import camb

from .cosmology import compute_sigma_chi
from .variance import compute_W_reference_method


def compute_vsf_eulerian_with_photoz(a1, a2, b1, b2, gamma, delta_v_lin, 
                                       omega_m, w, bias, R_euler_desired, 
                                       redshift_min, redshift_max, sigma_z,
                                       f_S_MB_approx_func,
                                       delta_NL_func=None,
                                       power_spectrum_cache=None,
                                       verbose=True):
    """
    Compute void size function (VSF) with photo-z effects. 
    
    Implements the complete excursion set formalism with photo-z damping:
    1. Sets up cosmology and computes power spectrum
    2. Computes variance S(R), dS/dR, and W parameter with photo-z effects
    3. Evaluates moving barrier B(S) and derivative dB/dS
    4. Computes multiplicity function f(S) using moving barrier approximation
    5. Transforms from Lagrangian to Eulerian space
    
    Parameters:  
    -----------
    a1, a2, b1, b2, gamma :   float
        Barrier function parameters
        Barrier:  B(S) = α(1 + (β/√S)^γ)
        where α = a1*|δ_v| + a2, β = b1*|δ_v| + b2
    delta_v_lin : float
        Linear void density contrast (typically negative, e.g., -0.8)
    omega_m, w : float
        Cosmological parameters (matter density, dark energy EOS)
    bias : float
        Void bias parameter for Eulerian transformation
    R_euler_desired : array
        Desired Eulerian radii (NOT USED - function creates its own grid)
    redshift_min, redshift_max : float
        Redshift range for power spectrum averaging
    sigma_z : float
        Photo-z scatter coefficient (dimensionless, σ_z/(1+z))
        Set to 0 for isotropic (no photo-z) case
    f_S_MB_approx_func : callable
        Moving barrier multiplicity function from analytical module
        Signature: f_S_MB_approx(s, W, B, dB_dS) -> array
    delta_NL_func : callable, optional
        Nonlinear delta transformation function (from utilities)
        If None, uses linear approximation
    power_spectrum_cache : dict, optional
        Cache for power spectra (passed by reference, modified in place)
    verbose : bool, optional
        Print diagnostic messages (default True)
        
    Returns:
    --------
    R_E : array
        Eulerian radii [Mpc/h]
    VSF_E : array
        Void size function dn/dR [h³/Mpc³]
    s :  array
        Variance array (for diagnostics)
        
    Notes:
    ------
    - Uses R grid from 100 → 1 Mpc/h (Lagrangian) to match reference
    - Moving barrier approximation requires LDD = s*W - 0.25 > 0
    - Photo-z damping suppresses small-scale structure
    
    Examples:
    ---------
    >>> from analytical import f_S_MB_approx
    >>> R_E, VSF_E, s = compute_vsf_eulerian_with_photoz(
    ...     a1=0.517, a2=-0.089, b1=0.098, b2=0.103, gamma=1,
    ...     delta_v_lin=-0.8, omega_m=0.3, w=-1. 0, bias=2,
    ...     R_euler_desired=None, redshift_min=1. 0, redshift_max=1.2,
    ...     sigma_z=0.01, f_S_MB_approx_func=f_S_MB_approx
    ... )
    """
    
    if power_spectrum_cache is None: 
        power_spectrum_cache = {}
    
    try:
        # ===== COSMOLOGY SETUP =====
        cosmo = FlatwCDM(H0=71, Om0=omega_m, w0=w, Ob0=0.044)
        z_mean = (redshift_min + redshift_max) / 2.0
        
        # ===== POWER SPECTRUM =====
        key = (round(omega_m, 4), round(w, 3))
        
        if key not in power_spectrum_cache:  
            
            pars = camb.CAMBparams()
            pars.set_cosmology(
                H0=cosmo.H0.value, 
                ombh2=cosmo. Ob0 * (cosmo.H0.value / 100)**2, 
                omch2=(cosmo.Om0 - cosmo.Ob0) * (cosmo.H0.value / 100)**2
            )
            pars.InitPower. set_params(ns=0.965)
            redshift = np.linspace(redshift_min, redshift_max, 10)[: :-1]
            pars. set_matter_power(redshifts=redshift, kmax=2.0)
            
            results = camb.get_results(pars)
            kh, z_arr, pk = results.get_matter_power_spectrum(
                minkh=1.e-4, maxkh=2, npoints=1000
            )
            Pk_average = np.mean(pk. T, axis=1)
            
            power_spectrum_cache[key] = (kh, Pk_average)
        
        kh, Pk_average = power_spectrum_cache[key]
        Pk_interp = interp1d(kh, Pk_average, kind='cubic', 
                             bounds_error=False, fill_value=0.0)
        
        # ===== LAGRANGIAN R GRID =====
        # Use reference grid:  100 → 1 Mpc/h (logarithmic)
        R_L = np.logspace(2, 0, 101)
        
        # ===== COMPUTE SIGMA_CHI =====
        if sigma_z > 0:
            sigma_chi = compute_sigma_chi(z_mean, sigma_z, cosmo)
        else:
            sigma_chi = 0.0
        
        # ===== VARIANCE AND W PARAMETER =====
        
        s_eff, dS_eff_dR, W_eff = compute_W_reference_method(
            Pk_interp, kh, R_L, sigma_chi, dRperc=5e-3, verbose=verbose
        )
        
        if sigma_z > 0:
            # Also compute isotropic for comparison
            s_iso, dS_iso_dR, W_iso = compute_W_reference_method(
                Pk_interp, kh, R_L, sigma_chi=0.0, dRperc=5e-3, verbose=verbose
            )
        else:
            s_iso, dS_iso_dR, W_iso = s_eff, dS_eff_dR, W_eff
        
        # ===== BARRIER FUNCTION =====
        alpha = a1 * abs(delta_v_lin) + a2
        beta = b1 * abs(delta_v_lin) + b2

        if sigma_z > 0:
            B = alpha * (1.0 + (beta / s_eff)**gamma)
            dB_dS = - alpha * beta**gamma * gamma * s_eff**(-gamma - 1.0)
            s = s_eff
            dsdR = dS_eff_dR
            W = W_eff
        else:
            B = alpha * (1.0 + (beta / s_iso)**gamma)
            dB_dS = - alpha * beta**gamma * gamma * s_iso**(-gamma - 1.0)
            s = s_iso
            dsdR = dS_iso_dR
            W = W_iso
        
        # Check moving barrier validity
        LDD = s * W - 0.25
        n_valid = np.sum(LDD > 0)
        
        if n_valid < len(LDD) and verbose:
            first_invalid = np.where(LDD <= 0)[0][0]
        
        # ===== MULTIPLICITY FUNCTION =====
        valid_mask = LDD > 0
        f_appr_R = np.zeros_like(s)
        
        if np.any(valid_mask):
            f_appr_R[valid_mask] = f_S_MB_approx_func(
                s[valid_mask], W[valid_mask], 
                B[valid_mask], dB_dS[valid_mask]
            ) * np.abs(dsdR[valid_mask])
            f_appr_R[~valid_mask] = np.nan
            
        else:
            f_appr_R[: ] = np.nan
        
        # ===== LAGRANGIAN VSF =====
        VSF_L = 3.0 / (4.0 * np.pi * R_L**3) * f_appr_R
        
        # ===== EULERIAN TRANSFORMATION =====
        if delta_NL_func is not None:
            # Use exact nonlinear transformation
            delta_v_NL = delta_NL_func(delta_v_lin)
            expansion_factor = (1.0 + delta_v_NL)**(-1.0/3.0)
        else:
            # Use linear approximation
            expansion_factor = (1 + delta_v_lin/bias)**(-1/3)
        
        R_E = R_L * expansion_factor
        VSF_E = VSF_L / expansion_factor
        
        return R_E, VSF_E, s
        
    except Exception as e: 
        if verbose:
            print(f"\n❌ Computation failed: {e}")
            import traceback
            traceback.print_exc()
        return np.array([]), np.array([]), None
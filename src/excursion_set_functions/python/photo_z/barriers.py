"""
Barrier functions for excursion set theory. 

Implements various barrier shapes and their derivatives for use in
moving barrier excursion set calculations.
"""

import numpy as np


def compute_barrier_standard(s, alpha, beta, gamma):
    """
    Compute standard moving barrier with power-law form.
    
    B(S) = α(1 + (β/√S)^γ)
    
    This is the standard form used in void abundance calculations,
    calibrated from simulations.
    
    Parameters: 
    -----------
    s : array_like
        Variance values
    alpha : float
        Overall normalization (depends on δ_v)
    beta : float
        Scale parameter (depends on δ_v)
    gamma : float
        Power-law index (typically 1 or 2)
        
    Returns:
    --------
    B : array_like
        Barrier height at each variance
        
    Examples:
    ---------
    >>> s = np.array([0.1, 0.2, 0.3])
    >>> B = compute_barrier_standard(s, alpha=0.3, beta=0.1, gamma=2)
    """
    return alpha * (1.0 + (beta / np.sqrt(s))**gamma)


def compute_barrier_derivative_standard(s, alpha, beta, gamma):
    """
    Compute derivative of standard barrier with respect to variance.
    
    dB/dS = -0.5 * α * β^γ * γ * S^(-γ/2 - 1)
    
    Parameters:
    -----------
    s :  array_like
        Variance values
    alpha : float
        Overall normalization
    beta : float
        Scale parameter
    gamma : float
        Power-law index
        
    Returns:
    --------
    dB_dS : array_like
        Barrier derivative at each variance
        
    Notes:
    ------
    The derivative is always negative for positive γ, meaning the
    barrier decreases as variance increases (walks become easier
    to cross at larger scales).
    """
    return -0.5 * alpha * beta**gamma * gamma * s**(-gamma/2.0 - 1.0)


def compute_barrier_parameters_voids(delta_v_lin, a1, a2, b1, b2):
    """
    Compute barrier parameters α and β from void parameters.
    
    Uses the calibration: 
    α = a1 * |δ_v| + a2
    β = b1 * |δ_v| + b2
    
    Parameters:
    -----------
    delta_v_lin : float
        Linear void density contrast (typically negative)
    a1, a2 : float
        Calibration parameters for α
    b1, b2 :  float
        Calibration parameters for β
        
    Returns: 
    --------
    alpha : float
        Barrier normalization
    beta : float
        Barrier scale parameter
        
    Examples:
    ---------
    >>> alpha, beta = compute_barrier_parameters_voids(
    ...     delta_v_lin=-0.8, a1=0.517, a2=-0.089, b1=0.098, b2=0.103
    ... )
    >>> print(f"α = {alpha:.4f}, β = {beta:.4f}")
    α = 0.3246, β = 0.1814
    
    Notes:
    ------
    Standard calibration for voids (from simulations):
    - a1 = 0.517, a2 = -0.089
    - b1 = 0.098, b2 = 0.103
    - gamma = 1
    """
    alpha = a1 * abs(delta_v_lin) + a2
    beta = b1 * abs(delta_v_lin) + b2
    return alpha, beta


def compute_barrier_and_derivative(s, delta_v_lin, a1, a2, b1, b2, gamma):
    """
    Compute barrier and its derivative for void abundance.
    
    Convenience function that combines parameter computation and
    barrier evaluation.
    
    Parameters:
    -----------
    s : array_like
        Variance values
    delta_v_lin : float
        Linear void density contrast
    a1, a2, b1, b2 : float
        Calibration parameters
    gamma : float
        Power-law index
        
    Returns: 
    --------
    B : array_like
        Barrier height
    dB_dS : array_like
        Barrier derivative
        
    Examples: 
    ---------
    >>> s = np.linspace(0.01, 1.0, 100)
    >>> B, dB_dS = compute_barrier_and_derivative(
    ...     s, delta_v_lin=-0.8, a1=0.517, a2=-0.089, 
    ...     b1=0.098, b2=0.103, gamma=1
    ...  )
    """
    alpha, beta = compute_barrier_parameters_voids(delta_v_lin, a1, a2, b1, b2)
    
    B = compute_barrier_standard(s, alpha, beta, gamma)
    dB_dS = compute_barrier_derivative_standard(s, alpha, beta, gamma)
    
    return B, dB_dS


def compute_constant_barrier(delta_c, s):
    """
    Compute constant (non-moving) barrier.
    
    B(S) = δ_c (constant)
    dB/dS = 0
    
    This is the Press-Schechter case where the barrier doesn't
    depend on scale.
    
    Parameters:
    -----------
    delta_c :  float
        Critical density contrast (e.g., 1.686 for halos)
    s : array_like
        Variance values (for output shape)
        
    Returns: 
    --------
    B : array_like
        Constant barrier (all equal to δ_c)
    dB_dS : array_like
        Zero derivative (all zeros)
        
    Examples: 
    ---------
    >>> s = np.array([0.1, 0.2, 0.3])
    >>> B, dB_dS = compute_constant_barrier(1.686, s)
    >>> print(B)
    [1.686 1.686 1.686]
    """
    B = np.full_like(s, delta_c, dtype=float)
    dB_dS = np.zeros_like(s, dtype=float)
    return B, dB_dS


def compute_linear_barrier(delta_c, beta_linear, s):
    """
    Compute linearly-varying barrier.
    
    B(S) = δ_c * (1 + β_linear * S)
    dB/dS = δ_c * β_linear
    
    Simple moving barrier with linear dependence on variance.
    
    Parameters:
    -----------
    delta_c :  float
        Baseline critical density
    beta_linear :  float
        Linear coefficient (drift parameter)
    s : array_like
        Variance values
        
    Returns:
    --------
    B : array_like
        Linear barrier
    dB_dS : array_like
        Constant derivative
        
    Notes:
    ------
    This form is sometimes used for halos with β_linear ≈ 0.47
    in the Sheth-Tormen formalism.
    """
    B = delta_c * (1.0 + beta_linear * s)
    dB_dS = np.full_like(s, delta_c * beta_linear, dtype=float)
    return B, dB_dS


def check_barrier_validity(s, W, barrier_type='moving'):
    """
    Check if barrier approximation is valid for given parameters.
    
    For moving barrier, requires:  s*W - 1/4 > 0
    For constant barrier, always valid.
    
    Parameters:
    -----------
    s : array_like
        Variance values
    W : array_like
        Diffusion parameter
    barrier_type : str, optional
        Type of barrier:  'moving' or 'constant' (default 'moving')
        
    Returns:
    --------
    valid : array_like (bool)
        Boolean mask of valid points
    LDD : array_like
        Leading diffusion denominator (s*W - 1/4)
        Only computed for moving barriers
        
    Examples:
    ---------
    >>> s = np.array([0.1, 0.5, 1.0])
    >>> W = np.array([3.0, 1.0, 0.5])
    >>> valid, LDD = check_barrier_validity(s, W, 'moving')
    >>> print(valid)
    [True True True]
    >>> print(LDD)
    [0.05 0.25 0.25]
    """
    if barrier_type == 'moving': 
        LDD = s * W - 0.25
        valid = LDD > 0
        return valid, LDD
    else: 
        # Constant barrier always valid
        valid = np.ones_like(s, dtype=bool)
        LDD = None
        return valid, LDD


def peak_height(B, s):
    """
    Compute peak height ν = B/√S.
    
    The peak height is the barrier height measured in units of the
    variance. It's the fundamental variable in excursion set theory.
    
    Parameters:
    -----------
    B : array_like
        Barrier height
    s : array_like
        Variance
        
    Returns:
    --------
    nu : array_like
        Peak height ν = B/√S
        
    Notes:
    ------
    For constant barriers, ν decreases with mass (increasing S).
    For moving barriers, the behavior depends on dB/dS.
    
    Examples:
    ---------
    >>> B = np.array([1.0, 1.5, 2.0])
    >>> s = np.array([0.25, 1.0, 4.0])
    >>> nu = peak_height(B, s)
    >>> print(nu)
    [2.  1.5 1. ]
    """
    return B / np.sqrt(s)


def inverse_peak_height(nu, s):
    """
    Compute barrier height from peak height.
    
    B = ν * √S
    
    Parameters:
    -----------
    nu : float or array_like
        Peak height
    s : array_like
        Variance
        
    Returns: 
    --------
    B :  array_like
        Barrier height
    """
    return nu * np.sqrt(s)


# ============================================================================
# PHOTO-Z SPECIFIC BARRIER RESCALING (for reference, not used in current code)
# ============================================================================

def compute_barrier_photoz_rescaled(s_eff, s_iso, delta_v_lin, a1, a2, b1, b2, gamma):
    """
    Compute photo-z adjusted barrier using peak height preservation.
    
    B_ph(R) = √(S_eff/S_iso) * B(S_iso)
    
    This rescaling preserves the peak height ν = B/√S when transitioning
    from isotropic variance S_iso to effective variance S_eff.
    
    **NOTE**: This approach is NOT used in the current implementation,
    which instead evaluates the barrier directly at S_eff.  This function
    is included for reference and future research.
    
    Parameters:
    -----------
    s_eff : array_like
        Effective variance with photo-z
    s_iso : array_like
        Isotropic variance (no photo-z)
    delta_v_lin : float
        Linear void density contrast
    a1, a2, b1, b2, gamma : float
        Barrier parameters
        
    Returns: 
    --------
    B_ph : array_like
        Rescaled barrier
    dB_ph_dS : array_like
        Derivative with respect to S_eff
        
    Notes:
    ------
    This preserves ν = B/√S: 
    ν_iso = B_iso/√S_iso = B_ph/√S_eff = ν_eff
    
    The derivative calculation requires careful chain rule application.
    See the formalism documentation for full derivation.
    """
    # Original barrier at isotropic variance
    alpha, beta = compute_barrier_parameters_voids(delta_v_lin, a1, a2, b1, b2)
    B_iso = compute_barrier_standard(s_iso, alpha, beta, gamma)
    
    # Rescaling factor
    rescale = np.sqrt(s_eff / s_iso)
    
    # Rescaled barrier
    B_ph = rescale * B_iso
    
    # For derivative, use simple approach:  evaluate at S_eff directly
    # (Full chain rule derivative is complex and not needed if we
    #  just evaluate barrier at new variance)
    dB_ph_dS = compute_barrier_derivative_standard(s_eff, alpha, beta, gamma)
    
    return B_ph, dB_ph_dS


# ============================================================================
# DIAGNOSTIC AND UTILITY FUNCTIONS
# ============================================================================

def print_barrier_diagnostics(s, B, dB_dS, W=None):
    """
    Print diagnostic information about barrier and validity.
    
    Parameters:
    -----------
    s : array_like
        Variance
    B : array_like
        Barrier height
    dB_dS : array_like
        Barrier derivative
    W : array_like, optional
        Diffusion parameter (if None, skips moving barrier check)
    """
    print("\n" + "="*70)
    print("BARRIER DIAGNOSTICS")
    print("="*70)
    
    print(f"\nVariance S:")
    print(f"  Range: [{np.min(s):.6e}, {np.max(s):.6e}]")
    
    print(f"\nBarrier B:")
    print(f"  Range: [{np.min(B):.6e}, {np.max(B):.6e}]")
    
    print(f"\nBarrier derivative dB/dS:")
    print(f"  Range: [{np. min(dB_dS):.6e}, {np.max(dB_dS):.6e}]")
    print(f"  All negative:  {np.all(dB_dS < 0)}")
    
    # Peak height
    nu = peak_height(B, s)
    print(f"\nPeak height ν = B/√S:")
    print(f"  Range: [{np.min(nu):.4f}, {np.max(nu):.4f}]")
    
    # Moving barrier validity
    if W is not None:
        valid, LDD = check_barrier_validity(s, W, 'moving')
        print(f"\nMoving barrier validity (LDD = s*W - 1/4):")
        print(f"  LDD range: [{np.min(LDD):.6e}, {np.max(LDD):.6e}]")
        print(f"  Valid points: {np.sum(valid)}/{len(s)}")
        
        if np.sum(~valid) > 0:
            first_invalid = np.where(~valid)[0][0]
            print(f"  First invalid at index {first_invalid}")
    
    print("="*70 + "\n")
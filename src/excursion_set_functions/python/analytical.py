"""
Analytical mass function formulas in excursion set theory.

Provides various analytical and semi-analytical fitting formulas for
the halo mass function derived from excursion set theory and calibrated
against N-body simulations.

Mass Function Models
--------------------
- Press-Schechter (1974): Original excursion set prediction
- Sheth-Tormen (1999): Ellipsoidal collapse corrections
- Tinker et al. (2008): Simulation-calibrated universal formula
- Musso et al. (2012): Moving barrier with stochastic corrections

Key Functions
-------------
f_ST_nu : Sheth-Tormen mass function
f_Tinker : Tinker mass function (3 or 5 parameters)
f_Musso_S : Moving barrier mass function
f_S_MB_approx : Approximate moving barrier solution

Usage
-----
These functions are used to predict the abundance of dark matter halos
as a function of mass and redshift, crucial for cosmological studies
of structure formation and galaxy evolution.

References
----------
Press & Schechter (1974), ApJ 187, 425
Sheth & Tormen (1999), MNRAS 308, 119
Tinker et al. (2008), ApJ 688, 709
Musso et al. (2012), MNRAS 427, 3145
"""
import numpy as np
from numba import jit
from scipy import special



@jit(nopython=True)
def f_ST_nu_unnorm(nu, norm, p, q):
    """
    Unnormalized Sheth-Tormen mass function.
    
    Computes the Sheth-Tormen formula without enforcing normalization.
    
    Parameters
    ----------
    nu : array_like
        Peak height δ_c/σ(M)
    norm : float
        Normalization constant
    p : float
        Power-law index parameter
    q : float
        Barrier shape parameter
        
    Returns
    -------
    array_like
        Unnormalized mass function f(ν)
        
    Notes
    -----
    The Sheth-Tormen mass function generalizes Press-Schechter by
    including ellipsoidal collapse corrections.
    """
    return norm * (1. + (q * nu * nu)**(-p)) * q**0.5 * np.exp(-q * nu * nu / 2.)

def f_ST_nu(nu, p, q):
    """
    Normalized Sheth-Tormen mass function.
    
    The Sheth-Tormen fitting formula for the halo mass function
    in excursion set theory with ellipsoidal collapse.
    
    Parameters
    ----------
    nu : array_like
        Peak height δ_c/σ(M)
    p : float
        Power-law index (typically ~0.3)
    q : float
        Barrier normalization (typically ~0.707)
        
    Returns
    -------
    array_like
        Normalized mass function f(ν) satisfying ∫ f(ν) dν = 1
        
    References
    ----------
    Sheth & Tormen (1999), MNRAS 308, 119
    """
    norm = np.sqrt(2./np.pi) / (1. + special.gamma (0.5 - p) / (2. ** p * np.sqrt(np.pi)))
    return f_ST_nu_unnorm(nu, norm, p, q)

@jit(nopython=True)
def f_Tinker(sigma, norm, a, b, c):
    """
    Tinker mass function (3-parameter version).
    
    Empirical fitting formula for halo mass function calibrated from
    N-body simulations.
    
    Parameters
    ----------
    sigma : array_like
        RMS density fluctuation σ(M)
    norm : float
        Overall normalization
    a : float
        Power-law exponent
    b : float
        Turnover scale
    c : float
        Exponential cutoff parameter
        
    Returns
    -------
    array_like
        Mass function f(σ)
        
    References
    ----------
    Tinker et al. (2008), ApJ 688, 709
    """
    return norm * ((b / sigma) ** a + 1) *  np.exp(-c / (sigma * sigma))

@jit(nopython=True)
def f_Tinker_5params(sigma, norm, p0, p1, p2, p3):
    """
    Extended Tinker mass function with 5 parameters.
    
    Generalized version allowing more flexibility in fitting.
    
    Parameters
    ----------
    sigma : array_like
        RMS density fluctuation
    norm : float
        Normalization
    p0, p1, p2, p3 : float
        Fitting parameters
        
    Returns
    -------
    array_like
        Mass function f(σ)
    """
    return norm * ((p1 / sigma) ** p0 + sigma ** (-p2)) *  np.exp(-p3 / (sigma * sigma))


def f_Tinker_normalized(sigma, p0, p1, p2, p3):
    """
    Normalized 5-parameter Tinker mass function.
    
    Automatically computes normalization to ensure ∫ f(σ) d ln σ⁻¹ = 1.
    
    Parameters
    ----------
    sigma : array_like
        RMS density fluctuation σ(M)
    p0, p1, p2, p3 : float
        Fitting parameters
        
    Returns
    -------
    array_like
        Normalized mass function
        
    Notes
    -----
    Normalization uses gamma functions for exact integration.
    """
    norm = 2. / (p1 ** p0 * p3**(-0.5 * p0) * special.gamma (0.5 * p0) + p3**(-0.5 * p2) * special.gamma (0.5 * p2))
    return norm * f_Tinker_5params(sigma, norm, p0, p1, p2, p3)



#@jit(nopython=True)
def f_Musso_S(S, Gamma2, B, dB_dS):
    """
    Musso et al. mass function with moving barrier.
    
    Excursion set theory with scale-dependent (moving) barrier.
    Accounts for barrier shape beyond simple constant threshold.
    
    Parameters
    ----------
    S : array_like
        Variance S = σ²(M)
    Gamma2 : float
        Diffusion coefficient (variance of drift)
    B : array_like
        Barrier height at scale S
    dB_dS : array_like
        Derivative dB/dS of barrier
        
    Returns
    -------
    array_like
        Mass function f(S)
        
    Notes
    -----
    Includes Markovian stochastic barrier effects. Reduces to
    standard formalism when dB/dS = 0.
    
    References
    ----------
    Musso et al. (2012), MNRAS 427, 3145
    """
    beta_star = -2.*dB_dS * S**0.5 +  B / S**0.5
    return np.exp(-0.5 * B**2 / S) / (8.*np.pi)**0.5 * beta_star / S * (
        0.5 + 0.5 * special.erf(Gamma2**0.5 * beta_star / 2**0.5) + 
        np.exp(-0.5 * Gamma2 * beta_star**2) / ((2.*np.pi * Gamma2)**0.5 * beta_star))




def f_Musso_fixed_nu(nu, Gamma2):
    """
    Musso mass function with constant barrier.
    
    Simplified version for fixed peak height with stochastic corrections.
    
    Parameters
    ----------
    nu : array_like
        Peak height ν = δ_c/σ
    Gamma2 : float
        Diffusion parameter controlling stochasticity
        
    Returns
    -------
    array_like
        Mass function f(ν)
        
    Notes
    -----
    When Gamma2 → 0, recovers Press-Schechter formula.
    """
    return np.exp(-0.5 * nu**2) / (8.*np.pi)**0.5 * (
        0.5 + 0.5 * special.erf(Gamma2**0.5 * nu / 2**0.5 )+ 
        np.exp(-0.5 * Gamma2 * nu**2) / (nu*(2.*np.pi * Gamma2)**0.5))


def f_S_MB_approx(s,W,B,dB_ds):
    """
    Approximation for moving barrier first crossing distribution.
    
    Approximate analytical solution for excursion set with moving barrier,
    combining perturbative and WKB-like approximations.
    
    Parameters
    ----------
    s : array_like
        Variance scale
    W : array_like
        Second moment of trajectory (related to variance derivative)
    B : array_like
        Barrier height
    dB_ds : array_like
        Barrier derivative
        
    Returns
    -------
    array_like
        First crossing distribution f(s)
        
    Notes
    -----
    Uses saddle-point approximation for the path integral.
    More accurate than pure perturbative expansion but less exact
    than full numerical solution.
    """

    LDD = s * W- 0.25
    D_BP = 0.5 * B / s - dB_ds
    AB2S = 0.5*s/LDD * D_BP**2
    
    I2a_S = LDD**0.5 / (4*np.pi*s) * np.exp(-0.5 / LDD * (s * dB_ds**2 + W * B**2 - B * dB_ds))
    I2b_2_S_noExp = 0.5 * D_BP * (special.erf(AB2S**0.5)+1) / (2.*np.pi*s)**0.5
        
        
    return I2a_S + (I2b_2_S_noExp + LDD**0.5/(4*np.pi*s) * np.exp(-AB2S)) * np.exp(-0.5*B**2 / s)




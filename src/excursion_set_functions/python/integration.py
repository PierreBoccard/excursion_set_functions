import numpy as np
"""
Analytical integration of power spectra with window functions.

Computes covariance matrices and variances for filtered cosmological
density fields using analytical integration methods. Combines spline
interpolation with analytical solutions for oscillatory integrals.

Key Functions
-------------
C_ij_TopHat : Covariance matrix for top-hat filtered fields
sigma2_TopHat : Variance of top-hat filtered density
dSdR_TopHat : Derivative of variance with respect to radius
cholensky_decomposition_matrix_from_Cij : Cholesky decomposition of covariance

Integration Methods
-------------------
- Low-k: Taylor series expansion
- High-k: Analytical integration of spline x trigonometric functions
- Uses sine/cosine integrals (Si, Ci) for exact results

Applications
------------
Essential for excursion set theory calculations in cosmology:
- Computing halo mass functions
- Predicting cluster abundances
- Understanding structure formation

References
----------
Window function formalism: Bardeen et al. (1986), ApJ 304, 15
"""
from numba import jit, prange
from numba.core import types
#from numba.typed import Dict
#from interpolation.splines import UCGrid, CGrid, nodes, eval_linear, filter_cubic, eval_cubic
#import pandas
#from scipy import integrate, interpolate, optimize, signal
import scipy.special as sp
#from multiprocessing import Pool, Process, Manager
#from multiprocessing.pool import ThreadPool
from .spline import *

float_array = types.float64[::1]

def cholensky_decomposition_matrix_from_Cij(Cij,Cijtype='numpy_narray',out_type='linearized'):
    """
    Compute Cholesky decomposition of covariance matrix.
    
    Decomposes covariance matrix C as C = L L^T where L is lower triangular.
    Used to generate correlated random walks in excursion set simulations.
    
    Parameters
    ----------
    Cij : ndarray or dict
        Covariance matrix, either as 2D array or dictionary
    Cijtype : str, optional
        Input format: 'numpy_narray' or 'dict'
    out_type : str, optional
        Output format: 'linearized' (1D array) or dict
        
    Returns
    -------
    L_ij_lin : ndarray or dict
        Lower triangular Cholesky factor
        If out_type='linearized', returns 1D array of length n(n+1)/2
        
    Notes
    -----
    Linearized output stores only lower triangular elements in row-major order,
    reducing memory usage for large matrices.
    """
    if Cijtype == 'numpy_narray':
        lenR = Cij.shape[0]
        L_ij = dict()
        for i in range(0,lenR):
            L_ij[i] = np.zeros(i+1)
        for j in range(0,lenR):
            L_ij[j][j] = (Cij[j,j] - np.sum(L_ij[j][:j] ** 2)) ** 0.5
            for i in range(j+1, lenR):
                L_ij[i][j] = (Cij[i,j] - np.sum(L_ij[i][:j] * L_ij[j][:j])) / L_ij[j][j]
    elif Cijtype == 'dict':
        L_ij = Dict.empty(types.int64, float_array)
        for i in Cij.keys():
            L_ij[i] = np.zeros(i+1)
        lenR = i + 1
        for j in range(0,lenR):
            L_ij[j][j] = (Cij[j][j] - np.sum(L_ij[j][:j] ** 2)) ** 0.5
            for i in range(j+1, lenR):
                L_ij[i][j] = (Cij[i][j] - np.sum(L_ij[i][:j] * L_ij[j][:j])) / L_ij[j][j]
    else:
        return -1
    if out_type == 'linearized':
        L_ij_lin = np.zeros(int((lenR+1)*lenR/2))
        j0 = 0
        for i in range(lenR):
            for j in range(0, i + 1):
                L_ij_lin[j0 + j] = L_ij[i][j]
            j0 += i + 1
        return L_ij_lin
    return L_ij



@jit(nopython=True,cache=True)
def TOPHAT_Taylor(x):
    """
    Taylor series expansion of top-hat window function for small x.
    
    Computes W(x)² where W(x) = 3(sin(x) - x*cos(x))/x³ for small x
    to avoid numerical instability near x=0.
    
    Parameters
    ----------
    x : float or array_like
        Argument (typically k*R where k is wavenumber, R is filter radius)
        
    Returns
    -------
    float or array_like
        Top-hat window function squared, accurate for |x| < 0.1
        
    Notes
    -----
    Series expansion to 8th order provides sufficient accuracy.
    """
    return 1. - x**2/10. +x**4/280. - x**6/15120. +x**8/1330560.

@jit(nopython=True,cache=True)
def dx_squareTOPHAT_Taylor(x):
    """
    Taylor expansion of derivative of squared top-hat window function.
    
    Computes d(W²)/dx for small x to avoid numerical instability.
    
    Parameters
    ----------
    x : float or array_like
        Argument value(s)
        
    Returns
    -------
    float or array_like
        Derivative of W²(x) with respect to x
    """
    return 2. * (-x/5. +x**3/70. - x**5/2520. +x**7/166320.) * (1. - x**2/10. +x**4/280. - x**6/15120. +x**8/1330560.)

@jit(nopython=True,cache=True)
def top_hat_rk_HR(r, k,X0=1e-1):
    """
    Real-space top-hat window function in Fourier space.
    
    Computes W(kR) = 3(sin(kR) - kR*cos(kR))/(kR)³
    Uses Taylor expansion for small kR to avoid numerical issues.
    
    Parameters
    ----------
    r : float
        Filter radius
    k : array_like
        Wavenumber array
    X0 : float, optional
        Threshold for switching to Taylor series (default 0.1)
        
    Returns
    -------
    ndarray
        Window function values at each k
        
    Notes
    -----
    The top-hat window in real space becomes this sinc-like function
    in Fourier space. Essential for computing power spectrum filtering.
    """
    x = r * k
    OUT = np.zeros(len(x))
    OUT[x <= X0] = TOPHAT_Taylor(x[x <= X0])
    OUT[x > X0] = 3. * (np.sin(x[x > X0]) - x[x > X0] * np.cos(x[x > X0])) / (x[x > X0] ** 3.)
    return OUT


@jit(nopython=True,cache=True)
def dr_square_top_hat_rk_HR(r, k,X0=1e-1):
    """
    Derivative of squared top-hat window function with respect to radius.
    
    Computes d(W²(kR))/dR = k * d(W²)/d(kR) for variance derivatives.
    
    Parameters
    ----------
    r : float
        Filter radius
    k : array_like
        Wavenumber array
    X0 : float, optional
        Threshold for Taylor expansion (default 0.1)
        
    Returns
    -------
    ndarray
        Derivative values at each k
        
    Notes
    -----
    Used in computing dσ²/dR, which relates first crossing rates
    to halo mass in excursion set theory.
    """
    x = r * k
    x = r * k
    OUT = np.zeros(len(x))
    mask = (x <= X0)
    OUT[mask] = dx_squareTOPHAT_Taylor(x[mask])
    mask[:] = ~mask
    OUT[mask] = 18. * ((x[mask]**2-3.) * np.sin(x[mask])**2 - 3. * x[mask]**2 * np.cos(x[mask])**2 + (6.*x[mask] - x[mask]**3) * np.sin(x[mask]) * np.cos(x[mask])) / x[mask] ** 7
    return OUT * k

@jit(nopython=True,cache=True)
def Factorial(N):
    """
    Compute factorial N! for integer N.
    
    Parameters
    ----------
    N : int
        Non-negative integer
        
    Returns
    -------
    int
        N! = N * (N-1) * ... * 2 * 1
    """
    if N <= 1:
        return 1
    return np.prod(np.arange(N)+1)

@jit(nopython=True,cache=True)
def a_fac(pow):
    """
    Compute coefficient for Taylor expansion of window function product.
    
    Returns the coefficient a_n in the small-k expansion of W(k*R1)*W(k*R2).
    
    Parameters
    ----------
    pow : int
        Power index (must be even)
        
    Returns
    -------
    float
        Taylor coefficient
        
    Notes
    -----
    Only even powers contribute; odd powers return 0.
    """
    if pow % 2 != 0:
        return 0.
    elif pow == 0:
        return 1.
    N = pow + 3
    return 3. * (N-1) / Factorial(N) * (-1)**((N + 1) / 2)

@jit(nopython=True,cache=True)
def IntegrationSmallK(Pk0,k0,n,R1,R2,Omax=8):
    """
    Compute small-k contribution to covariance integral via Taylor expansion.
    
    For k → 0, uses power series expansion of window functions to
    analytically integrate Pk(k) * k^(n+2) * W(k*R1) * W(k*R2).
    
    Parameters
    ----------
    Pk0 : float
        Power spectrum normalization at k=k0
    k0 : float
        Reference wavenumber (typically smallest k in array)
    n : float
        Power spectrum spectral index (P(k) ∝ k^n at low k)
    R1, R2 : float
        Filter radii
    Omax : int, optional
        Maximum order of Taylor expansion (default 8)
        
    Returns
    -------
    float
        Integral contribution from k < k0
        
    Notes
    -----
    This avoids numerical issues from oscillatory window functions
    at small k*R. Higher Omax improves accuracy but is rarely needed.
    """
    ALPHA = np.zeros(9)
    ALPHA[0] = 1.
    ALPHA[2] = a_fac(2)
    ALPHA[4] = a_fac(4) * (R1**4 + R2 **4) + \
               a_fac(2) ** 2 * R1 ** 2 * R2 **2
    ALPHA[6] = a_fac(6) * (R1 ** 6 + R2 ** 6) + \
               a_fac(2) * a_fac(4) * R1 ** 2 * R2 ** 2 * (R1 ** 4 + R2 ** 4)
    ALPHA[8] = a_fac(8) * (R1 ** 8 + R2 ** 8) + \
               a_fac(2) * a_fac(6) * R1 ** 2 * R2 ** 2 * (R1 ** 6 + R2 ** 6) + \
               a_fac(4) ** 2 * R1 ** 4 * R2 ** 4
    return Pk0 / (2.*np.pi**2) * np.sum(ALPHA[:Omax+1] * k0 ** (np.arange(Omax+1) + 3) / (np.arange(Omax+1) + 3 + n))
    #return Pk0 / (2.*np.pi**2) * np.sum(ALPHA[:Omax+1] * k0 ** 3 / (np.arange(Omax+1) + 3 + n))





##########
def C_ij_TopHat_MAIN(Coeffs, x, a, b):
    """
    Compute covariance integral for high-k regime using analytical methods.
    
    Analytically integrates cubic spline representation of power spectrum
    multiplied by oscillatory window functions using sine/cosine integrals.
    
    Parameters
    ----------
    Coeffs : ndarray
        Cubic spline coefficients for P(k) in explicit form (n-1, 4)
    x : array_like
        Wavenumber array (knot positions)
    a : float
        Inverse of first filter radius (1/R1)
    b : float
        Inverse of second filter radius (1/R2)
        
    Returns
    -------
    float
        Covariance integral contribution C_ij for k > threshold
        
    Notes
    -----
    Uses analytical formulas involving Si(x) and Ci(x) (sine and cosine
    integrals) to exactly integrate products of polynomials with sin/cos.
    This is much more accurate than numerical quadrature for oscillatory
    integrands and avoids issues with adaptive integration.
    
    The formulation handles both cross-correlations (a≠b) and 
    auto-correlations (a=b) with appropriate limiting behavior.
    """
    #INT0 = 0.
    #if SmallK_args == 0:
    #    INT0 = 0
    #else:
    #    INT0 = IntegrationSmallK(Pk[0], k[0], n, R1, R2, Omax=OMAX)

    Si_diff, Ci_diff = sp.sici((a - b) * x)
    Si_sum, Ci_sum = sp.sici((a + b) * x)
    cos_diff = np.cos((a - b) * x)
    cos_sum = np.cos((a + b) * x)
    sin_diff = np.sin((a - b) * x)
    sin_sum = np.sin((a + b) * x)

    '''
    T1_C3 = 0.5 * (Ci_diff - Ci_sum)
    T1_C2 = 0.5 * ((b-a) * Si_diff + (a+b) * Si_sum - cos_diff / x + cos_sum / x)
    T1_C1 = 0.25*(-(a-b) ** 2 * Ci_diff + (a+b) ** 2 * Ci_sum - cos_diff / x ** 2 + cos_sum / x ** 2 + (a-b) * sin_diff / x - (a+b) * sin_sum / x )
    T1_C0  = ((a-b)**3 * Si_diff + ((a-b)**2*x**2-2.) * cos_diff/ x**3 + (a-b) * sin_diff / x ** 2 - 
              (a+b)**3 * Si_sum - ((a+b)**2*x**2-2.) * cos_sum / x**3 - (a+b) * sin_sum / x**2) / 12
    
    T2_C3 = 0.5 * (x * sin_diff / (a-b) + cos_diff / (a-b) ** 2 + 
                   x * sin_sum / (a+b) + cos_sum / (a+b) ** 2)
    T2_C2 = 0.5 * (sin_diff / (a-b) + sin_sum / (a+b))
    T2_C1 = 0.5 * (Ci_diff + Ci_sum)
    T2_C0 = -0.5 * ((a-b) * Si_diff + (a+b) * Si_sum + cos_diff / x + cos_sum / x)

    T3_C3 = -0.5 * (cos_diff / (a-b) + cos_sum / (a+b))
    T3_C2 = 0.5 * (Si_diff + Si_sum)
    T3_C1 = 0.5 * ((a-b) * Ci_diff - sin_diff / x + 
                   (a+b) * Ci_sum - sin_sum / x)
    T3_C0 = 0.25 * (-(a-b)**2 * Si_diff - (a-b) * cos_diff / x - sin_diff / x ** 2 -
                     (a+b)**2 * Si_sum - (a+b) * cos_sum / x - sin_sum / x ** 2)

    T4_C3 = 0.5 * (cos_diff / (a-b) - cos_sum / (a+b))
    T4_C2 = 0.5 * (-Si_diff + Si_sum)
    T4_C1 = 0.5 * ((b-a) * Ci_diff + sin_diff / x + 
                   (a+b) * Ci_sum - sin_sum / x)
    T4_C0 = 0.25 * ((a-b)**2 * Si_diff + (a-b) * cos_diff / x + sin_diff / x ** 2 -
                    (a+b)**2 * Si_sum - (a+b) * cos_sum / x - sin_sum / x ** 2)
    '''

    Coeff_tot_3 = 0.5 * (Ci_diff - Ci_sum + a*b*x*(sin_diff / (a-b) + sin_sum / (a+b)) + 
                         (a*b / (a-b)**2 - 1) * cos_diff + (a*b / (a+b)**2 + 1) * cos_sum)
    Coeff_tot_2 = 0.5 * ((cos_sum - cos_diff) / x + a*b / (a-b) * sin_diff + a*b / (a+b) * sin_sum)
    Coeff_tot_1 = 0.25 * ((a**2 + b**2) * (Ci_diff - Ci_sum) - ((a-b) * sin_diff - (a+b) * sin_sum) / x - (cos_diff - cos_sum) / x**2 )
    Coeff_tot_0 = (-(a**3 - b**3) * Si_diff + (a**3 + b**3) * Si_sum - (a-b) * sin_diff/x**2 + (a+b) * sin_sum/x**2 -
                    ((a**2+b**2+a*b)/x + 1./x**3) * cos_diff + ((a**2+b**2-a*b)/x + 1./x**3) * cos_sum) / 6.
    
    #check_3 = T1_C3 / (a**3*b**3) + T2_C3 / (a**2*b**2) - T3_C3 / (a**3*b**2) - T4_C3 / (a**2*b**3)
    #check_2 = T1_C2 / (a**3*b**3) + T2_C2 / (a**2*b**2) - T3_C2 / (a**3*b**2) - T4_C2 / (a**2*b**3)
    #check_1 = T1_C1 / (a**3*b**3) + T2_C1 / (a**2*b**2) - T3_C1 / (a**3*b**2) - T4_C1 / (a**2*b**3)
    #check_0 = T1_C0 / (a**3*b**3) + T2_C0 / (a**2*b**2) - T3_C0 / (a**3*b**2) - T4_C0 / (a**2*b**3)

    INT0 = 0.
    INT0 += np.sum((Coeff_tot_3[1:] - Coeff_tot_3[:-1]) * Coeffs[:,3])
    INT0 += np.sum((Coeff_tot_2[1:] - Coeff_tot_2[:-1]) * Coeffs[:,2])
    INT0 += np.sum((Coeff_tot_1[1:] - Coeff_tot_1[:-1]) * Coeffs[:,1])
    INT0 += np.sum((Coeff_tot_0[1:] - Coeff_tot_0[:-1]) * Coeffs[:,0])
    INT0 *= 9. / (2. * np.pi**2) / (a**3*b**3)

    return INT0





    

def C_ii_TopHat_MAIN(
        Coeffs, x, a):
    """
    Compute variance integral for high-k regime (diagonal covariance case).
    
    Specialized version of C_ij_TopHat_MAIN for auto-correlation (i=j),
    taking advantage of simplifications when both radii are equal.
    
    Parameters
    ----------
    Coeffs : ndarray
        Cubic spline coefficients for P(k)
    x : array_like
        Wavenumber array
    a : float
        Inverse filter radius (1/R)
        
    Returns
    -------
    float
        Variance integral σ²(R) for k > threshold
        
    Notes
    -----
    Uses Si(2ax) and Ci(2ax) integrals. More efficient than general
    C_ij_TopHat_MAIN due to symmetry simplifications.
    """
    #INT0 = 0.
    #if SmallK_args == 0:
    #    INT0 = 0
    #else:
    #    INT0 = IntegrationSmallK(Pk[0], k[0], n, R1, R2, Omax=OMAX)
    Si_2ax, Ci_2ax = sp.sici(2. * a * x)
    cos_2ax = np.cos(2. * a * x)
    sin_2ax = np.sin(2. * a * x)
    lnx = np.log(x)

    #T1_C3 = 0.5 * (lnx - Ci_2ax)
    #T1_C2 = a * Si_2ax + (cos_2ax - 1.) / (2. * x)
    #T1_C1 = a * a * Ci_2ax - a * sin_2ax / (2. * x) + (cos_2ax - 1.) / (4. * x * x)
    #T1_C0  = -2. * a*a*a / 3. * Si_2ax - (a * a/(3. * x) - 1. / (6. * x**3)) * cos_2ax - a * sin_2ax / (6. * x**2) - 1. / (6. * x**3)
    
    #T2_C3 = cos_2ax / (8. * a*a) + x * sin_2ax / (4. * a) + 0.25 * x ** 2
    #T2_C2 = 0.5 * x + sin_2ax / (4. * a)
    #T2_C1 = 0.5 * Ci_2ax + 0.5 * lnx
    #T2_C0 = -a * Si_2ax - (cos_2ax + 1.) / (2. * x)

    #T3_C3 = -cos_2ax / (2. * a)
    #T3_C2 = Si_2ax
    #T3_C1 = 2. * a * Ci_2ax - sin_2ax / x
    #T3_C0 = -2. * a ** 2 * Si_2ax - sin_2ax / (2. * x**2) - a * cos_2ax / x
    
    
    Coeff_tot_3 = (0.5 * lnx - 0.5 * Ci_2ax + 5. / 8. * cos_2ax + 0.25 * a * x * sin_2ax + 0.25 * a**2 * x**2)
    Coeff_tot_2 = ((cos_2ax - 1.) / (2. * x) + 0.5 * a ** 2 * x + 0.25 * a * sin_2ax)
    Coeff_tot_1 = (0.5 * a**2 * (lnx - Ci_2ax) + 0.5 * a * sin_2ax / x + 0.25 * (cos_2ax - 1.) / x**2)
    Coeff_tot_0 = (a**3 * Si_2ax / 3. + (cos_2ax - 3.) * a**2 / (6. * x) + a * sin_2ax / (3. * x**2) + (cos_2ax - 1.) / (6. * x**3))


    #check_3 = T1_C3 / a**6 + T2_C3 / a**4 - T3_C3 / a**5
    #check_2 = T1_C2 / a**6 + T2_C2 / a**4 - T3_C2 / a**5
    #check_1 = T1_C1 / a**6 + T2_C1 / a**4 - T3_C1 / a**5
    #check_0 = T1_C0 / a**6 + T2_C0 / a**4 - T3_C0 / a**5

    INT0 = 0.
    INT0 += np.sum((Coeff_tot_3[1:] - Coeff_tot_3[:-1]) * Coeffs[:,3])
    INT0 += np.sum((Coeff_tot_2[1:] - Coeff_tot_2[:-1]) * Coeffs[:,2])
    INT0 += np.sum((Coeff_tot_1[1:] - Coeff_tot_1[:-1]) * Coeffs[:,1])
    INT0 += np.sum((Coeff_tot_0[1:] - Coeff_tot_0[:-1]) * Coeffs[:,0])
    INT0 *= 9. / (2. * np.pi**2) / a ** 6
    

    return INT0


def C_ij_TopHat_MAIN_lowR(
        Pk, k, R1, R2, IDchange, OMAX, n):
    """
    Compute covariance integral for low-k regime using direct integration.
    
    For k*R << 1, window functions are smooth and non-oscillatory, allowing
    simple polynomial integration of the spline.
    
    Parameters
    ----------
    Pk : array_like
        Power spectrum values
    k : array_like
        Wavenumber array
    R1, R2 : float
        Filter radii
    IDchange : int
        Index where high-k regime begins
    OMAX : int
        Order for small-k Taylor expansion
    n : float
        Spectral index
        
    Returns
    -------
    float
        Covariance for k < k_threshold
        
    Notes
    -----
    Combines Taylor expansion at very low k with spline integration
    up to the transition scale where oscillatory methods take over.
    """
    
    coeffs = cubic_spline_coeffs(k, Pk * k**2 * top_hat_rk_HR(R1, k) * top_hat_rk_HR(R2, k))
    if OMAX <= 0:
        INT0 = 0.
    else:
        INT0 = IntegrationSmallK(Pk[0], k[0], n, R1, R2, Omax=OMAX)

    INT0 += np.sum((coeffs[:IDchange,0] + 
                    coeffs[:IDchange,1] / 2. +
                    coeffs[:IDchange,2] / 3. + 
                    coeffs[:IDchange,3] / 4.) * (k[1:IDchange+1] -  k[:IDchange]))
    return INT0 / (2. * np.pi ** 2)


def C_ii_TopHat_MAIN_lowR(
        Pk, k, R, IDchange, OMAX, n):
    """
    Compute variance integral for low-k regime.
    
    Diagonal (auto-correlation) version of C_ij_TopHat_MAIN_lowR.
    
    Parameters
    ----------
    Pk : array_like
        Power spectrum
    k : array_like
        Wavenumber array
    R : float
        Filter radius
    IDchange : int
        Transition index to high-k regime
    OMAX : int
        Taylor expansion order
    n : float
        Spectral index
        
    Returns
    -------
    float
        Variance for k < k_threshold
    """
    
    coeffs = cubic_spline_coeffs(k, Pk * k**2 * top_hat_rk_HR(R, k)**2)
    if OMAX <= 0:
        INT0 = 0
    else:
        INT0 = IntegrationSmallK(Pk[0], k[0], n, R, R, Omax=OMAX)

    INT0 += np.sum((coeffs[:IDchange,0] + 
                    coeffs[:IDchange,1] / 2. +
                    coeffs[:IDchange,2] / 3. + 
                    coeffs[:IDchange,3] / 4.) * (k[1:IDchange+1] -  k[:IDchange]))
    return INT0 / (2. * np.pi ** 2)

@jit(nopython=True)
def find_id_change(x_array,x_max):
    """
    Find index where x_array first exceeds x_max.
    
    Binary-like search to determine transition point between low-k
    and high-k integration regimes.
    
    Parameters
    ----------
    x_array : array_like
        Sorted array to search
    x_max : float
        Threshold value
        
    Returns
    -------
    int
        Index where x_array[ind] >= x_max, or len(x_array)-1 if never
        
    Notes
    -----
    Used to split k-space into regions where different integration
    methods are optimal.
    """
    ind = 0
    len_arr_mn1 = len(x_array) - 1
    while (x_array[ind] < x_max) & (ind < len_arr_mn1):
        ind += 1
    return ind


def C_ij_apply_async(i,j,Cij_out,R,k,Pk,Coeffs,OmaxSmallK, n):
    """
    Compute single covariance matrix element (helper for parallelization).
    
    Computes C_ij for a specific pair of radii. Can be used with
    multiprocessing to parallelize covariance matrix computation.
    
    Parameters
    ----------
    i, j : int
        Indices of radii pair
    Cij_out : ndarray
        Output covariance matrix (modified in place)
    R : array_like
        Array of filter radii
    k : array_like
        Wavenumber array
    Pk : array_like
        Power spectrum
    Coeffs : ndarray
        Spline coefficients
    OmaxSmallK : int
        Small-k expansion order
    n : float
        Spectral index
        
    Notes
    -----
    Updates Cij_out[i,j] and Cij_out[j,i] with computed covariance.
    Symmetric matrix structure is exploited.
    """
    IDchange = find_id_change(k,(R[i] * R[j]) ** -0.5)
    Cij_out[i,j] = C_ij_TopHat_MAIN_lowR(Pk, k, R[i], R[j], IDchange, OmaxSmallK, n)
    Cij_out[i,j] += C_ij_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i],R[j])
    Cij_out[j,i] = Cij_out[i,j]



def C_ij_TopHat(Pk, k, R, n=0.96, OmaxSmallK=-1):
    """
    Compute covariance matrix for top-hat filtered density field.
    
    Calculates C_ij = ∫ Pk(k) W(k*R_i) W(k*R_j) k² dk / (2π²)
    where W is the top-hat window function. This covariance is used in
    excursion set theory for correlated random walks.
    
    Parameters
    ----------
    Pk : array_like
        Power spectrum values
    k : array_like
        Wavenumber array
    R : array_like
        Array of filter radii
    n : float, optional
        Power spectrum index for small-k approximation (default 0.96)
    OmaxSmallK : int, optional
        Maximum order for small-k Taylor expansion (default -1, no expansion)
        
    Returns
    -------
    Cij_out : ndarray
        Covariance matrix (len(R), len(R))
        
    Notes
    -----
    Uses cubic spline interpolation of the power spectrum and analytical
    integration of oscillatory integrals for efficiency and accuracy.
    """

    Coeffs = cubic_spline_coeffs(k,Pk)
    explicit_from_implicit_coeffs(Coeffs,k)

    #dk  = k[1:] - k[:-1]
    #x0_dx = k[:-1] / dx

    Cij_out = np.zeros((len(R),len(R)))
    for i in range(len(R)):
        for j in range(i):
            IDchange = find_id_change(k,(R[i] * R[j]) ** -0.5)
            Cij_out[i,j] = C_ij_TopHat_MAIN_lowR(Pk, k, R[i], R[j], IDchange, OmaxSmallK, n)
            Cij_out[i,j] += C_ij_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i],R[j])
            #print(i,j,' id:',IDchange,'t1:',C_ij_TopHat_MAIN_lowR(Pk, k, R[i], R[j], IDchange, OmaxSmallK, n),'t2:',C_ij_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i],R[j]),'tot,',Cij_out[i,j])
            Cij_out[j,i] = Cij_out[i,j]
        IDchange = find_id_change(k,1./R[i])
        Cij_out[i,i] = C_ii_TopHat_MAIN_lowR(Pk, k, R[i], IDchange, OmaxSmallK, n)
        Cij_out[i,i] += C_ii_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i])
        #print(i,' id:',IDchange,'t1:',C_ii_TopHat_MAIN_lowR(Pk, k, R[i], IDchange, OmaxSmallK, n),'t2:',C_ii_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i]),'tot,',Cij_out[i,i])

        
    return Cij_out
    

def sigma2_TopHat(
        Pk, k, R, n=0.96,OmaxSmallK=-1):
    """
    Compute variance of top-hat filtered density field.
    
    Calculates σ²(R) = ∫ Pk(k) W²(k*R) k² dk / (2π²)
    This is the variance of the density contrast smoothed on scale R.
    
    Parameters
    ----------
    Pk : array_like
        Power spectrum P(k)
    k : array_like
        Wavenumber array
    R : array_like
        Filter radius array
    n : float, optional
        Spectral index for small-k (default 0.96)
    OmaxSmallK : int, optional
        Taylor expansion order for small-k (default -1)
        
    Returns
    -------
    OUT : ndarray
        Variance σ²(R) at each radius
        
    Notes
    -----
    This is equivalent to the diagonal of C_ij_TopHat but computed
    more efficiently. Central quantity in excursion set theory.
    """

    Coeffs = cubic_spline_coeffs(k,Pk)
    explicit_from_implicit_coeffs(Coeffs,k)

    OUT = np.zeros(len(R))
    for i in range(0,len(R)):
        IDchange = find_id_change(k,1./R[i])
        OUT[i] = C_ii_TopHat_MAIN_lowR(Pk, k, R[i], IDchange, OmaxSmallK, n)
        OUT[i] += C_ii_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i])
    return OUT


def sigma2_2_TopHat_numdiff(
        Pk, k, R, n=0.96,OmaxSmallK=-1,dRperc=5e-3):
    """
    Compute second moment of variance using numerical differentiation.
    
    Approximates d²σ²/dR² using finite differences. This quantity appears
    in higher-order excursion set calculations.
    
    Parameters
    ----------
    Pk : array_like
        Power spectrum
    k : array_like
        Wavenumber array
    R : array_like
        Filter radii
    n : float, optional
        Spectral index (default 0.96)
    OmaxSmallK : int, optional
        Small-k expansion order (default -1)
    dRperc : float, optional
        Relative step size for differentiation (default 5e-3 = 0.5%)
        
    Returns
    -------
    OUT : ndarray
        Second derivative d²σ²/dR² at each radius
        
    Notes
    -----
    Uses 3-point stencil for second derivative. Accuracy depends on dRperc.
    """

    Coeffs = cubic_spline_coeffs(k,Pk)
    explicit_from_implicit_coeffs(Coeffs,k)

    grid_R1R2 = np.zeros((len(R),3))
    for i in range(0,len(R)):
        j=0 #--
        IDchange = find_id_change(k,1./R[i])
        grid_R1R2[i,j] = C_ii_TopHat_MAIN_lowR(Pk, k, R[i]*(1.-dRperc), IDchange, OmaxSmallK, n)
        grid_R1R2[i,j] += C_ii_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i]*(1.-dRperc))
        j=1 #+-
        grid_R1R2[i,j] = C_ij_TopHat_MAIN_lowR(Pk, k, R[i]*(1.+dRperc), R[i]*(1.-dRperc), IDchange, OmaxSmallK, n)
        grid_R1R2[i,j] += C_ij_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i]*(1.+dRperc), R[i]*(1.-dRperc))
        j=2 #++
        grid_R1R2[i,j] = C_ii_TopHat_MAIN_lowR(Pk, k, R[i]*(1.+dRperc), IDchange, OmaxSmallK, n)
        grid_R1R2[i,j] += C_ii_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i]*(1.+dRperc))
    #grid_dR1R2 = np.zeros(grid_R1R2.shape)
    #grid_dR1R2[:,0] = (grid_R1R2[:,2] - grid_R1R2[:,1]) / (2. * R*dRperc)
    #grid_dR1R2[:,1] = (grid_R1R2[:,1] - grid_R1R2[:,0]) / (2. * R*dRperc)
    OUT = 0.25 * (grid_R1R2[:,2] - 2. * grid_R1R2[:,1] + grid_R1R2[:,0]) / (R*dRperc)**2
    return OUT



def sigma2_d2_2_TopHat_numdiff(
        Pk, k, R, accuracy=1, n=0.96,OmaxSmallK=-1,dRperc=5e-3):
    """
    High-accuracy computation of fourth moment of variance.
    
    Computes d⁴σ²/dR⁴ using high-order finite difference stencils.
    Used in advanced excursion set formalism with barrier corrections.
    
    Parameters
    ----------
    Pk : array_like
        Power spectrum
    k : array_like
        Wavenumber array
    R : array_like
        Filter radii
    accuracy : int, optional
        Accuracy order: 1-4 (default 1)
        Higher values use more points for better accuracy
    n : float, optional
        Spectral index (default 0.96)
    OmaxSmallK : int, optional
        Small-k expansion order (default -1)
    dRperc : float, optional
        Relative step size (default 5e-3)
        
    Returns
    -------
    OUT : ndarray
        Fourth derivative at each radius
        
    Notes
    -----
    Accuracy levels:
    - 1: 3-point stencil
    - 2: 5-point stencil  
    - 3: 7-point stencil
    - 4: 9-point stencil
    """

    Coeffs = cubic_spline_coeffs(k,Pk)
    explicit_from_implicit_coeffs(Coeffs,k)

    OUT = np.zeros(len(R))
    if accuracy < 1:
        accuracy = 1
    if accuracy == 1:
        c_deriv = np.array([1.,-2.,1.])
    elif accuracy == 2:
        c_deriv = np.array([-1./12.,4./3.,-5./2.,4./3.,-1./12.])
    elif accuracy == 3:
        c_deriv = np.array([1./90., -3./20., 3./2.,-49./18.,3./2., -3./20., 1./90.])
    else:
        accuracy = 4
        c_deriv = np.array([-1./560., 8./315.,-1./5., 8./5., -205./72., 8./5.,-1./5., 8./315., -1./560.])

    nderiv = 2 * accuracy + 1
    for i in range(0,len(R)):
        IDchange = find_id_change(k,1./R[i])
        for jj in range(nderiv):
            j_h = jj - accuracy
            OUT[i] += c_deriv[jj]**2 * (
                C_ii_TopHat_MAIN_lowR(Pk, k, R[i]*(1. + j_h * dRperc), IDchange, OmaxSmallK, n) + 
                C_ii_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i]*(1. + j_h * dRperc)))
            for kk in range(jj):
                k_h = kk - accuracy
                OUT[i] += 2. * c_deriv[jj] * c_deriv[kk] * (
                    C_ij_TopHat_MAIN_lowR(Pk, k, R[i] * (1. + k_h * dRperc), R[i] * (1. + j_h * dRperc), IDchange, OmaxSmallK, n) + 
                    C_ij_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i] * (1. + k_h * dRperc), R[i] * (1. + j_h * dRperc)))
        OUT[i] /= (R[i]*dRperc)**4
    return OUT








def dSdR_TopHat_MAIN_lowR(Pk,k, R, IDchange, OMAX, n):
    """
    Compute variance derivative for low-k regime.
    
    Integrates dσ²/dR using window function derivative for non-oscillatory
    part of k-space.
    
    Parameters
    ----------
    Pk : array_like
        Power spectrum
    k : array_like
        Wavenumber array
    R : float
        Filter radius
    IDchange : int
        Transition index
    OMAX : int
        Small-k expansion order
    n : float
        Spectral index
        
    Returns
    -------
    float
        Low-k contribution to dσ²/dR
    """
    

    Pk_k2_w1w2 = Pk * k * k * dr_square_top_hat_rk_HR(R,k)
    
    coeffs = cubic_spline_coeffs(k, Pk_k2_w1w2)

    
    INT0 = 0.
    if (OMAX > 0):
        INT0 += IntegrationSmallK(Pk[0], k[0], n, R, R, OMAX)

    INT0 += np.sum((coeffs[:IDchange,0] + 
                    coeffs[:IDchange,1] / 2. +
                    coeffs[:IDchange,2] / 3. + 
                    coeffs[:IDchange,3] / 4.) * (k[1:IDchange+1] -  k[:IDchange]))
    #for i in range(IDchange):
    #    INT0 += (coeffs[i][0] + 
    #             coeffs[i][1] / 2. +
    #             coeffs[i][2] / 3. + 
    #             coeffs[i][3] / 4.) * (k[i+1] -  k[i])
    
    
    INT0 /= 2. * np.pi * np.pi
    
    return INT0


def dSdR_TopHat_MAIN(coeffs, x, a, IDchange):
    """
    Compute variance derivative for high-k regime using analytical methods.
    
    Integrates dσ²/dR analytically using spline coefficients and
    trigonometric integrals. Includes both direct integral and the
    term from differentiating 1/R⁶ factor.
    
    Parameters
    ----------
    coeffs : ndarray
        Cubic spline coefficients for P(k)
    x : array_like
        Wavenumber array
    a : float
        Inverse radius 1/R
    IDchange : int
        Starting index for high-k regime
        
    Returns
    -------
    float
        High-k contribution to dσ²/dR
        
    Notes
    -----
    Derivative includes two terms:
    1. Derivative of window function (direct)
    2. Derivative of 1/R⁶ normalization factor
    """

    a2 = a*a
    a3 = a*a*a
    a6 = a3*a3

    Si_2ax, Ci_2ax = sp.sici(2. * a * x[IDchange:])
    lnx = np.log(x[IDchange:])
    sin_2ax = np.sin(2. * a * x[IDchange:])
    cos_2ax = np.cos(2. * a * x[IDchange:])

    Coeff_tot_3 = (0.5 * lnx - 0.5 * Ci_2ax + 5. / 8. * cos_2ax + 0.25 * a * x[IDchange:] * sin_2ax + 0.25 * a2 * x[IDchange:]*x[IDchange:])
    Coeff_tot_2 = ((cos_2ax - 1.) / (2. * x[IDchange:]) + 0.5 * a2 * x[IDchange:] + 0.25 * a * sin_2ax)
    Coeff_tot_1 = (0.5 * a2 * (lnx - Ci_2ax) + 0.5 * a * sin_2ax / x[IDchange:] + 0.25 * (cos_2ax - 1.) / (x[IDchange:]*x[IDchange:]))
    Coeff_tot_0 = (a3 * Si_2ax / 3. + (cos_2ax - 3.) * a2 / (6. * x[IDchange:]) + a * sin_2ax / (3. * x[IDchange:]*x[IDchange:]) + (cos_2ax - 1.) / (6. * x[IDchange:]*x[IDchange:]*x[IDchange:]))

    da_Coeff_tot_3 = 0.5 * (a*x[IDchange:]*x[IDchange:] - 1./a) * cos_2ax - x[IDchange:]*sin_2ax + 0.5*a*x[IDchange:]*x[IDchange:]
    da_Coeff_tot_2 = 0.5 * a*x[IDchange:] * cos_2ax - 0.75 * sin_2ax + a*x[IDchange:]
    da_Coeff_tot_1 = a * (lnx - Ci_2ax + 0.5*cos_2ax)
    da_Coeff_tot_0 = a2 * Si_2ax  + a * (cos_2ax - 1.) / x[IDchange:]

    INT0 = np.sum((da_Coeff_tot_3[1:] - da_Coeff_tot_3[:-1]) * coeffs[IDchange:,3] + 
                  (da_Coeff_tot_2[1:] - da_Coeff_tot_2[:-1]) * coeffs[IDchange:,2] + 
                  (da_Coeff_tot_1[1:] - da_Coeff_tot_1[:-1]) * coeffs[IDchange:,1] + 
                  (da_Coeff_tot_0[1:] - da_Coeff_tot_0[:-1]) * coeffs[IDchange:,0])
    INT0 -= 6./a * np.sum((Coeff_tot_3[1:] - Coeff_tot_3[:-1]) * coeffs[IDchange:,3] +
                          (Coeff_tot_2[1:] - Coeff_tot_2[:-1]) * coeffs[IDchange:,2] + 
                          (Coeff_tot_1[1:] - Coeff_tot_1[:-1]) * coeffs[IDchange:,1] + 
                          (Coeff_tot_0[1:] - Coeff_tot_0[:-1]) * coeffs[IDchange:,0])    
    INT0 *= 9. / (2. * np.pi * np.pi) / a6

    return INT0



def dSdR_TopHat(Pk, k, R, n=0.96, OmaxSmallK=-1):
    """
    Compute derivative of variance with respect to radius.
    
    Calculates dσ²/dR for top-hat filtered field. This derivative is used
    in excursion set calculations to relate crossing statistics to mass.
    
    Parameters
    ----------
    Pk : array_like
        Power spectrum
    k : array_like
        Wavenumber array
    R : array_like
        Filter radius array
    n : float, optional
        Spectral index (default 0.96)
    OmaxSmallK : int, optional
        Small-k expansion order (default -1)
        
    Returns
    -------
    OUT : ndarray
        Derivative dσ²/dR at each radius
        
    Notes
    -----
    The derivative involves the derivative of the window function,
    which introduces additional oscillatory integrals.
    """

    len_R = R.shape[0]

    coeffs = cubic_spline_coeffs(k, Pk)    
    explicit_from_implicit_coeffs(coeffs, k)


    OUT = np.empty(len_R)


    for i in range(len_R):
        IDchange = find_id_change(k,1./R[i])
        OUT[i] = dSdR_TopHat_MAIN_lowR(Pk, k, R[i], IDchange, OmaxSmallK, n)
        OUT[i] += dSdR_TopHat_MAIN(coeffs, k, R[i],IDchange)


    return OUT

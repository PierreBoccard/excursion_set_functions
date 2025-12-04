"""
Tophat window function and derivatives for photo-z extension.

This module implements the tophat window functions in Fourier space (eqs. 85-86).
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def tophat_window(x):
    """
    Tophat window function in Fourier space: W_T(x) = 3(sin(x) - x*cos(x))/x^3
    
    Parameters
    ----------
    x : float or array
        Argument kR
        
    Returns
    -------
    W : float or array
        Tophat window function value
    """
    result = np.zeros_like(x)
    small = np.abs(x) < 1e-2
    large = ~small
    
    # Taylor expansion for small x
    x2 = x[small] ** 2
    x4 = x2 ** 2
    x6 = x4 * x2
    x8 = x6 * x2
    result[small] = 1.0 - x2 / 10.0 + x4 / 280.0 - x6 / 15120.0 + x8 / 1330560.0
    
    # Full expression for large x
    xl = x[large]
    result[large] = 3.0 * (np.sin(xl) - xl * np.cos(xl)) / (xl ** 3)
    
    return result


@jit(nopython=True, cache=True)
def tophat_window_derivative(x):
    """
    First derivative of tophat window function: W'_T(x) = dW_T/dx
    
    From eq. (85):
    W'_T(x) = 3 * [sin(x)/x^2 - 3(sin(x) - x*cos(x))/x^4]
    
    Parameters
    ----------
    x : float or array
        Argument kR
        
    Returns
    -------
    dW : float or array
        First derivative of tophat window function
    """
    result = np.zeros_like(x)
    small = np.abs(x) < 1e-2
    large = ~small
    
    # Taylor expansion for small x: dW/dx = -x/5 + x^3/70 - x^5/2520 + ...
    xs = x[small]
    x2 = xs ** 2
    x4 = x2 ** 2
    x6 = x4 * x2
    result[small] = -xs / 5.0 + xs ** 3 / 70.0 - xs ** 5 / 2520.0 + xs ** 7 / 166320.0
    
    # Full expression for large x
    xl = x[large]
    sin_x = np.sin(xl)
    cos_x = np.cos(xl)
    result[large] = 3.0 * (sin_x / (xl ** 2) - 3.0 * (sin_x - xl * cos_x) / (xl ** 4))
    
    return result


@jit(nopython=True, cache=True)
def tophat_window_second_derivative(x):
    """
    Second derivative of tophat window function: W''_T(x) = d^2W_T/dx^2
    
    Computed as d/dx of dW/dx, where:
    dW/dx = 3 * [sin(x)/x^2 - 3*(sin(x) - x*cos(x))/x^4]
    
    Resulting in:
    d^2W/dx^2 = 3 * [cos(x)/x^2 - 5*sin(x)/x^3 + 12*(sin(x) - x*cos(x))/x^5]
    
    Parameters
    ----------
    x : float or array
        Argument kR
        
    Returns
    -------
    d2W : float or array
        Second derivative of tophat window function
    """
    result = np.zeros_like(x)
    small = np.abs(x) < 1e-2
    large = ~small
    
    # Taylor expansion for small x: d^2W/dx^2 = -1/5 + 3*x^2/70 - x^4/504 + ...
    xs = x[small]
    x2 = xs ** 2
    x4 = x2 ** 2
    result[small] = -1.0 / 5.0 + 3.0 * x2 / 70.0 - x4 / 504.0 + x2 ** 3 / 23760.0
    
    # Full expression for large x
    xl = x[large]
    sin_x = np.sin(xl)
    cos_x = np.cos(xl)
    result[large] = 3.0 * (cos_x / (xl ** 2) - 5.0 * sin_x / (xl ** 3) 
                           + 12.0 * (sin_x - xl * cos_x) / (xl ** 5))
    
    return result

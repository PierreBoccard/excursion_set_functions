"""
Cubic spline interpolation and integration utilities.

Provides fast cubic spline operations optimized with Numba JIT compilation.
Implements not-a-knot boundary conditions for natural interpolation behavior.

Key Features
------------
- Cubic spline coefficient computation (not-a-knot conditions)
- Spline evaluation at arbitrary points
- First derivative computation
- Definite and indefinite integration
- Implicit/explicit coefficient transformations

Main Functions
--------------
cubic_spline_coeffs : Compute spline coefficients from data
get_values : Evaluate spline at points
get_derivative : Compute spline derivative
get_integral : Integrate spline over interval

Performance
-----------
All functions are JIT-compiled with Numba for high performance.
Optimized for sorted evaluation points to minimize interval searches.
"""
import numpy as np
from numba import jit, prange
from numba.core import types
from numba.typed import Dict
#from numba.extending import overload

@jit(nopython=True)
def Solve_tridiagonal_system(a,b,c,d,w,g,p):
    """
    Solve tridiagonal linear system using Thomas algorithm.
    
    Efficient O(n) solver for systems of the form:
    b[0]*p[0] + c[0]*p[1] = d[0]
    a[i-1]*p[i-1] + b[i]*p[i] + c[i]*p[i+1] = d[i]
    a[n-2]*p[n-2] + b[n-1]*p[n-1] = d[n-1]
    
    Parameters
    ----------
    a : array_like
        Lower diagonal (length n-1)
    b : array_like
        Main diagonal (length n)
    c : array_like
        Upper diagonal (length n-1)
    d : array_like
        Right-hand side vector (length n)
    w : array_like
        Work array for forward elimination (length n-1)
    g : array_like
        Work array for forward elimination (length n)
    p : array_like
        Output solution vector (length n), modified in place
        
    Notes
    -----
    Solution is returned in the array p, which is modified in place.
    Work arrays w and g are also modified.
    """
    #a = Lower Diag, b = Main Diag, c = Upper Diag, d = solution vector
    n = len(d)
    #w= np.zeros(n-1)
    #g= np.zeros(n,)
    #p = np.zeros(n)

    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]

    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1,n):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
    p[n-1] = g[n-1]
    for i in range(n-1,0,-1):
        p[i-1] = g[i-1] - w[i-1]*p[i]


@jit(nopython=True)
def Tridiagonal_elements_for_k_not_a_knot(x,y,a, b, c, d):
    """
    Build tridiagonal system for cubic spline with not-a-knot boundary conditions.
    
    Sets up the linear system to solve for cubic spline derivatives at knots.
    The not-a-knot condition enforces continuity of the third derivative at
    the second and second-to-last interior points.
    
    Parameters
    ----------
    x : array_like
        Knot positions (length n)
    y : array_like
        Function values at knots (length n)
    a : array_like
        Output lower diagonal (length n-1), modified in place
    b : array_like
        Output main diagonal (length n), modified in place
    c : array_like
        Output upper diagonal (length n-1), modified in place
    d : array_like
        Output right-hand side (length n), modified in place
        
    Notes
    -----
    The not-a-knot condition is more natural than natural splines for
    functions without known endpoint derivatives.
    """
    #a = Lower Diag, b = Main Diag, c = Upper Diag, d = solution vector
    #n = len(x)
    #a = np.zeros(n-1)
    #b = np.zeros(n)
    #c = np.zeros(n-1)
    #d = np.zeros(n)
    f_first = - 1. / (x[2] - x[1]) ** 2
    b[0] = 1. / (x[1] - x[0]) ** 2
    c[0] = 1. / (x[1] - x[0]) ** 2 - 1. / (x[2] - x[1]) ** 2
    d[0] = 2. * ((y[1] - y[0]) / (x[1] - x[0]) ** 3 - (y[2] - y[1]) / (x[2] - x[1]) ** 3)
    a[-1] = 1. / (x[-1] - x[-2]) ** 2 - 1. / (x[-2] - x[-3]) ** 2
    b[-1] = 1. / (x[-1] - x[-2]) ** 2
    f_last = - 1. / (x[-2] - x[-3]) ** 2
    d[-1] = 2. * ((y[-1] - y[-2]) / (x[-1] - x[-2]) ** 3 - (y[-2] - y[-3]) / (x[-2] - x[-3]) ** 3)
    for i in range(1,len(x)-1):
        a[i-1] = 1. / (x[i] - x[i-1])
        b[i] = 2. / (x[i] - x[i-1]) + 2. / (x[i+1] - x[i])
        c[i] = 1. / (x[i+1] - x[i])
        d[i] = 3. * ((y[i] - y[i-1]) / (x[i] - x[i-1]) ** 2 + (y[i+1] - y[i]) / (x[i+1] - x[i]) ** 2)
    b[0] += -f_first * a[0] / c[1]
    c[0] += -f_first * b[1] / c[1]
    d[0] += -f_first * d[1] / c[1]
    a[-1] += -f_last * b[-2] / a[-2]
    b[-1] += -f_last * c[-1] / a[-2]
    d[-1] += -f_last * d[-2] / a[-2]

@jit(nopython=True)
def Tridiagonal_elements_for_k_not_a_knot_left(x,y,a, b, c, d,second_der_xn):
    """
    Build tridiagonal system with not-a-knot on left and fixed second derivative on right.
    
    Hybrid boundary conditions: not-a-knot at the beginning and specified
    second derivative at the end point.
    
    Parameters
    ----------
    x : array_like
        Knot positions
    y : array_like
        Function values at knots
    a, b, c, d : array_like
        Tridiagonal system arrays, modified in place
    second_der_xn : float
        Specified second derivative at right endpoint
    """
    f_first = - 1. / (x[2] - x[1]) ** 2
    b[0] = 1. / (x[1] - x[0]) ** 2
    c[0] = 1. / (x[1] - x[0]) ** 2 - 1. / (x[2] - x[1]) ** 2
    d[0] = 2. * ((y[1] - y[0]) / (x[1] - x[0]) ** 3 - (y[2] - y[1]) / (x[2] - x[1]) ** 3)
    a[-1] = 1. / (x[-1] - x[-2])
    b[-1] = 2. / (x[-1] - x[-2])
    d[-1] = 3. * (y[-1] - y[-2]) / (x[-1] - x[-2]) ** 2 + second_der_xn / 2.
    for i in range(1,len(x)-1):
        a[i-1] = 1. / (x[i] - x[i-1])
        b[i] = 2. / (x[i] - x[i-1]) + 2. / (x[i+1] - x[i])
        c[i] = 1. / (x[i+1] - x[i])
        d[i] = 3. * ((y[i] - y[i-1]) / (x[i] - x[i-1]) ** 2 + (y[i+1] - y[i]) / (x[i+1] - x[i]) ** 2)
    b[0] += -f_first * a[0] / c[1]
    c[0] += -f_first * b[1] / c[1]
    d[0] += -f_first * d[1] / c[1]

@jit(nopython=True)
def cubic_spline_coeffs(x,y):
    """
    Compute cubic spline coefficients with not-a-knot boundary conditions.
    
    Constructs a C² continuous piecewise cubic polynomial interpolating
    the data points (x, y). Returns coefficients in implicit form
    (relative to interval endpoints).
    
    Parameters
    ----------
    x : array_like
        Knot positions, must be strictly increasing (length n)
    y : array_like
        Function values at knots (length n)
        
    Returns
    -------
    Coeff : ndarray
        Coefficient matrix (n-1, 4) where row i contains [a, b, c, d]
        for the cubic on interval [x[i], x[i+1]]:
        S_i(t) = a + b*t + c*t² + d*t³
        where t = (x - x[i]) / (x[i+1] - x[i])
        
    Notes
    -----
    Coefficients are in "implicit" form, scaled by interval width.
    Use explicit_from_implicit_coeffs to convert to standard polynomial form.
    """
    #a = Lower Diag, b = Main Diag, c = Upper Diag, d = solution vector
    n = len(x)
    Coeff = np.zeros((len(x)-1,4))
    a = np.zeros(n-1)
    b = np.zeros(n)
    c = np.zeros(n-1)
    d = np.zeros(n)
    w = np.zeros(n-1)
    g = np.zeros(n)
    k = np.zeros(n)

    Tridiagonal_elements_for_k_not_a_knot(x, y, a, b, c, d)
    Solve_tridiagonal_system(a, b, c, d, w, g, k)

    A = k[:-1] * (x[1:] - x[:-1]) - (y[1:] - y[:-1])
    B = -k[1:] * (x[1:] - x[:-1]) + (y[1:] - y[:-1])
    Coeff[:,0] = y[:-1]
    Coeff[:,1] = A + y[1:] - y[:-1]
    Coeff[:,2] = B - 2.*A
    Coeff[:,3] = A - B
    return Coeff


@jit(nopython=True)
def explicit_from_implicit_coeffs(Coeff,x):
    """
    Convert spline coefficients from implicit to explicit form.
    
    Transforms coefficients from interval-scaled form to standard polynomial
    coefficients in terms of the original x coordinate:
    S(x) = c₀ + c₁*x + c₂*x² + c₃*x³
    
    Parameters
    ----------
    Coeff : ndarray
        Coefficient matrix (n-1, 4) in implicit form, modified in place
    x : array_like
        Knot positions (length n)
        
    Notes
    -----
    This function modifies Coeff in place. After calling, the coefficients
    represent the polynomial in the original x coordinate system rather than
    the normalized interval parameter t.
    """
    dx  = x[1:] - x[:-1]
    Coeff[:,1] /= dx
    Coeff[:,2] /= dx ** 2
    Coeff[:,3] /= dx ** 3
    
    Coeff[:,0] += -Coeff[:,1] * x[:-1] + Coeff[:,2] * x[:-1] ** 2 - Coeff[:,3] * x[:-1] ** 3
    Coeff[:,1] += -2. * Coeff[:,2] * x[:-1] + 3. * Coeff[:,3] * x[:-1] ** 2
    Coeff[:,2] += -3. * Coeff[:,3] * x[:-1]



@jit(nopython=True)
def get_derivatives_and_values(derivative_out,y_out,x_der,Coeffs,x_input_arr):
    x_max = np.max(x_der)
    I_MAX = 0
    I_R_lastPart = 0
    for i in range(0,len(x_input_arr)):
        I_MAX += (x_max >= x_input_arr[i])
    for x_val in x_der:
        I_R_lastPart += (x_val <= x_input_arr[-2])
    if I_R_lastPart < len(x_der):
        delta_x = x_input_arr[-1] - x_input_arr[-2]
        i_out = -2
        for i_R in range(I_R_lastPart,len(x_der)):
            t = (x_der[i_R] - x_input_arr[i_out]) / delta_x
            y_out[i_R] = Coeffs[i_out,0] + \
                         Coeffs[i_out,1] * t + \
                         Coeffs[i_out,2] * t ** 2 + \
                         Coeffs[i_out,3] * t ** 3
            derivative_out[i_R] = (Coeffs[i_out,1] +
                                   2 * Coeffs[i_out,2] * t +
                                   3 * Coeffs[i_out,3] * t ** 2) / delta_x
        I_MAX = len(x_der)-1

    i_out = -1
    for i_R in range(0,I_R_lastPart):
        for i in range(i_out+1,I_MAX):
            i_out += (x_der[i_R] >= x_input_arr[i])
        delta_x = x_input_arr[i_out+1] - x_input_arr[i_out]
        t = (x_der[i_R] - x_input_arr[i_out]) / delta_x
        y_out[i_R] = Coeffs[i_out,0] + \
                     Coeffs[i_out,1] * t + \
                     Coeffs[i_out,2] * t ** 2 + \
                     Coeffs[i_out,3] * t ** 3
        derivative_out[i_R] = (Coeffs[i_out,1] +
                               2 * Coeffs[i_out,2] * t +
                               3 * Coeffs[i_out,3] * t ** 2) / delta_x
        i_out -= 1


@jit(nopython=True)
def get_values_sorted(x_eval,x,coeffs):
    """
    Evaluate spline at sorted points (efficient version).
    
    Evaluates cubic spline at multiple points. Assumes x_eval is already
    sorted in ascending order for optimal performance.
    
    Parameters
    ----------
    x_eval : array_like
        Sorted evaluation points
    x : array_like
        Spline knot positions
    coeffs : ndarray
        Spline coefficients in explicit form (n-1, 4)
        
    Returns
    -------
    y_out : ndarray
        Spline values at x_eval
        
    Notes
    -----
    For unsorted evaluation points, use get_values instead.
    This function is faster when x_eval is pre-sorted.
    """
    len_out = len(x_eval)
    len_x = len(x)
    y_out = np.empty(len_out)
    delta_x = 0.
    t = 0.
    i_out=0
    len_x_mn2 = len_x - 2
    
    for i in range(0,len_out):
        while ((x_eval[i] >= x[i_out+1]) & (i_out < len_x_mn2)):
            i_out += 1
                
        delta_x = x[i_out+1] - x[i_out]
        t = (x_eval[i] - x[i_out]) / delta_x
        y_out[i] = coeffs[i_out,0] + \
                   coeffs[i_out,1] * t + \
                   coeffs[i_out,2] * t * t + \
                   coeffs[i_out,3] * t * t * t
    return y_out



@jit(nopython=True)
def get_single_value(x_eval,x,coeffs):
    """
    Evaluate spline at a single point.
    
    Parameters
    ----------
    x_eval : float
        Single evaluation point
    x : array_like
        Spline knot positions
    coeffs : ndarray
        Spline coefficients in explicit form
        
    Returns
    -------
    y : float
        Spline value at x_eval
    """
    len_x = len(x)
    delta_x = 0.
    t = 0.
    i_out=0
    len_x_mn2 = len_x - 2
    
    while ((x_eval >= x[i_out+1]) & (i_out < len_x_mn2)):
        i_out += 1
            
    delta_x = x[i_out+1] - x[i_out]
    t = (x_eval - x[i_out]) / delta_x
    return coeffs[i_out,0] + coeffs[i_out,1] * t +  coeffs[i_out,2] * t * t +  coeffs[i_out,3] * t * t * t

@jit(nopython=True)
def get_values(x_eval,x,coeffs):
    """
    Evaluate spline at arbitrary (unsorted) points.
    
    Sorts evaluation points internally for efficiency, then evaluates spline.
    Use this when x_eval may not be sorted.
    
    Parameters
    ----------
    x_eval : array_like
        Evaluation points (any order)
    x : array_like
        Spline knot positions
    coeffs : ndarray
        Spline coefficients in explicit form (n-1, 4)
        
    Returns
    -------
    y_out : ndarray
        Spline values at x_eval (in original order)
    """
    len_out = len(x_eval)
    len_x = len(x)
    y_out = np.empty(len_out)
    delta_x = 0.
    t = 0.
    i_out=0
    len_x_mn2 = len_x - 2
    
    ID = np.argsort(x_eval)
    for i in ID:
        while ((x_eval[i] >= x[i_out+1]) & (i_out < len_x_mn2)):
            i_out += 1
                
        delta_x = x[i_out+1] - x[i_out]
        t = (x_eval[i] - x[i_out]) / delta_x
        y_out[i] = coeffs[i_out,0] + \
                   coeffs[i_out,1] * t + \
                   coeffs[i_out,2] * t * t + \
                   coeffs[i_out,3] * t * t * t
    return y_out


@jit(nopython=True)
def get_derivative_sorted(x_eval,x,coeffs):
    """
    Evaluate spline derivative at sorted points.
    
    Computes first derivative dS/dx at evaluation points.
    Assumes x_eval is sorted for efficiency.
    
    Parameters
    ----------
    x_eval : array_like
        Sorted evaluation points
    x : array_like
        Spline knot positions
    coeffs : ndarray
        Spline coefficients in explicit form
        
    Returns
    -------
    dy_out : ndarray
        Spline derivative values at x_eval
    """
    len_out = len(x_eval)
    len_x = len(x)
    y_out = np.empty(len_out)
    delta_x = 0.
    t = 0.
    i_out=0
    len_x_mn2 = len_x - 2

    for i in range(len_out):
        while ((x_eval[i] >= x[i_out+1]) & (i_out < len_x_mn2)):
            i_out += 1
        delta_x = x[i_out+1] - x[i_out]
        t = (x_eval[i] - x[i_out]) / delta_x
        y_out[i] = (coeffs[i_out,1] + \
                    coeffs[i_out,2] * t * 2. + \
                    coeffs[i_out,3] * t * t * 3.) / delta_x
    
    return y_out


@jit(nopython=True)
def get_derivative(x_eval,x,coeffs):
    """
    Evaluate spline derivative at arbitrary (unsorted) points.
    
    Computes first derivative at evaluation points. Handles unsorted input
    by sorting internally.
    
    Parameters
    ----------
    x_eval : array_like
        Evaluation points (any order)
    x : array_like
        Spline knot positions
    coeffs : ndarray
        Spline coefficients in explicit form
        
    Returns
    -------
    dy_out : ndarray
        Spline derivative values at x_eval (in original order)
    """
    len_out = len(x_eval)
    len_x = len(x)
    y_out = np.empty(len_out)
    delta_x = 0.
    t = 0.
    i_out=0
    len_x_mn2 = len_x - 2

    ID = np.argsort(x_eval)
    for i in ID:
        while ((x_eval[i] >= x[i_out+1]) & (i_out < len_x_mn2)):
            i_out += 1
        delta_x = x[i_out+1] - x[i_out]
        t = (x_eval[i] - x[i_out]) / delta_x
        y_out[i] = (coeffs[i_out,1] + \
                    coeffs[i_out,2] * t * 2. + \
                    coeffs[i_out,3] * t * t * 3.) / delta_x
    
    return y_out
    
@jit(nopython=True)
def get_integral(x1, x2, x, coeffs):
    """
    Compute definite integral of spline from x1 to x2.
    
    Integrates the cubic spline over the interval [x1, x2]:
    ∫_{x1}^{x2} S(x) dx
    
    Parameters
    ----------
    x1 : float
        Lower integration limit
    x2 : float
        Upper integration limit
    x : array_like
        Spline knot positions
    coeffs : ndarray
        Spline coefficients in explicit form
        
    Returns
    -------
    integral : float
        Definite integral value
        
    Notes
    -----
    Handles intervals spanning multiple spline pieces automatically.
    """
    len_x = len(x)
    len_x_mn2 = len_x - 2
    i_out = 0
    len_x_mn2 = len_x - 2

    while ((x1 >= x[i_out+1]) & (i_out < len_x_mn2)):
        i_out += 1
    
    delta_x = x[i_out+1] - x[i_out]
    t = (x1 - x[i_out]) / delta_x
    
    integr_out = -(coeffs[i_out,0] * t +
                   coeffs[i_out,1] * t * t / 2. +
                   coeffs[i_out,2] * t * t * t / 3. +
                   coeffs[i_out,3] * t * t * t * t / 4.) * delta_x

    while ((x2 >= x[i_out+1]) & (i_out < len_x_mn2)) :
        integr_out += (coeffs[i_out,0] + coeffs[i_out,1] / 2. +
                       coeffs[i_out,2] / 3. + coeffs[i_out,3] / 4.) * delta_x
        i_out += 1
        delta_x = x[i_out+1] - x[i_out]
    
    
    t = (x2 - x[i_out]) / delta_x
    integr_out += (coeffs[i_out,0] * t + \
                    coeffs[i_out,1] * t * t / 2. + \
                    coeffs[i_out,2] * t * t * t / 3. + \
                    coeffs[i_out,3] * t * t * t * t / 4.) * delta_x

    return integr_out

    

@jit(nopython=True)
def get_integral_array(x_eval,x,coeffs):
    """
    Compute cumulative integrals over consecutive intervals.
    
    For array [x₀, x₁, x₂, ..., xₙ], computes:
    [∫_{x₀}^{x₁} S(x)dx, ∫_{x₁}^{x₂} S(x)dx, ..., ∫_{xₙ₋₁}^{xₙ} S(x)dx]
    
    Parameters
    ----------
    x_eval : array_like
        Boundaries of integration intervals (length n+1)
    x : array_like
        Spline knot positions
    coeffs : ndarray
        Spline coefficients in explicit form
        
    Returns
    -------
    y_out : ndarray
        Array of integral values (length n)
        
    Notes
    -----
    Efficient for computing many consecutive integrals.
    """
    len_out = len(x_eval)-1
    len_x = len(x)
    y_out = np.empty(len_out)
    delta_x = 0.
    t = 0.
    i_out=0
    len_x_mn2 = len_x - 2

    while ((x_eval[0] >= x[i_out+1]) & (i_out < len_x_mn2)):
        i_out += 1
    
    delta_x = x[i_out+1] - x[i_out]
    t = (x_eval[0] - x[i_out]) / delta_x
    
    for i in range(len_out):
        y_out[i] = -(coeffs[i_out,0] * t +
                     coeffs[i_out,1] * t * t / 2. +
                     coeffs[i_out,2] * t * t * t / 3. +
                     coeffs[i_out,3] * t * t * t * t / 4.) * delta_x
        while ((x_eval[i+1] >= x[i_out+1]) & (i_out < len_x_mn2)):
            y_out[i] += (coeffs[i_out,0] + coeffs[i_out,1] / 2. + 
                         coeffs[i_out,2] / 3. + coeffs[i_out,3] / 4.) * delta_x
            i_out += 1
            delta_x = x[i_out+1] - x[i_out]
        
        
        t = (x_eval[i+1] - x[i_out]) / delta_x
        y_out[i] += (coeffs[i_out,0] * t + 
                     coeffs[i_out,1] * t * t / 2. + 
                     coeffs[i_out,2] * t * t * t / 3. + 
                     coeffs[i_out,3] * t * t * t * t / 4.) * delta_x
    

    return y_out
    


@jit(nopython=True)
def get_values_sigma(x_eval,x,coeffs_dict,len_out):
    len_x = len(x)
    y_out = np.empty(len_out)
    #I_R_lastPart = 0

    #while ((x_eval[I_R_lastPart] <= x[len_x-2]) & (I_R_lastPart < len_out)):
    #    I_R_lastPart += 1

    i_out = 0
    while ((x_eval >= x[i_out+1]) & (i_out < len_x - 2)):
        i_out += 1

    delta_x = x[i_out+1] - x[i_out]
    t = (x_eval - x[i_out]) / delta_x
    t2 = t * t
    t3 = t * t * t
    for i in range(0, len_out):
        y_out[i] = coeffs_dict[i][i_out,0] + \
                   coeffs_dict[i][i_out,1] * t + \
                   coeffs_dict[i][i_out,2] * t2 + \
                   coeffs_dict[i][i_out,3] * t3
    return y_out



@jit(nopython=True)
def get_integrals(integrals_out,estrema_array,Coeffs,x_input_arr):
    x_max = np.max(estrema_array)
    I_MAX = 0
    I_R_lastPart = 0
    for i in range(0,len(x_input_arr)):
        I_MAX += (x_max >= x_input_arr[i])
    for x_val in estrema_array:
        I_R_lastPart += (x_val <= x_input_arr[-2])

    i_R = 0
    i_out = -1
    for i in range(0, I_MAX):
        i_out += (estrema_array[i_R] >= x_input_arr[i])
    delta_x = x_input_arr[i_out + 1] - x_input_arr[i_out]
    t = (estrema_array[i_R] - x_input_arr[i_out]) / delta_x
    i_in = i_out
    i_out -= 1
    for i_R in range(0,I_R_lastPart):
        integrals_out[i_R] = -(Coeffs[i_in, 0] * t +
                               Coeffs[i_in, 1] * t ** 2 / 2 +
                               Coeffs[i_in, 2] * t ** 3 / 3 +
                               Coeffs[i_in, 3] * t ** 4 / 4) * delta_x
        for i in range(i_out+1,I_MAX):
            i_out += (estrema_array[i_R+1] >= x_input_arr[i])
        integrals_out[i_R] += np.sum((Coeffs[i_in:i_out, 0] + Coeffs[i_in:i_out, 1] / 2 +
                                      Coeffs[i_in:i_out, 2] / 3+ Coeffs[i_in:i_out, 3] / 4) *
                                     (x_input_arr[i_in+1:i_out+1] - x_input_arr[i_in:i_out]))
        delta_x = x_input_arr[i_out+1] - x_input_arr[i_out]
        t = (estrema_array[i_R+1] - x_input_arr[i_out]) / delta_x
        integrals_out[i_R] += (Coeffs[i_out, 0] * t +
                               Coeffs[i_out, 1] * t ** 2 / 2 +
                               Coeffs[i_out, 2] * t ** 3 / 3 +
                               Coeffs[i_out, 3] * t ** 4 / 4) * delta_x
        i_in = i_out
        i_out -= 1


@jit(nopython=True)
def get_integral_scalar_array(x1, x_eval, x, coeffs):
    len_out = x_eval.shape[0]
    y_out = np.empty(len_out)

    #double integr_offset,integr_incremental, delta_x, t
    i_out = 0
    len_x_mn2 = x.shape[0] - 2

    integr_offset = 0.
    while ((x1 >= x[i_out+1]) & (i_out < len_x_mn2)):
        delta_x = x[i_out+1] - x[i_out]
        integr_offset += (coeffs[i_out][0] + coeffs[i_out][1] / 2. +
                          coeffs[i_out][2] / 3. + coeffs[i_out][3] / 4.) * delta_x
        i_out += 1
    
    delta_x = x[i_out+1] - x[i_out]
    t = (x1 - x[i_out]) / delta_x
    
    integr_offset += (coeffs[i_out,0] * t + 
                      coeffs[i_out,1] * t * t / 2. + 
                      coeffs[i_out,2] * t * t * t / 3. + 
                      coeffs[i_out,3] * t * t * t * t / 4.) * delta_x

    i_out = 0
    integr_incremental = 0
    while ((x1 >= x[i_out+1]) & (i_out < len_x_mn2)):
        delta_x = x[i_out+1] - x[i_out]
        integr_incremental += (coeffs[i_out][0] + coeffs[i_out][1] / 2. +
                               coeffs[i_out][2] / 3. + coeffs[i_out][3] / 4.) * delta_x
        i_out += 1
    
    sort_ind = np.argsort(x_eval)

    for i in sort_ind:
        while ((x_eval[i] >= x[i_out+1]) & (i_out < len_x_mn2)):
            integr_incremental += (coeffs[i_out][0] + coeffs[i_out][1] / 2. +
                                   coeffs[i_out][2] / 3. + coeffs[i_out][3] / 4.) * delta_x
            i_out += 1
            delta_x = x[i_out+1] - x[i_out]
        
        
        t = (x_eval[i] - x[i_out]) / delta_x
        y_out[i] =  integr_incremental + (coeffs[i_out][0] * t + 
                                          coeffs[i_out][1] * t * t / 2. + 
                                          coeffs[i_out][2] * t * t * t / 3. + 
                                          coeffs[i_out][3] * t * t * t * t / 4.) * delta_x - integr_offset
    return y_out

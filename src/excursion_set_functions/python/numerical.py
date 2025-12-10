"""
Numerical simulation of excursion set first crossing statistics.

This module provides high-performance Monte Carlo methods for computing
first crossing distributions in excursion set theory. Uses Numba JIT
compilation and parallelization for efficient simulation of correlated
random walks through filter scales.

Key Functions
-------------
first_crossing_single_barrier : 
    Main interface for computing first crossing counts
first_crossing_profile_single_barrier :
    Extended version computing full path statistics and histograms

Implementation Notes
--------------------
- Uses Cholesky decomposition for generating correlated Gaussian paths
- Parallel execution across multiple CPU cores
- Optimized memory layout for cache efficiency
- Supports both constant and scale-dependent barriers

References
----------
Excursion set theory: Bond et al. (1991), ApJS 103, 1
"""
import numpy as np
from numba import jit, prange, get_num_threads
try:
    from numba import get_thread_id
except:
    from numba.np.ufunc.parallel import _get_thread_id as get_thread_id
from numba.core import types
from numba.typed import Dict

__all__ = ["first_crossing_single_barrier", "first_crossing_profile_single_barrier", "first_crossing_profile_single_barrier"]

float_array = types.float64[::1]
float_array2D = types.float64[:,::1]
int_array = types.int64[::1]
int_array2D = types.int64[:,::1]





@jit(nopython=True)
def first_crossing_perCore_scalar_barrier_single_numba(F_ij_reshaped, N_paths, N_Rfilt, delta_c):
    """
    Compute first crossing statistics for a single CPU core with scalar barrier.
    
    Simulates random walks through filter scales and counts the first crossing
    of a constant barrier delta_c. Uses Cholesky decomposition matrix F_ij to
    generate correlated random walks.
    
    Parameters
    ----------
    F_ij_reshaped : array_like
        Linearized Cholesky decomposition matrix (lower triangular)
    N_paths : int
        Number of random walk paths to simulate
    N_Rfilt : int
        Number of filter scales
    delta_c : float
        Constant barrier height
        
    Returns
    -------
    NumCrossing : ndarray
        Count of first crossings at each filter scale
    """
    NumCrossing = np.zeros(N_Rfilt, dtype=np.int_)
    RAND = np.empty(N_Rfilt)
    progr_ind_out = 0
    FiltPath = 0.
    bool_cond = True
    for nn in range(0,N_paths):
        progr_ind_out = 0
        FiltPath = 0.
        i=0
        bool_cond = True
        while bool_cond:
            RAND[i] = np.random.normal()
            FiltPath = 0.
            for s in range(0,i+1):
                FiltPath += F_ij_reshaped[progr_ind_out + s] * RAND[s]
            bool_cond = (FiltPath < delta_c)
            progr_ind_out += i + 1
            i += 1
            bool_cond &= (i < N_Rfilt)
        NumCrossing[i-1] += (FiltPath >= delta_c)
    return NumCrossing

@jit(nopython=True,parallel=True)
def first_crossing_scalar_barrier_single_numba(F_ij_reshaped, N_paths, delta_c, nCPU):
    """
    Parallel computation of first crossing with scalar barrier.
    
    Distributes the computation across multiple CPU cores and aggregates results.
    
    Parameters
    ----------
    F_ij_reshaped : array_like
        Linearized Cholesky decomposition matrix
    N_paths : int
        Total number of paths to simulate
    delta_c : float
        Constant barrier height
    nCPU : int
        Number of CPU cores to use
        
    Returns
    -------
    NumCrossing : ndarray
        Total count of first crossings at each filter scale
    """
    N_Rfilt = int(round((np.sqrt(8 * len(F_ij_reshaped) + 1) - 1) / 2))
    Cross_perCore = Dict.empty(types.int64, int_array)
    for nn in range(0, nCPU):
        Cross_perCore[nn] = np.zeros(N_Rfilt, dtype=np.int_)
    for nn in prange(0,nCPU):
        N_paths_core = int(N_paths / nCPU)
        N_paths_core += nn < (N_paths % nCPU)
        Cross_perCore[nn][:] = first_crossing_perCore_scalar_barrier_single_numba(F_ij_reshaped, N_paths_core, N_Rfilt, delta_c)

    NumCrossing = np.zeros(N_Rfilt, dtype=np.int_)
    for nn in range(0,nCPU):
        NumCrossing += Cross_perCore[nn]
    return NumCrossing





@jit(nopython=True)
def first_crossing_perCore_array_barrier_single_numba(F_ij_reshaped, N_paths, N_Rfilt, delta_c):
    """
    Compute first crossing statistics for a single CPU core with scale-dependent barrier.
    
    Similar to scalar barrier version but allows barrier height to vary with filter scale.
    
    Parameters
    ----------
    F_ij_reshaped : array_like
        Linearized Cholesky decomposition matrix
    N_paths : int
        Number of random walk paths to simulate
    N_Rfilt : int
        Number of filter scales
    delta_c : array_like
        Barrier height at each filter scale
        
    Returns
    -------
    NumCrossing : ndarray
        Count of first crossings at each filter scale
    """
    NumCrossing = np.zeros(N_Rfilt, dtype=np.int_)
    RAND = np.empty(N_Rfilt)
    progr_ind_out = 0
    FiltPath = 0.
    bool_cond = True
    for nn in range(0,N_paths):
        progr_ind_out = 0
        FiltPath = 0.
        i=0
        bool_cond = True
        while bool_cond:
            RAND[i] = np.random.normal()
            FiltPath = 0.
            for s in range(0,i+1):
                FiltPath += F_ij_reshaped[progr_ind_out + s] * RAND[s]
            bool_cond = (FiltPath < delta_c[i])
            progr_ind_out += i + 1
            i += 1
            bool_cond &= (i < N_Rfilt)
        NumCrossing[i-1] += (FiltPath >= delta_c[i-1])
    return NumCrossing

@jit(nopython=True,parallel=True)
def first_crossing_array_barrier_single_numba(F_ij_reshaped, N_paths, delta_c, nCPU):
    """
    Parallel computation of first crossing with scale-dependent barrier.
    
    Distributes the computation across multiple CPU cores for array barrier case.
    
    Parameters
    ----------
    F_ij_reshaped : array_like
        Linearized Cholesky decomposition matrix
    N_paths : int
        Total number of paths to simulate
    delta_c : array_like
        Barrier height at each filter scale
    nCPU : int
        Number of CPU cores to use
        
    Returns
    -------
    NumCrossing : ndarray
        Total count of first crossings at each filter scale
    """
    N_Rfilt = int(round((np.sqrt(8 * len(F_ij_reshaped) + 1) - 1) / 2))
    Cross_perCore = Dict.empty(types.int64, int_array)
    for nn in range(0, nCPU):
        Cross_perCore[nn] = np.zeros(N_Rfilt, dtype=np.int_)
    for nn in prange(0,nCPU):
        N_paths_core = int(N_paths / nCPU)
        N_paths_core += nn < (N_paths % nCPU)
        Cross_perCore[nn][:] = first_crossing_perCore_array_barrier_single_numba(F_ij_reshaped, N_paths_core, N_Rfilt, delta_c)

    NumCrossing = np.zeros(N_Rfilt, dtype=np.int_)
    for nn in range(0,nCPU):
        NumCrossing += Cross_perCore[nn]
    return NumCrossing



def first_crossing_single_barrier(F_ij_reshaped, N_paths, delta_c, nCPU=-1):
    """
    Main interface for first crossing distribution computation.
    
    Automatically detects whether barrier is scalar or array and dispatches
    to appropriate parallel implementation. Used in excursion set theory to
    compute halo mass functions.
    
    Parameters
    ----------
    F_ij_reshaped : array_like
        Linearized Cholesky decomposition of covariance matrix
    N_paths : int
        Number of Monte Carlo paths to simulate
    delta_c : float or array_like
        Critical density threshold. Can be scalar (constant) or array (scale-dependent)
    nCPU : int, optional
        Number of CPU cores to use. Default -1 uses all available cores
        
    Returns
    -------
    NumCrossing : ndarray
        Number of first crossings at each filter scale
        
    Notes
    -----
    The Cholesky decomposition F_ij is stored in linearized form to optimize
    memory access in the Monte Carlo simulation.
    """
    if (nCPU < 0) | (nCPU > get_num_threads()):
        nCPU = get_num_threads()
    if np.isscalar(delta_c):
        return first_crossing_scalar_barrier_single_numba(F_ij_reshaped, N_paths, delta_c,nCPU)
    return first_crossing_array_barrier_single_numba(F_ij_reshaped, N_paths, delta_c,nCPU)


@jit(nopython=True)
def histo_profile(histo,FiltPath,N_Rfilt,binsize,offset,nbins):
    """
    Build histogram of filtered path values at each scale.
    
    Updates histogram counts for the filtered random walk trajectory.
    Used to compute probability distributions of density field values.
    
    Parameters
    ----------
    histo : ndarray
        2D histogram array (nbins x N_Rfilt) to update
    FiltPath : array_like
        Filtered path values at each scale
    N_Rfilt : int
        Number of filter scales
    binsize : float
        Width of histogram bins
    offset : float
        Minimum value (left edge of first bin)
    nbins : int
        Number of histogram bins
    """
    for i in range(N_Rfilt):
        ind = int((FiltPath[i] - offset) / binsize)
        if (ind < nbins) & (ind >= 0):
            histo[ind,i] += 1



@jit(nopython=True)
def first_crossing_profile_perCore_array_barrier_single(
    F_ij_reshaped, N_paths, N_Rfilt, delta_c,
    hist_dict,binsize,offset,nbins):
    """
    Compute first crossing profiles for a single CPU core.
    
    Extended version that not only counts crossings but also computes:
    - Mean filtered path after crossing
    - Full histogram of path values at each scale
    
    Parameters
    ----------
    F_ij_reshaped : array_like
        Linearized Cholesky decomposition matrix
    N_paths : int
        Number of paths to simulate
    N_Rfilt : int
        Number of filter scales
    delta_c : array_like
        Barrier height at each scale
    hist_dict : dict
        Dictionary of histogram arrays for each crossing scale
    binsize : float
        Width of histogram bins
    offset : float
        Minimum histogram value
    nbins : int
        Number of histogram bins
        
    Returns
    -------
    NumCrossing : ndarray
        Count of first crossings at each scale
    FiltPath_mean : ndarray
        Mean path values conditioned on crossing at each scale
    """
    
    FiltPath = np.zeros(N_Rfilt)
    FiltPath_mean = np.zeros((N_Rfilt,N_Rfilt))
    NumCrossing = np.zeros(N_Rfilt, dtype=np.int_)
    RAND = np.empty(N_Rfilt)
    #IDcross = np.zeros(N_paths, dtype=np.int_)
    #RANDmatr = np.zeros((N_Rfilt,N_paths), dtype=np.float_)
    Nx=0
    uncrossed = True
    progr = 0
    for nn in range(0,N_paths):
        progr_ind_out = 0
        #FiltPath = 0.
        i=0
        uncrossed = True
        while uncrossed & (i < N_Rfilt):
            RAND[i] = np.random.normal()
            FiltPath[i] = 0.
            for s in range(0,i+1):
                FiltPath[i] += F_ij_reshaped[progr_ind_out + s] * RAND[s]
            uncrossed = (FiltPath[i] < delta_c[i])
            progr_ind_out += i + 1
            i += 1
            #uncrossed &= (i < N_Rfilt)
        if not uncrossed:
            j=i
            NumCrossing[i-1] += 1
            while (j < N_Rfilt):
                RAND[j] = np.random.normal()
                FiltPath[j] = 0.
                for s in range(0,j+1):
                    FiltPath[j] += F_ij_reshaped[progr_ind_out + s] * RAND[s]
                progr_ind_out += j + 1
                j += 1
            FiltPath_mean[:,i-1] += FiltPath
            histo_profile(hist_dict[i-1],FiltPath,N_Rfilt,binsize,offset,nbins)
            
    return NumCrossing, FiltPath_mean



hist_dict_type = types.DictType(types.int64, int_array2D)
@jit(nopython=True, parallel=True)
def first_crossing_profile_array_barrier_single(
    F_ij_reshaped, N_paths, delta_c,delta_min,delta_max,nbins,nCPU):
    """
    Parallel computation of complete first crossing profiles.
    
    Computes crossing statistics, mean paths, and histograms across multiple cores.
    This provides comprehensive information about the excursion set trajectories.
    
    Parameters
    ----------
    F_ij_reshaped : array_like
        Linearized Cholesky decomposition matrix
    N_paths : int
        Total number of paths to simulate
    delta_c : array_like
        Barrier height at each scale
    delta_min : float
        Minimum value for histogram range
    delta_max : float
        Maximum value for histogram range
    nbins : int
        Number of histogram bins
    nCPU : int
        Number of CPU cores to use
        
    Returns
    -------
    NumCrossing : ndarray
        Total first crossings at each scale
    FiltPath_mean : ndarray
        Mean conditioned paths
    histo_bins : ndarray
        Histogram bin edges
    hist_dict : dict
        Complete histograms for each crossing scale
    """
    offset = delta_min
    binsize = (delta_max - delta_min) / nbins
    histo_bins = np.linspace(delta_min,delta_max,nbins+1)
    
    #nCPU = get_num_threads()
    N_Rfilt = int(round((np.sqrt(8 * len(F_ij_reshaped) + 1) - 1) / 2))

    NumCrossing_perCore = Dict.empty(types.int64, int_array)
    FiltPath_mean_perCore = Dict.empty(types.int64, float_array2D)
    hist_dict_perCore = Dict.empty(types.int64, hist_dict_type)
    for nn in range(0, nCPU):
        NumCrossing_perCore[nn] = np.zeros(N_Rfilt, dtype=np.int_)
        FiltPath_mean_perCore[nn] = np.zeros((N_Rfilt,N_Rfilt), dtype=np.float_)
        hist_dict_perCore[nn] = Dict.empty(types.int64, int_array2D)
        for i in range(N_Rfilt):
            hist_dict_perCore[nn][i] = np.zeros((nbins,N_Rfilt), dtype=np.int_)
    for nn in prange(0, nCPU):
        N_paths_core = int(N_paths / nCPU)
        N_paths_core += nn < (N_paths % nCPU)
        NumCrossing_perCore[nn][:], FiltPath_mean_perCore[nn][:,:] = \
            first_crossing_profile_perCore_array_barrier_single(
                F_ij_reshaped, N_paths_core, N_Rfilt, delta_c,
                hist_dict_perCore[nn],binsize,offset,nbins)
        
    NumCrossing = np.zeros(N_Rfilt, dtype=np.int_)
    FiltPath_mean = np.zeros((N_Rfilt,N_Rfilt), dtype=np.float_)
    hist_dict = Dict.empty(types.int64, int_array2D)
    for i in range(N_Rfilt):
        hist_dict[i] = np.zeros((nbins,N_Rfilt), dtype=np.int_)
    for nn in range(0, nCPU):
        NumCrossing += NumCrossing_perCore[nn]
        FiltPath_mean += FiltPath_mean_perCore[nn]
        for i in range(N_Rfilt):
            hist_dict[i] += hist_dict_perCore[nn][i]
    for i in range(N_Rfilt):
        if NumCrossing[i] > 0:
            FiltPath_mean[:,i] /= NumCrossing[i]
    return NumCrossing, FiltPath_mean, histo_bins, hist_dict



def first_crossing_profile_single_barrier(F_ij_reshaped, N_paths, delta_c,delta_min,delta_max,nbins,nCPU=-1):
    """
    Main interface for computing complete first crossing profiles.
    
    High-level function that computes not only when paths cross the barrier,
    but also the full statistical distribution of path values at all scales.
    
    Parameters
    ----------
    F_ij_reshaped : array_like
        Linearized Cholesky decomposition of covariance matrix
    N_paths : int
        Number of Monte Carlo realizations
    delta_c : float or array_like
        Critical barrier height(s)
    delta_min : float
        Minimum value for histogram
    delta_max : float
        Maximum value for histogram
    nbins : int
        Number of histogram bins
    nCPU : int, optional
        Number of CPU cores (-1 = all available)
        
    Returns
    -------
    NumCrossing : ndarray
        First crossing counts
    FiltPath_mean : ndarray
        Conditional mean paths
    histo_bins : ndarray
        Histogram bin edges
    hist_dict : dict
        Full histograms at each scale
        
    Notes
    -----
    This function is more expensive than first_crossing_single_barrier but
    provides complete information about the distribution of density values.
    """
    if (nCPU < 0) | (nCPU > get_num_threads()):
        nCPU = get_num_threads()
    if np.isscalar(delta_c):
        return first_crossing_scalar_barrier_single_numba(
            F_ij_reshaped, N_paths, np.fill(int(round((np.sqrt(8 * len(F_ij_reshaped) + 1) - 1) / 2)),delta_c),delta_min,delta_max,nbins,nCPU)
    return first_crossing_profile_array_barrier_single(F_ij_reshaped, N_paths, delta_c,delta_min,delta_max,nbins,nCPU)
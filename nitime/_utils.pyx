""" -*- python -*- file
C-level implementation of the following routines in utils.py:

  * tridisolve()

"""

# cython: profile=True

import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
def tridisolve(cnp.ndarray[cnp.npy_double, ndim=1] d,
               cnp.ndarray[cnp.npy_double, ndim=1] e,
               cnp.ndarray[cnp.npy_double, ndim=1] b, overwrite_b=True):
    """
    Symmetric tridiagonal system solver, from Golub and Van Loan pg 157

    Parameters
    ----------

    d : ndarray
      main diagonal stored in d[:]
    e : ndarray
      superdiagonal stored in e[:-1]
    b : ndarray
      RHS vector

    Returns
    -------

    x : ndarray
      Solution to Ax = b (if overwrite_b is False). Otherwise solution is
      stored in previous RHS vector b

    """
    # indexing
    cdef int N = len(b)
    cdef int k

    # work vectors
    cdef cnp.ndarray[cnp.npy_double, ndim=1] dw
    cdef cnp.ndarray[cnp.npy_double, ndim=1] ew
    cdef cnp.ndarray[cnp.npy_double, ndim=1] x
    dw = d.copy()
    ew = e.copy()
    if overwrite_b:
        x = b
    else:
        x = b.copy()
    for k in xrange(1, N):
        # e^(k-1) = e(k-1) / d(k-1)
        # d(k) = d(k) - e^(k-1)e(k-1) / d(k-1)
        t = ew[k - 1]
        ew[k - 1] = t / dw[k - 1]
        dw[k] = dw[k] - t * ew[k - 1]
    for k in xrange(1, N):
        x[k] = x[k] - ew[k - 1] * x[k - 1]
    x[N - 1] = x[N - 1] / dw[N - 1]
    for k in xrange(N - 2, -1, -1):
        x[k] = x[k] / dw[k] - ew[k] * x[k + 1]

    if not overwrite_b:
        return x

@cython.boundscheck(False)
def adaptive_weights(cnp.ndarray[cnp.npy_complex128, ndim=2] yk, 
                     cnp.ndarray[cnp.npy_double, ndim=1] eigvals, 
                     sides='onesided', max_iter=150):
    r"""
    Perform an iterative procedure to find the optimal weights for K
    direct spectral estimators of DPSS tapered signals.

    Parameters
    ----------

    yk : ndarray (K, N)
       The K DFTs of the tapered sequences
    eigvals : ndarray, length-K
       The eigenvalues of the DPSS tapers
    sides : str
       Whether to compute weights on a one-sided or two-sided spectrum
    max_iter : int
       Maximum number of iterations for weight computation

    Returns
    -------

    weights, nu

       The weights (array like sdfs), and the
       "equivalent degrees of freedom" (array length-L)

    Notes
    -----

    The weights to use for making the multitaper estimate, such that
    :math:`S_{mt} = \sum_{k} |w_k|^2S_k^{mt} / \sum_{k} |w_k|^2`

    If there are less than 3 tapers, then the adaptive weights are not
    found. The square root of the eigenvalues are returned as weights,
    and the degrees of freedom are 2*K

    """
    from nitime.algorithms import mtm_cross_spectrum
    K = len(eigvals)
    if K < 3:
        ## print("""
        ## Warning--not adaptively combining the spectral estimators
        ## due to a low number of tapers.
        ## """)
        # we'll hope this is a correct length for L
        N = yk.shape[1]
        L = N / 2 + 1 if sides == 'onesided' else N
        def_weights = np.tile( np.sqrt(eigvals)[:, None], (1, L))
        return (def_weights, np.ones( (L,) ) *2 * K)

    cdef cnp.ndarray[cnp.float64_t, ndim=1] rt_eig = np.empty( (K,) )
    rt_eig[:] = np.sqrt(eigvals)

    # combine the SDFs in the traditional way in order to estimate
    # the variance of the timeseries
    N = yk.shape[1]
    sdf = mtm_cross_spectrum(yk, yk, eigvals[:, None], sides=sides)
    L = sdf.shape[-1]
    var_est = np.sum(sdf, axis=-1) / N
    
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bband_sup = np.empty( (K,) )
    bband_sup[:] = (1-eigvals)*var_est

    # The process is to iteratively switch solving for the following
    # two expressions:
    # (1) Adaptive Multitaper SDF:
    # S^{mt}(f) = [ sum |d_k(f)|^2 S_k(f) ]/ sum |d_k(f)|^2
    #
    # (2) Weights
    # d_k(f) = [sqrt(lam_k) S^{mt}(f)] / [lam_k S^{mt}(f) + E{B_k(f)}]
    #
    # Where lam_k are the eigenvalues corresponding to the DPSS tapers,
    # and the expected value of the broadband bias function
    # E{B_k(f)} is replaced by its full-band integration
    # (1/2pi) int_{-pi}^{pi} E{B_k(f)} = sig^2(1-lam_k)

    # start with an estimate from incomplete data--the first 2 tapers
    cdef cnp.ndarray[cnp.float64_t, ndim=1] sdf_two = np.empty( (L,) )
    sdf_two[:] = mtm_cross_spectrum(yk[:2], yk[:2], eigvals[:2, None],
                                    sides=sides)
    
    # for numerical considerations, don't bother doing adaptive
    # weighting after 150 dB down
    min_pwr = sdf_two.max() * 10 ** (-150/10.)

    w_def = rt_eig[:,None] * sdf_two
    w_def /= eigvals[:, None] * sdf_two + bband_sup[:,None]

    cdef cnp.ndarray[cnp.float64_t, ndim=2] d_sdfs = np.empty( (L, K) )
    d_sdfs = np.abs(yk[:,:L].T)**2
    if L < N:
        d_sdfs *= 2

    cdef cnp.ndarray[cnp.float64_t, ndim=2] d_k = np.zeros( (L, K) )
    cdef int n
    cdef int f
    cdef int k
    cdef double sdf_num
    cdef double sdf_den
    cdef double cfn_num
    cdef double cfn_den
    total_iters = 0
    
    for f in xrange(L):
        n = 0

        sdf_iter = sdf_two[f]

        cfn_num = 0
        cfn_den = 0
        for k in xrange(K):
            cfn_num += eigvals[k] * (sdf_iter - d_sdfs[f, k])
            cfn_den += eigvals[k] * sdf_iter - bband_sup[k]
        cfn_init = cfn_num / cfn_den

        while True:
            for k in xrange(K):
                d_k[f,k] = rt_eig[k] * sdf_iter
                d_k[f,k] /= eigvals[k] * sdf_iter + bband_sup[k]


            sdf_num = 0
            sdf_den = 0
            for k in xrange(K):
                sdf_num += d_k[f, k]**2 * d_sdfs[f, k]
                sdf_den += d_k[f, k]**2
                
            sdf_iter = sdf_num / sdf_den

            # Compute the cost function from eq 5.4 in Thomson 1982
            cfn_num = 0
            cfn_den = 0
            for k in xrange(K):
                cfn_num += eigvals[k] * (sdf_iter - d_sdfs[f, k])
                cfn_den += eigvals[k] * sdf_iter - bband_sup[k]
            cfn = cfn_num / cfn_den

            if (cfn == 0) or (np.abs(cfn - cfn_init) < 1e-2 * np.abs(cfn_init)):
                #print 'breaking', n, 'iters'
                total_iters += n
                break
            n += 1
            if n == max_iter:
                #print 'breaking maxiter', np.abs((cfn - cfn_init) / cfn_init)
                total_iters += n
                for k in xrange(K):
                    d_k[f, k] = w_def[k, f]
                break
            cfn_init = cfn

    # d_k is now L x K
    nu = 2 * (d_k ** 2).sum(axis=-1)
    return d_k.T, nu.T

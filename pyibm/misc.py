"""Implementation of miscellaneous functions."""

import numba
from scipy.sparse.linalg import eigsh


def condition_number(A):
    """Estimate the smallest and largest eigenvalues, and condition number.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The sparse operator.

    Returns
    -------
    float
        Estimation of the smallest eigenvalue.
    float
        Estimation of the largest eigenvalue.
    float
        Condition number (ratio max/min).

    """
    evals_large, _ = eigsh(A, 1, which='LM')
    evals_small, _ = eigsh(A, 1, sigma=0, which='LM')
    lambda_min, lambda_max = evals_small[0], evals_large[0]
    cond = lambda_max / lambda_min
    return lambda_min, lambda_max, cond


@numba.njit
def _idx(i, j, k, shape):
    ny, nx = shape[-2:]
    return k * (ny * nx) + j * nx + i


@numba.njit
def _ijk(I, shape):
    ny, nx = shape[-2:]
    if len(shape) == 2:
        return I % nx, I // nx, 0
    lda = ny * nx
    return I % nx, (I % lda) // nx, I // lda

"""Implementation of miscellaneous functions."""

import numba


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

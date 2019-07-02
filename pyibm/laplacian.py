"""Implementation of the Laplacian operator."""

import numba
import numpy
from scipy.sparse import coo_matrix

from .grid import GridBase
from .misc import _idx, _ijk


def assemble_LHat(gridx, gridy, gridz=GridBase()):
    """Assemble the Laplacian operator LHat."""
    ndim = gridx.ndim
    size = gridx.size + gridy.size + gridz.size
    nnz = 2 * ndim + 1
    rows = numpy.zeros(size * nnz, dtype=numpy.int32)
    cols = numpy.zeros(size * nnz, dtype=numpy.int32)
    data = numpy.zeros(size * nnz, dtype=numpy.float64)
    offset, counter = 0, 0
    for grid in ([gridx, gridy, gridz][:ndim]):
        dx, dy = grid.x.get_widths(), grid.y.get_widths()
        dz = numpy.array([1.0]) if ndim == 2 else grid.z.get_widths()
        counter = _kernel(dx, dy, dz, grid.shape, offset, counter,
                          rows, cols, data)
        offset += grid.size
    LHat = coo_matrix((data, (rows, cols)), shape=(size, size))
    return LHat.tocsr()


@numba.njit
def _kernel(dx, dy, dz, shape, offset, counter, rows, cols, data):
    ndim = len(shape)
    N, M = shape[-2:]
    size = 1
    for s in shape:
        size *= s
    for I in range(size):
        i, j, k = _ijk(I, shape)
        Iw, Ie = _idx(i - 1, j, k, shape), _idx(i + 1, j, k, shape)
        Is, In = _idx(i, j - 1, k, shape), _idx(i, j + 1, k, shape)
        dx_w = dx[i] if i == 0 else dx[i - 1]
        dx_e = dx[i] if i == M - 1 else dx[i + 1]
        dy_s = dy[j] if j == 0 else dy[j - 1]
        dy_n = dy[j] if j == N - 1 else dy[j + 1]
        Cw = 2 / dx[i] / (dx[i] + dx_w)
        Ce = 2 / dx[i] / (dx[i] + dx_e)
        Cs = 2 / dy[j] / (dy[j] + dy_s)
        Cn = 2 / dy[j] / (dy[j] + dy_n)
        if i > 0:
            rows[counter] = I + offset
            cols[counter] = Iw + offset
            data[counter] = Cw
            counter += 1
        if i < M - 1:
            rows[counter] = I + offset
            cols[counter] = Ie + offset
            data[counter] = Ce
            counter += 1
        if j > 0:
            rows[counter] = I + offset
            cols[counter] = Is + offset
            data[counter] = Cs
            counter += 1
        if j < N - 1:
            rows[counter] = I + offset
            cols[counter] = In + offset
            data[counter] = Cn
            counter += 1
        C = - (Cw + Ce + Cs + Cn)
        if ndim == 3:
            P = shape[0]
            Ib, If = _idx(i, j, k - 1, shape), _idx(i, j, k + 1, shape)
            dz_b = dz[k] if k == 0 else dz[k - 1]
            dz_f = dz[k] if k == P - 1 else dz[k + 1]
            Cb = 2 / dz[k] / (dz[k] + dz_b)
            Cf = 2 / dz[k] / (dz[k] + dz_f)
            if k > 0:
                rows[counter] = I + offset
                cols[counter] = Ib + offset
                data[counter] = Cb
                counter += 1
            if k < P - 1:
                rows[counter] = I + offset
                cols[counter] = If + offset
                data[counter] = Cf
                counter += 1
            C += - (Cb + Cf)
        rows[counter] = I + offset
        cols[counter] = I + offset
        data[counter] = C
        counter += 1
    return counter

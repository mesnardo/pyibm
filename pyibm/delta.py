"""Implementation of the n-dimensional delta operator."""

import numba
import numpy
from scipy.sparse import coo_matrix

from .deltakernels import *
from .grid import GridBase
from .misc import _idx, _ijk


def assemble_delta(body, gridc, gridx, gridy, gridz=GridBase(),
                   kernel=delta_roma_et_al_1999, kernel_size=2):
    """Assemble the delta sparse matrix in CSR format."""
    ndim = body.ndim
    size = gridx.size + gridy.size + gridz.size
    num_rows = body.ndim * body.size
    nnz = (2 * kernel_size + 1)**ndim
    rows = numpy.zeros(num_rows * nnz, dtype=numpy.int32)
    cols = numpy.zeros(num_rows * nnz, dtype=numpy.int32)
    data = numpy.zeros(num_rows * nnz, dtype=numpy.float64)
    X, Y, Z, neighbors = body.x, body.y, body.z, body.neighbors
    # Get cell widths in the uniform region.
    i, j, k = gridc.ijk(neighbors[0])
    dx, dy = gridc.x.widths[i], gridc.y.widths[j]
    # Assemble rows.
    if ndim == 2:
        offset, counter = 0, 0
        for dof, grid in enumerate([gridx, gridy]):
            x, y = grid.x.vertices, grid.y.vertices
            counter = _kernel2d(X, Y, x, y, dx, dy,
                                neighbors, dof,
                                gridc.shape, grid.shape, offset, counter,
                                kernel, kernel_size,
                                rows, cols, data)
            offset += grid.size
    elif ndim == 3:
        dz = gridc.z.widths[k]
        offset, counter = 0, 0
        for dof, grid in enumerate([gridx, gridy, gridz]):
            x, y, z = grid.x.vertices, grid.y.vertices, grid.z.vertices
            counter = _kernel3d(X, Y, Z, x, y, z, dx, dy, dz,
                                neighbors, dof,
                                gridc.shape, grid.shape, offset, counter,
                                kernel, kernel_size,
                                rows, cols, data)
            offset += grid.size
    Op = coo_matrix((data, (rows, cols)), shape=(body.ndim * body.size, size))
    Op.eliminate_zeros()
    return Op.tocsr()


@numba.njit
def _kernel2d(X, Y, x, y, dx, dy, neighbors, dof,
              shape, shape_col, offset, counter,
              kernel, ks, rows, cols, data):
    ndim = 2
    num_markers = X.size
    for l in range(num_markers):
        i, j, _ = _ijk(neighbors[l], shape)
        for jj in range(j - ks, j + ks + 1):
            ry = abs(Y[l] - y[jj])
            for ii in range(i - ks, i + ks + 1):
                rx = abs(X[l] - x[ii])
                rows[counter] = ndim * l + dof
                cols[counter] = _idx(ii, jj, 0, shape_col) + offset
                data[counter] = (delta(rx, dx, kernel=kernel) *
                                 delta(ry, dy, kernel=kernel))
                counter += 1
    return counter


@numba.njit
def _kernel3d(X, Y, Z, x, y, z, dx, dy, dz, neighbors, dof,
              shape, shape_col, offset, counter,
              kernel, ks, rows, cols, data):
    ndim = 3
    num_markers = X.size
    for l in range(num_markers):
        i, j, k = _ijk(neighbors[l], shape)
        for kk in range(k - ks, k + ks + 1):
            rz = abs(Z[l] - z[kk])
            for jj in range(j - ks, j + ks + 1):
                ry = abs(Y[l] - y[jj])
                for ii in range(i - ks, i + ks + 1):
                    rx = abs(X[l] - x[ii])
                    rows[counter] = ndim * l + dof
                    cols[counter] = _idx(ii, jj, kk, shape_col) + offset
                    data[counter] = (delta(rx, dx, kernel=kernel) *
                                     delta(ry, dy, kernel=kernel) *
                                     delta(rz, dz, kernel=kernel))
                    counter += 1
    return counter

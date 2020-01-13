"""Implementation of the gradient operator."""

import numba
import numpy
from scipy.sparse import coo_matrix

from .grid import GridBase
from .misc import _idx, _ijk


def assemble_GHat(gridc, gridx, gridy, gridz=GridBase()):
    """Assemble the gradient operator GHat for a staggered grid.

    Parameters
    ----------
    gridc : pyibm.GridCellCentered
        Cell-centered grid.
    gridx : pyibm.GridFaceX
        Grid at X faces.
    gridy : pyibm.GridFaceY
        Grid at Y faces.
    gridz : pyibm.GridFaceZ (optional)
        Grid at Z faces (for 3D cases); default: GridBase().

    Returns
    -------
    GHat: scipy.sparse.csr_matrix
        The gradient operator.

    """
    ndim = gridc.ndim  # number of dimension
    size = gridx.size + gridy.size + gridz.size  # number of rows
    nnz = 2  # estimated number of nonzeros per row
    rows = numpy.zeros(nnz * size, dtype=numpy.int32)
    cols = numpy.zeros(nnz * size, dtype=numpy.int32)
    data = numpy.zeros(nnz * size, dtype=numpy.float64)
    # Assemble rows for GridFaceX points.
    offset = 0  # row index offset
    dx = gridx.x.widths  # grid spacings in x-direction
    _kernel(gridx.shape, gridc.shape, offset, 0, dx, _stencil_x,
            rows, cols, data)
    # Assemble rows for GridFaceY points.
    offset += gridx.size  # update offset index
    dy = gridy.y.widths
    _kernel(gridy.shape, gridc.shape, offset, 1, dy, _stencil_y,
            rows, cols, data)
    if ndim == 3:
        # Assemble rows for GridFaceZ points.
        offset += gridy.size  # update offset index
        dz = gridz.z.widths
        _kernel(gridz.shape, gridc.shape, offset, 2, dz, _stencil_z,
                rows, cols, data)
    GHat = coo_matrix((data, (rows, cols)), shape=(size, gridc.size))
    return GHat.tocsr()


@numba.njit
def _stencil_x(i, j, k):
    return [(i, j, k), (i + 1, j, k)]


@numba.njit
def _stencil_y(i, j, k):
    return [(i, j, k), (i, j + 1, k)]


@numba.njit
def _stencil_z(i, j, k):
    return [(i, j, k), (i, j, k + 1)]


@numba.njit
def _kernel(shape_row, shape_col, offset, direction, widths, stencil,
            rows, cols, data):
    ndim = len(shape_row)
    size = 1
    for s in shape_row:
        size *= s
    stencil_size = len(stencil(0, 0, 0))
    for I in range(size):
        row = I + offset
        ijk = _ijk(I, shape_row)
        ijk_stencil = stencil(*ijk)
        _slice = slice(stencil_size * row, stencil_size * (row + 1))
        rows[_slice] = row
        cols[_slice] = [_idx(*i, shape_col) for i in ijk_stencil]
        c = 1 / widths[ijk[direction]]
        data[_slice] = [-c, c]

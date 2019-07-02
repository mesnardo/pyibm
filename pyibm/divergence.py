"""Implementation of the divergence operator."""

import numba
import numpy
from scipy.sparse import coo_matrix

from .grid import GridBase
from .misc import _idx, _ijk


def assemble_DHat(gridc, gridx, gridy, gridz=GridBase()):
    """Assemble the divergence operator DHat for a staggered grid.

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
    DHat: scipy.sparse.csr_matrix
        The divergence operator.

    """
    ndim = gridc.ndim  # number of dimension
    size = gridx.size + gridy.size + gridz.size  # number of columns
    nnz = 2 * ndim  # estimated number of nonzeros per row
    rows = numpy.zeros(nnz * gridc.size, dtype=numpy.int32)
    cols = numpy.zeros(nnz * gridc.size, dtype=numpy.int32)
    data = numpy.zeros(nnz * gridc.size, dtype=numpy.float64)
    # Assemble rows for GridFaceX points.
    offset = 0  # column index offset
    dy = gridx.y.get_widths()
    dz = numpy.array([1.0]) if ndim == 2 else gridx.z.get_widths()
    _kernel(gridc.shape, gridx.shape, offset, 0, dy, dz, _stencil_x,
            rows, cols, data)
    # Assemble rows for GridFaceY points.
    offset += gridx.size  # update offset index
    dx = gridy.x.get_widths()
    dz = numpy.array([1.0]) if ndim == 2 else gridy.z.get_widths()
    _kernel(gridc.shape, gridy.shape, offset, 1, dx, dz, _stencil_y,
            rows, cols, data)
    if ndim == 3:
        # Assemble rows for GridFaceZ points.
        offset += gridy.size  # update offset index
        dx = gridz.x.get_widths()
        dy = gridz.y.get_widths()
        _kernel(gridc.shape, gridz.shape, offset, 2, dx, dy, _stencil_z,
                rows, cols, data)
    # Assemble operator in COO format.
    DHat = coo_matrix((data, (rows, cols)),
                      shape=(gridc.size, size))
    # Return operator in CSR format.
    return DHat.tocsr()


@numba.njit
def _stencil_x(i, j, k):
    return [(i - 1, j, k), (i, j, k)]


@numba.njit
def _stencil_y(i, j, k):
    return [(i, j - 1, k), (i, j, k)]


@numba.njit
def _stencil_z(i, j, k):
    return [(i, j, k - 1), (i, j, k)]


@numba.njit
def _kernel(shape_row, shape_col, offset, direction, dx, dy, stencil,
            rows, cols, data):
    ndim = len(shape_row)
    N = shape_row[ndim - direction - 1]
    count = 2 * offset
    size = 1
    for s in shape_row:
        size *= s
    if direction == 0:
        sub = [1, 2]
    elif direction == 1:
        sub = [0, 2]
    elif direction == 2:
        sub = [0, 1]
    for I in range(size):
        row = I
        ijk = _ijk(I, shape_row)
        ijk_stencil = stencil(*ijk)
        i, j, k = ijk[sub[0]], ijk[sub[1]], ijk[direction]
        c = dx[i] * dy[j]
        if k > 0:
            rows[count] = row
            cols[count] = _idx(*ijk_stencil[0], shape_col) + offset
            data[count] = -c
            count += 1
        if k < N - 1:
            rows[count] = row
            cols[count] = _idx(*ijk_stencil[1], shape_col) + offset
            data[count] = c
            count += 1

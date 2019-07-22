"""Implementation of the scaling operators."""

import numba
import numpy
from scipy.sparse import coo_matrix, identity

from .grid import GridBase
from .misc import _idx, _ijk


def assemble_MHat(gridx, gridy, gridz=GridBase()):
    """Assemble the diagonal sparse matrix MHat in CSR format.

    Parameters
    ----------
    gridx : pyibm.GridFaceX
        Grid at X faces.
    gridy : pyibm.GridFaceY
        Grid at Y faces.
    gridz : pyibm.GridFaceZ (optional)
        Grid at Z faces (for 3D cases); default: GridBase().

    Returns
    -------
    MHat: scipy.sparse.csr_matrix
        The diagonal operator MHat.

    """
    ndim = gridx.ndim
    size = gridx.size + gridy.size + gridz.size
    nnz = 1  # estimated number of nonzeros per row
    rows = numpy.zeros(nnz * size, dtype=numpy.int32)
    cols = numpy.zeros(nnz * size, dtype=numpy.int32)
    data = numpy.zeros(nnz * size, dtype=numpy.float64)
    # Assemble rows for GridFaceX points.
    offset = 0  # offset index
    dx = gridx.x.get_widths()
    _kernel_MHat(gridx.shape, offset, 0, dx, rows, cols, data)
    # Assemble rows for GridFaceY points.
    offset += gridx.size  # update offset index
    dy = gridy.y.get_widths()
    _kernel_MHat(gridy.shape, offset, 1, dy, rows, cols, data)
    if ndim == 3:
            # Assemble rows for GridFaceX points.
        offset += gridy.size  # update offset index
        dz = gridz.z.get_widths()
        _kernel_MHat(gridz.shape, offset, 2, dz, rows, cols, data)
    # Assemble operator in COO format.
    MHat = coo_matrix((data, (rows, cols)), shape=(size, size))
    # Return operator in CSR format.
    return MHat.tocsr()


@numba.njit
def _kernel_MHat(shape, offset, direction, dx, rows, cols, data):
    size = 1
    for s in shape:
        size *= s
    for row in range(size):
        ijk = _ijk(row, shape)
        c = dx[ijk[direction]]
        rows[row + offset] = row + offset
        cols[row + offset] = row + offset
        data[row + offset] = c


def assemble_R(gridx, gridy, gridz=GridBase()):
    """Assemble the diagonal sparse matrix R in CSR format.

    Parameters
    ----------
    gridx : pyibm.GridFaceX
        Grid at X faces.
    gridy : pyibm.GridFaceY
        Grid at Y faces.
    gridz : pyibm.GridFaceZ (optional)
        Grid at Z faces (for 3D cases); default: GridBase().

    Returns
    -------
    R: scipy.sparse.csr_matrix
        The diagonal operator R.

    """
    ndim = gridx.ndim
    size = gridx.size + gridy.size + gridz.size
    nnz = 1
    rows = numpy.zeros(nnz * size, dtype=numpy.int32)
    cols = numpy.zeros(nnz * size, dtype=numpy.int32)
    data = numpy.zeros(nnz * size, dtype=numpy.float64)
    # Assemble rows for GridFaceX points.
    offset = 0  # offset index
    dy = gridx.y.get_widths()
    dz = numpy.array([1.0]) if ndim == 2 else gridx.z.get_widths()
    _kernel_R(gridx.shape, offset, 0, dy, dz, rows, cols, data)
    # Assemble rows for GridFaceY points.
    offset += gridx.size  # update offset index
    dx = gridy.x.get_widths()
    dz = numpy.array([1.0]) if ndim == 2 else gridy.z.get_widths()
    _kernel_R(gridy.shape, offset, 1, dx, dz, rows, cols, data)
    if ndim == 3:
        # Assemble rows for GridFaceZ points.
        offset += gridy.size  # update offset index
        dx = gridz.x.get_widths()
        dy = gridz.y.get_widths()
        _kernel_R(gridz.shape, offset, 2, dx, dy, rows, cols, data)
    # Assemble operator in COO format.
    R = coo_matrix((data, (rows, cols)), shape=(size, size))
    # Return operator in CSR format.
    return R.tocsr()


@numba.njit
def _kernel_R(shape, offset, direction, dy, dz, rows, cols, data):
    size = 1
    for s in shape:
        size *= s
    if direction == 0:
        sub = [1, 2]
    elif direction == 1:
        sub = [0, 2]
    elif direction == 2:
        sub = [0, 1]
    for row in range(size):
        ijk = _ijk(row, shape)
        j, k = ijk[sub[0]], ijk[sub[1]]
        c = dy[j] * dz[k]
        rows[row + offset] = row + offset
        cols[row + offset] = row + offset
        data[row + offset] = c


def diagonal_inv(A):
    """Assemble the inverse of the diagonal operator A."""
    AInv = A.tocsr()
    AInv.data = 1 / AInv.data
    return AInv


def assemble_BN(gridx, gridy, gridz=GridBase(),
                dt=1.0, alpha=0.5, N=1, L=None, M=None):
    """Assemble diagonal operator BN."""
    assert N >= 1, "N should be >= 1"
    size = gridx.size + gridy.size + gridz.size
    Bn = dt * identity(size)
    if N == 1:
        return Bn.tocsr()
    assert L is not None, "Missing L"
    if M is not None:
        MInv = diagonal_inv(M)
    else:
        MInv = identity(size)
    P = identity(size)
    for k in range(2, N + 1):
        P = P @ MInv @ L
        Bn += dt**k * alpha**(k - 1) * P @ MInv
    return Bn.tocsr()


def assemble_surfaces(body):
    """Assemble operator with surface areas."""
    S = body.ds * identity(body.ndim * body.size)
    return S

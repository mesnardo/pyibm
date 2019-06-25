"""Module with implementation of the operators."""

from collections.abc import Iterable
import numpy
from scipy.sparse import coo_matrix, csr_matrix

from .delta import delta, delta_roma_et_al_1999


def set_row(row, cols, data, A):
    """Set entries in COO matrix."""
    if not isinstance(cols, Iterable):
        cols, data = [cols], [data]
    A.row = numpy.append(A.row, [row for _ in range(len(cols))])
    A.col = numpy.append(A.col, cols)
    A.data = numpy.append(A.data, data)
    return


def assemble_GHat(gridc, gridx, gridy, gridz=None):
    """Assemble the gradient operator."""
    ndim = gridc.ndim
    size = gridx.size + gridy.size
    if ndim == 3:
        assert gridz is not None, 'Missing gridz'
        size += gridz.size
    GHat = coo_matrix((size, gridc.size))
    row, offset = 0, 0
    dx = gridx.x.get_widths()
    for _ in range(gridx.size):
        try:
            i, j, k = gridx.ijk(row - offset)
        except:
            (i, j), k = gridx.ijk(row - offset), None
        Iw, Ie = gridc.idx(i, j, k), gridc.idx(i + 1, j, k)
        c = 1 / dx[i]
        set_row(row, [Iw, Ie], [-c, c], GHat)
        row += 1
    offset += gridx.size
    dy = gridy.y.get_widths()
    for _ in range(gridy.size):
        try:
            i, j, k = gridy.ijk(row - offset)
        except:
            (i, j), k = gridy.ijk(row - offset), None
        Is, In = gridc.idx(i, j, k), gridc.idx(i, j + 1, k)
        c = 1 / dy[j]
        set_row(row, [Is, In], [-c, c], GHat)
        row += 1
    if ndim == 3:
        offset += gridy.size
        dz = gridz.z.get_widths()
        for _ in range(gridz.size):
            i, j, k = gridz.ijk(row - offset)
            Id, Iu = gridc.idx(i, j, k), gridc.idx(i, j, k + 1)
            c = 1 / dz[k]
            set_row(row, [Id, Iu], [-c, c], GHat)
            row += 1
    return GHat.tocsr()


def assemble_DHat(gridc, gridx, gridy, gridz=None):
    """Assemble the gradient operator."""
    ndim = gridc.ndim
    size = gridx.size + gridy.size
    if ndim == 3:
        assert gridz is not None, 'Missing gridz'
        size += gridz.size
    DHat = coo_matrix((gridc.size, size))
    offset = 0
    dy = gridx.y.get_widths()
    dz = ([1.0] if ndim == 2 else gridx.z.get_widths())
    for row in range(gridc.size):
        try:
            i, j, k = gridc.ijk(row)
        except:
            (i, j), k = gridc.ijk(row), None
        Iw, Ie = gridx.idx(i - 1, j, k), gridx.idx(i, j, k)
        c = dy[j] * dz[k]
        set_row(row, [Iw + offset, Ie + offset], [-c, c], DHat)
    offset += gridx.size
    dx = gridy.x.get_widths()
    dz = ([1.0] if gridy.ndim == 2 else gridy.z.get_widths())
    for row in range(gridc.size):
        try:
            i, j, k = gridc.ijk(row)
        except:
            (i, j), k = gridc.ijk(row), None
        Is, In = gridy.idx(i, j - 1, k), gridy.idx(i, j, k)
        c = dx[i] * dz[k]
        set_row(row, [Is + offset, In + offset], [-c, c], DHat)
    if ndim == 3:
        offset += gridy.size
        dx = gridz.x.get_widths()
        dy = gridz.y.get_widths()
        for row in range(gridc.size):
            i, j, k = gridc.ijk(row)
            Id, Iu = gridz.idx(i, j, k - 1), gridz.idx(i, j, k)
            c = dx[i] * dy[j]
            set_row(row, [Id + offset, Iu + offset], [-c, c], DHat)
    return DHat.tocsr()


def assemble_L(gridx, gridy):
    """Assemble Laplacian operator."""
    L = numpy.zeros((gridx.size + gridy.size, gridx.size + gridy.size))
    dx, dy = gridx.dx, gridx.dy
    for row in range(gridx.size):
        i, j = gridx.ij(row)
        if i > 0:
            L[row, gridx.idx(i - 1, j)] = 1 / dx**2
        L[row, row] += -2 / dx**2
        if i < gridx.nx - 1:
            L[row, gridx.idx(i + 1, j)] = 1 / dx**2
        if j > 0:
            L[row, gridx.idx(i, j - 1)] = 1 / dy**2
        L[row, row] += -2 / dy**2
        if j < gridx.ny - 1:
            L[row, gridx.idx(i, j + 1)] = 1 / dy**2
    offset = gridx.size
    for row in range(offset, gridy.size + offset):
        i, j = gridy.ij(row - offset)
        if i > 0:
            L[row, gridy.idx(i - 1, j) + offset] = 1 / dx**2
        L[row, row] += -2 / dx**2
        if i < gridy.nx - 1:
            L[row, gridy.idx(i + 1, j) + offset] = 1 / dx**2
        if j > 0:
            L[row, gridy.idx(i, j - 1) + offset] = 1 / dy**2
        L[row, row] += -2 / dy**2
        if j < gridy.ny - 1:
            L[row, gridy.idx(i, j + 1) + offset] = 1 / dy**2
    return coo_matrix(L)


def assemble_BN(gridx, gridy, dt, N=1, L=None, Minv=None):
    """Assemble diagonal operator BN."""
    assert N >= 1, "N should >= 1"
    I = numpy.diag(numpy.ones(gridx.size + gridy.size))
    Bn = dt * I
    if N == 1:
        return coo_matrix(Bn)
    else:
        assert L is not None, "Missing L"
        assert Minv is not None, "Missing Minv"
        P = I.copy()
        for k in range(2, N + 1):
            P = P @ Minv @ L
            Bn += dt**k / 2**(k - 1) * P @ Minv
        return coo_matrix(Bn)


def assemble_R(gridx, gridy, gridz=None):
    """Assemble diagonal operator R."""
    ndim = gridx.ndim
    size = gridx.size + gridy.size
    if ndim == 3:
        assert gridz is not None, 'Missing gridz'
        size += gridz.size
    R = coo_matrix((size, size))
    row, offset = 0, 0
    dy = gridx.y.get_widths()
    dz = ([1.0] if ndim == 2 else gridx.z.get_widths())
    for _ in range(gridx.size):
        try:
            i, j, k = gridx.ijk(row - offset)
        except:
            (i, j), k = gridx.ijk(row - offset), 1
        set_row(row, row, dy[j] * dz[k], R)
        row += 1
    offset += gridx.size
    dx = gridy.x.get_widths()
    dz = ([1.0] if ndim == 2 else gridy.z.get_widths())
    for _ in range(gridy.size):
        try:
            i, j, k = gridy.ijk(row - offset)
        except:
            (i, j), k = gridy.ijk(row - offset), 1
        set_row(row, row, dx[i] * dz[k], R)
        row += 1
    if ndim == 3:
        offset += gridy.size
        dx = gridz.x.get_widths()
        dy = gridz.y.get_widths()
        for _ in range(gridz.size):
            i, j, k = gridx.ijk(row - offset)
            set_row(row, row, dx[i] * dy[j], R)
            row += 1
    return R.tocsr()


def assemble_RInv(R):
    """Assemble diagonal operator RInv."""
    RInv = csr_matrix(R, copy=True)
    RInv.data = numpy.array([1 / d for d in R.data])
    return RInv


def assemble_MHat(gridx, gridy, gridz=None):
    """Assemble diagonal operator MHat."""
    ndim = gridx.ndim
    size = gridx.size + gridy.size
    if ndim == 3:
        assert gridz is not None, 'Missing gridz'
        size += gridz.size
    MHat = coo_matrix((size, size))
    row, offset = 0, 0
    dx = gridx.x.get_widths()
    for _ in range(gridx.size):
        try:
            i, j, k = gridx.ijk(row - offset)
        except:
            (i, j), k = gridx.ijk(row - offset), 1
        set_row(row, row, dx[i], MHat)
        row += 1
    offset += gridx.size
    dy = gridy.y.get_widths()
    for _ in range(gridy.size):
        try:
            i, j, k = gridy.ijk(row - offset)
        except:
            (i, j), k = gridy.ijk(row - offset), 1
        set_row(row, row, dy[j], MHat)
        row += 1
    if ndim == 3:
        offset += gridy.size
        dz = gridz.z.get_widths()
        for _ in range(gridz.size):
            i, j, k = gridz.ijk(row - offset)
            set_row(row, row, dz[k], MHat)
            row += 1
    return MHat.tocsr()


def assemble_MHatInv(MHat):
    """Assemble diagonal operator MHatInv."""
    MHatInv = csr_matrix(MHat, copy=True)
    MHatInv.data = numpy.array([1 / d for d in MHat.data])
    return MHatInv


def assemble_delta(gridx, gridy, body, kernel=delta_roma_et_al_1999):
    """Assemble delta operator."""
    M, N = body.ndim * body.size, gridx.size + gridy.size
    Op = numpy.zeros((M, N))
    X, Y = body.x, body.y
    x, y = gridx.x, gridx.y
    for k in range(body.size):
        row = body.ndim * k
        for col in range(gridx.size):
            i, j = col % gridx.nx, col // gridx.nx
            dist = [abs(x[i] - X[k]), abs(y[j] - Y[k])]
            val = delta(dist, gridx.dx, kernel=kernel)
            Op[row, col] = (val > 1e-6) * val
    offset = gridx.size
    x, y = gridy.x, gridy.y
    for k in range(body.size):
        row = body.ndim * k + 1
        for col in range(gridy.size):
            i, j = col % gridy.nx, col // gridy.nx
            dist = [abs(x[i] - X[k]), abs(y[j] - Y[k])]
            val = delta(dist, gridy.dy, kernel=kernel)
            Op[row, col + offset] = (val > 1e-6) * val
    return coo_matrix(Op)


def assemble_surfaces(body):
    """Assemble operator with surface areas."""
    S = numpy.diag(body.ds * numpy.ones(body.ndim * body.size))
    return coo_matrix(S)

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
        Grid at Z faces (for 3D cases); default: None.

    Returns
    -------
    GHat: scipy.sparse.csr_matrix
        The gradient operator.

    """
    # Initialize empty sparse matrix in COO format.
    ndim = gridc.ndim
    size = gridx.size + gridy.size
    if ndim == 3:
        assert gridz is not None, 'Missing gridz'
        size += gridz.size
    GHat = coo_matrix((size, gridc.size))
    # Assemble rows for GridFaceX points.
    row, offset = 0, 0  # row index and offset index
    dx = gridx.x.get_widths()  # grid spacings in x-direction
    for _ in range(gridx.size):
        # Get directional indices.
        i, j, k = gridx.ijk(row - offset)
        # Get natural index for West and East points.
        Iw, Ie = gridc.idx(i, j, k), gridc.idx(i + 1, j, k)
        # Set row values for West and East points.
        c = 1 / dx[i]
        set_row(row, [Iw, Ie], [-c, c], GHat)
        row += 1
    # Assemble rows for GridFaceY points.
    offset += gridx.size  # update offset
    dy = gridy.y.get_widths()
    for _ in range(gridy.size):
        # Get directional indices.
        i, j, k = gridy.ijk(row - offset)
        # Get natural index for South and North points.
        Is, In = gridc.idx(i, j, k), gridc.idx(i, j + 1, k)
        # Set row values for South and North points.
        c = 1 / dy[j]
        set_row(row, [Is, In], [-c, c], GHat)
        row += 1
    if ndim == 3:
        # Assemble rows for GridFaceZ points.
        offset += gridy.size  # update offset
        dz = gridz.z.get_widths()
        for _ in range(gridz.size):
            # Get directional indices.
            i, j, k = gridz.ijk(row - offset)
            # Get natural index for Back and Front points.
            Ib, If = gridc.idx(i, j, k), gridc.idx(i, j, k + 1)
            # Set row values for Back and Front points.
            c = 1 / dz[k]
            set_row(row, [Ib, If], [-c, c], GHat)
            row += 1
    # Return sparse matrix in CSR format
    return GHat.tocsr()


def assemble_DHat(gridc, gridx, gridy, gridz=None):
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
        Grid at Z faces (for 3D cases); default: None.

    Returns
    -------
    DHat: scipy.sparse.csr_matrix
        The divergence operator.

    """
    # Initialize empty sparse matrix in COO format.
    ndim = gridc.ndim
    size = gridx.size + gridy.size
    if ndim == 3:
        assert gridz is not None, 'Missing gridz'
        size += gridz.size
    DHat = coo_matrix((gridc.size, size))
    # Assemble columns for GridFaceX points.
    offset = 0  # column offset index
    dy = gridx.y.get_widths()
    dz = [1.0] if ndim == 2 else gridx.z.get_widths()
    for row in range(gridc.size):
        # Get directional indices of cell-centered point.
        i, j, k = gridc.ijk(row)
        # Get natural index for West and East points.
        Iw, Ie = gridx.idx(i - 1, j, k), gridx.idx(i, j, k)
        # Set row values for West and East points.
        c = dy[j] * dz[k]
        if i > 0:  # West point not at left boundary
            set_row(row, Iw, -c, DHat)
        if i < gridc.M - 1:  # East point not at right boundary
            set_row(row, Ie, c, DHat)
    # Assemble columns for GridFaceY points.
    offset += gridx.size  # update offset
    dx = gridy.x.get_widths()
    dz = [1.0] if ndim == 2 else gridy.z.get_widths()
    for row in range(gridc.size):
        # Get directional indices of cell-centered point.
        i, j, k = gridc.ijk(row)
        # Get natural index for South and North points.
        Is, In = gridy.idx(i, j - 1, k), gridy.idx(i, j, k)
        # Set row values for South and North points.
        c = dx[i] * dz[k]
        if j > 0:  # South point not at bottom boundary
            set_row(row, Is + offset, -c, DHat)
        if j < gridc.N - 1:  # North point not at top boundary
            set_row(row, In + offset, c, DHat)
    if ndim == 3:
        # Assemble columns for GridFaceZ points.
        offset += gridy.size  # update offset
        dx = gridz.x.get_widths()
        dy = gridz.y.get_widths()
        for row in range(gridc.size):
            # Get directional indices of cell-centered point.
            i, j, k = gridc.ijk(row)
            # Get natural index for Back and Front points.
            Ib, If = gridz.idx(i, j, k - 1), gridz.idx(i, j, k)
            # Set row values for Back and Front points.
            c = dx[i] * dy[j]
            if k > 0:  # Back point not at back boundary
                set_row(row, Ib + offset, -c, DHat)
            if k < gridc.P - 1:  # Front point not at front boundary
                set_row(row, If + offset, c, DHat)
    # Return sparse matrix in CSR format
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
    """Assemble the diagonal sparse matrix R in CSR format.

    Parameters
    ----------
    gridx : pyibm.GridFaceX
        Grid at X faces.
    gridy : pyibm.GridFaceY
        Grid at Y faces.
    gridz : pyibm.GridFaceZ (optional)
        Grid at Z faces (for 3D cases); default: None.

    Returns
    -------
    R: scipy.sparse.csr_matrix
        The diagonal operator R.

    """
    # Initialize sparse matrix in COO format.
    ndim = gridx.ndim
    size = gridx.size + gridy.size
    if ndim == 3:
        assert gridz is not None, 'Missing gridz'
        size += gridz.size
    R = coo_matrix((size, size))
    # Assemble rows for GridFaceX points.
    dy = gridx.y.get_widths()
    dz = [1.0] if ndim == 2 else gridx.z.get_widths()
    for I in range(gridx.size):
        # Get directional indices of GridFaceX point.
        i, j, k = gridx.ijk(I)
        # Set row value.
        set_row(I, I, dy[j] * dz[k], R)
    # Assemble rows for GridFaceY points.
    offset = gridx.size  # set offset
    dx = gridy.x.get_widths()
    dz = [1.0] if ndim == 2 else gridy.z.get_widths()
    for I in range(gridy.size):
        # Get directional indices of GridFaceY point.
        i, j, k = gridy.ijk(I)
        # Set row value.
        set_row(I + offset, I + offset, dx[i] * dz[k], R)
    if ndim == 3:
        # Assemble rows for GridFaceZ points.
        offset += gridy.size  # update offset
        dx = gridz.x.get_widths()
        dy = gridz.y.get_widths()
        for I in range(gridz.size):
            # Get directional indices of GridFaceZ point.
            i, j, k = gridz.ijk(I)
            # Set row value.
            set_row(I + offset, I + offset, dx[i] * dy[j], R)
    # Return sparse matrix in CSR format.
    return R.tocsr()


def assemble_RInv(R):
    """Assemble diagonal sparse matrix RInv in CSR format.

    Parameters
    ----------
    R : scipy.sparse.csr_matrix
        The diagonal operator R.

    Returns
    -------
    RInv : scipy.sparse.csr_matrix
        The inverse of the diagonal operator R.

    """
    RInv = csr_matrix(R, copy=True)
    RInv.data = numpy.array([1 / d for d in R.data])
    return RInv


def assemble_MHat(gridx, gridy, gridz=None):
    """Assemble the diagonal sparse matrix MHat in CSR format.

    Parameters
    ----------
    gridx : pyibm.GridFaceX
        Grid at X faces.
    gridy : pyibm.GridFaceY
        Grid at Y faces.
    gridz : pyibm.GridFaceZ (optional)
        Grid at Z faces (for 3D cases); default: None.

    Returns
    -------
    MHat: scipy.sparse.csr_matrix
        The diagonal operator MHat.

    """
    # Initialize sparse matrix in COO format.
    ndim = gridx.ndim
    size = gridx.size + gridy.size
    if ndim == 3:
        assert gridz is not None, 'Missing gridz'
        size += gridz.size
    MHat = coo_matrix((size, size))
    # Assemble rows for GridFaceX points.
    dx = gridx.x.get_widths()
    for I in range(gridx.size):
        # Get directional indices of GridFaceX point.
        i, j, k = gridx.ijk(I)
        # Set row value.
        set_row(I, I, dx[i], MHat)
    # Assemble rows for GridFaceY points.
    offset = gridx.size  # set offset
    dy = gridy.y.get_widths()
    for I in range(gridy.size):
        # Get directional indices for GridFaceY point.
        i, j, k = gridy.ijk(I)
        # Set row value.
        set_row(I + offset, I + offset, dy[j], MHat)
    if ndim == 3:
        # Aseemble rows for GridFaceZ points.
        offset += gridy.size  # update offset
        dz = gridz.z.get_widths()
        for I in range(gridz.size):
            # Get direction indices for GridFaceZ point.
            i, j, k = gridz.ijk(I)
            # Set row value.
            set_row(I + offset, I + offset, dz[k], MHat)
    # Return sparse matrix in CSR format.
    return MHat.tocsr()


def assemble_MHatInv(MHat):
    """Assemble diagonal sparse matrix MHatInv in CSR format.

    Parameters
    ----------
    MHat : scipy.sparse.csr_matrix
        The diagonal operator MHat.

    Returns
    -------
    MHatInv : scipy.sparse.csr_matrix
        The inverse of the diagonal operator MHat.

    """
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

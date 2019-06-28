"""Module with implementation of the operators."""

from collections.abc import Iterable
import numpy
from scipy.sparse import coo_matrix, csr_matrix

from .delta import delta, delta_roma_et_al_1999
from .grid import GridBase


def set_row(row, cols, data, A):
    """Set entries in COO matrix."""
    if not isinstance(cols, Iterable):
        cols, data = [cols], [data]
    A.row = numpy.append(A.row, [row for _ in range(len(cols))])
    A.col = numpy.append(A.col, cols)
    A.data = numpy.append(A.data, data)
    return


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
    # Initialize empty sparse matrix in COO format.
    ndim = gridc.ndim
    size = gridx.size + gridy.size + gridz.size
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
    # Initialize empty sparse matrix in COO format.
    ndim = gridc.ndim
    size = gridx.size + gridy.size + gridz.size
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


def assemble_LHat(gridx, gridy, gridz=GridBase()):
    """Assemble Laplacian operator LHat."""
    ndim = gridx.ndim
    size = gridx.size + gridy.size + gridz.size
    L = coo_matrix((size, size))
    offset = 0
    for grid in ([gridx, gridy, gridz][:ndim]):
        dx, dy = grid.x.get_widths(), grid.y.get_widths()
        dz = [1.0] if ndim == 2 else grid.z.get_widths()
        for I in range(grid.size):
            cols, vals = [], []
            i, j, k = grid.ijk(I)

            Iw, Ie = grid.idx(i - 1, j, k), grid.idx(i + 1, j, k)
            Is, In = grid.idx(i, j - 1, k), grid.idx(i, j + 1, k)

            dx_w = dx[i] if i == 0 else dx[i - 1]
            dx_e = dx[i] if i == grid.M - 1 else dx[i + 1]
            dy_s = dy[j] if j == 0 else dy[j - 1]
            dy_n = dy[j] if j == grid.N - 1 else dy[j + 1]

            Cw = 2 / dx[i] / (dx[i] + dx_w)
            Ce = 2 / dx[i] / (dx[i] + dx_e)
            Cs = 2 / dy[j] / (dy[j] + dy_s)
            Cn = 2 / dy[j] / (dy[j] + dy_n)

            if i > 0:
                cols.append(Iw)
                vals.append(Cw)
            if i < grid.M - 1:
                cols.append(Ie)
                vals.append(Ce)
            if j > 0:
                cols.append(Is)
                vals.append(Cs)
            if j < grid.N - 1:
                cols.append(In)
                vals.append(Cn)
            C = - (Cw + Ce + Cs + Cn)
            if ndim == 3:
                Ib, If = grid.idx(i, j, k - 1), grid.idx(i, j, k + 1)
                dz_b = dz[k] if k == 0 else dz[k - 1]
                dz_f = dz[k] if k == grid.P - 1 else dz[k + 1]
                Cb = 2 / dz[k] / (dz[k] + dz_b)
                Cf = 2 / dz[k] / (dz[k] + dz_f)
                if k > 0:
                    cols.append(Ib)
                    vals.append(Cb)
                if k < grid.P - 1:
                    cols.append(If)
                    vals.append(Cf)
                C += - (Cb + Cf)
            cols.append(I)
            vals.append(C)
            set_row(I + offset, [c + offset for c in cols], vals, L)
        offset += grid.size
    return csr_matrix(L)


def assemble_BN(gridx, gridy, gridz=GridBase(),
                dt=1.0, N=1, L=None, MInv=None):
    """Assemble diagonal operator BN."""
    assert N >= 1, "N should be >= 1"
    I = numpy.diag(numpy.ones(gridx.size + gridy.size + gridz.size))
    Bn = dt * I
    if N == 1:
        return csr_matrix(Bn)
    else:
        assert L is not None, "Missing L"
        assert MInv is not None, "Missing MInv"
        P = I.copy()
        for k in range(2, N + 1):
            P = P @ MInv @ L
            Bn += dt**k / 2**(k - 1) * P @ MInv
        return csr_matrix(Bn)


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
    # Initialize sparse matrix in COO format.
    ndim = gridx.ndim
    size = gridx.size + gridy.size + gridz.size
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
    # Initialize sparse matrix in COO format.
    ndim = gridx.ndim
    size = gridx.size + gridy.size + gridz.size
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


def assemble_delta(body, gridc, gridx, gridy, gridz=GridBase(),
                   kernel=delta_roma_et_al_1999, kernel_size=2):
    """Assemble the delta sparse matrix in CSR format."""
    # Initialize sparse matrix in COO format.
    ndim = body.ndim
    size = gridx.size + gridy.size + gridz.size
    Op = coo_matrix((ndim * body.size, size))
    X, Y, Z = body.x, body.y, body.z
    # Get cell widths in the uniform region.
    i, j, k = gridc.ijk(body.neighbors[0])
    dx, dy = gridc.x.get_widths()[i], gridc.y.get_widths()[j]
    # Assemble rows.
    ks = kernel_size  # use nickname
    if ndim == 2:
        offset = 0
        dr = [dx, dy]
        for dof, grid in enumerate([gridx, gridy]):
            x, y = grid.x.vertices, grid.y.vertices
            for l in range(body.size):
                row = body.ndim * l + dof  # row index
                i, j, k = gridc.ijk(body.neighbors[l])
                for jj in range(j - ks, j + ks + 1):
                    ry = abs(Y[l] - y[jj])
                    for ii in range(i - ks, i + ks + 1):
                        rx = abs(X[l] - x[ii])
                        r = [rx, ry]
                        val = delta(r, dr, kernel=kernel)
                        col = grid.idx(ii, jj, k) + offset
                        set_row(row, col, val, Op)
            offset += grid.size
    elif ndim == 3:
        offset = 0
        dz = gridc.z.get_widths()[k]
        dr = [dx, dy, dz]
        for dof, grid in enumerate([gridx, gridy, gridz]):
            x, y, z = grid.x.vertices, grid.y.vertices, grid.z.vertices
            for l in range(body.size):
                row = body.ndim * l + dof  # row index
                i, j, k = gridc.ijk(body.neighbors[l])
                for kk in range(k - ks, k + ks + 1):
                    rz = abs(Z[l] - z[kk])
                    for jj in range(j - ks, j + ks + 1):
                        ry = abs(Y[l] - y[jj])
                        for ii in range(i - ks, i + ks + 1):
                            rx = abs(X[l] - x[ii])
                            r = [rx, ry, rz]
                            val = delta(r, dr, kernel=kernel)
                            col = grid.idx(ii, jj, kk) + offset
                            set_row(row, col, val, Op)
            offset += grid.size
    # Return sparse matrix in CSR format.
    Op.eliminate_zeros()
    return Op.tocsr()


def assemble_surfaces(body):
    """Assemble operator with surface areas."""
    S = numpy.diag(body.ds * numpy.ones(body.ndim * body.size))
    return csr_matrix(S)

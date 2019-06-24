"""Module with implementation of the operators."""

import numpy
from scipy.sparse import coo_matrix

from .delta import delta, delta_roma_et_al_1999


def assemble_G_hat(gridc, gridx, gridy, gridz=None):
    """Assemble the gradient operator."""
    size = gridx.size + gridy.size
    if gridc.ndim == 3:
        assert gridz is not None, 'Missing gridz'
        size += gridz.size
    G = numpy.zeros((size, gridc.size))
    row, offset = 0, 0
    widths = gridc.x.get_widths()
    for _ in range(gridx.size):
        try:
            i, j, k = gridx.ijk(row - offset)
        except:
            (i, j), k = gridx.ijk(row - offset), None
        print(k, j, i)
        G[row, gridc.idx(i, j, k)] = -1.0 / widths[i]
        G[row, gridc.idx(i + 1, j, k)] = -1.0 / widths[i + 1]
        row += 1
    offset += gridx.size
    for _ in range(gridy.size):
        try:
            i, j, k = gridy.ijk(row - offset)
        except:
            (i, j), k = gridy.ijk(row - offset), None
        G[row, gridc.idx(i, j, k)] = -1.0 / widths[j]
        G[row, gridc.idx(i, j + 1, k)] = -1.0 / widths[j + 1]
        row += 1
    if gridc.ndim == 3:
        offset += gridy.size
        for _ in range(gridz.size):
            i, j, k = gridz.ijk(row - offset)
            G[row, gridc.idx(i, j, k)] = -1.0 / widths[k]
            G[row, gridc.idx(i, j, k + 1)] = -1.0 / widths[k + 1]
            row += 1
    return coo_matrix(G)


def assemble_Dhat(gridc, gridx, gridy):
    """Assemble the divergence operator."""
    D = numpy.zeros((gridc.size, gridx.size + gridy.size))
    dx, dy = gridc.dx, gridc.dy
    offset = gridx.size
    for row in range(gridc.size):
        i, j = gridc.ij(row)
        if i > 0:
            D[row, gridx.idx(i - 1, j)] = -dy
        if i < gridc.nx - 1:
            D[row, gridx.idx(i, j)] = dy
        if j > 0:
            D[row, gridy.idx(i, j - 1) + offset] = -dx
        if j < gridc.ny - 1:
            D[row, gridy.idx(i, j) + offset] = dx
    return coo_matrix(D)


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


def assemble_R(gridx, gridy):
    """Assemble diagonal operator R."""
    Rx = gridx.dy * numpy.ones(gridx.size)
    Ry = gridy.dx * numpy.ones(gridy.size)
    R = numpy.diag(numpy.concatenate((Rx, Ry)))
    return coo_matrix(R)


def assemble_Rinv(gridx, gridy):
    """Assemble diagonal operator Rinv."""
    Rinvx = 1 / gridx.dy * numpy.ones(gridx.size)
    Rinvy = 1 / gridy.dx * numpy.ones(gridy.size)
    Rinv = numpy.diag(numpy.concatenate((Rinvx, Rinvy)))
    return coo_matrix(Rinv)


def assemble_Mhat(gridx, gridy):
    """Assemble diagonal operator Mhat."""
    Mhatx = gridx.dx * numpy.ones(gridx.size)
    Mhaty = gridy.dy * numpy.ones(gridy.size)
    Mhat = numpy.diag(numpy.concatenate((Mhatx, Mhaty)))
    return coo_matrix(Mhat)


def assemble_Mhatinv(gridx, gridy):
    """Assemble diagonal operator Mhatinv."""
    Mhatinvx = gridx.dx * numpy.ones(gridx.size)
    Mhatinvy = gridy.dy * numpy.ones(gridy.size)
    Mhatinv = numpy.diag(numpy.concatenate((Mhatinvx, Mhatinvy)))
    return coo_matrix(Mhatinv)


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

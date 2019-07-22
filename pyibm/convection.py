"""Implement the routines to compute the convective terms."""

import cupy
import math
from numba import cuda


@cuda.jit
def _convective2d_x(ux, uy, dx, dy, out):
    """Kernel for the x-velocity convective terms."""
    i, j = cuda.grid(2)
    n, m = ux.shape
    if 1 <= i < m - 1 and 1 <= j < m - 1:
        hxx = (((ux[j, i + 1] + ux[j, i]) / 2)**2 -
               ((ux[j, i] + ux[j, i - 1]) / 2)**2) / dx[i]
        hxy = ((ux[j, i] + ux[j + 1, i]) / 2 *
               (uy[j, i] + uy[j, i + 1]) / 2 -
               (ux[j - 1, i] + ux[j, i]) / 2 *
               (uy[j - 1, i] + uy[j - 1, i + 1]) / 2) / dy[j]
        out[j, i] = hxx + hxy


def convective_x(ux, uy, threads=(16, 16)):
    """Convective terms for the x-velocity component."""
    out = cupy.empty(ux.values.shape)
    dx = ux.grid.x.get_widths()
    dy = ux.grid.y.get_widths()
    blocks = tuple(math.ceil(s / b) for s, b in zip(ux.shape, threads))
    _convective2d_x[blocks, threads](ux.values, uy.values,
                                     dx, dy, out)
    return cupy.asnumpy(out)


@cuda.jit
def _convective2d_y(ux, uy, dx, dy, out):
    """Kernel for the y-velocity convective terms."""
    i, j = cuda.grid(2)
    n, m = uy.shape
    if 1 <= i < m - 1 and 1 <= j < m - 1:
        hyx = ((ux[j, i] + ux[j + 1, i]) / 2 *
               (uy[j, i] + uy[j, i + 1]) / 2 -
               (ux[j, i - 1] + ux[j + 1, i - 1]) / 2 *
               (uy[j, i - 1] + uy[j, i]) / 2) / dx[i]
        hyy = (((uy[j, i] + uy[j + 1, i]) / 2)**2 -
               ((uy[j - 1, i] + uy[j, i]) / 2)**2) / dy[j]
        out[j, i] = hyx + hyy


def convective_y(ux, uy, threads=(16, 16)):
    """Convective terms for the y-velocity component."""
    out = cupy.empty(uy.values.shape)
    dx = uy.grid.x.get_widths()
    dy = uy.grid.y.get_widths()
    blocks = tuple(math.ceil(s / b) for s, b in zip(ux.shape, threads))
    _convective2d_y[blocks, threads](ux.values, uy.values,
                                     dx, dy, out)
    return cupy.asnumpy(out)

"""Implementation of miscellaneous functions."""

import numpy

from .field import EulerianField


def convection(ux, uy, uz=EulerianField()):
    """Compute the convective terms."""
    gridx, gridy, gridz = ux.grid, uy.grid, uz.grid
    size = gridx.size + gridy.size + gridz.size
    conv = numpy.empty(size)
    ndim = gridx.ndim
    if uz is not None:
        gridz = uz.grid
    dx = gridx.x.get_widths()

    if ndim == 2:
        dx = gridx.x.get_widths()
        for I in range(ux.size):
            i, j, _ = gridx.ijk(I)
            Iw, Ie = gridx.idx(i - 1, j), gridx.idx(i + 1, j)
            Is, In = gridx.idx(i, j - 1), gridx.idx(i, j + 1)
            conv[I] = 


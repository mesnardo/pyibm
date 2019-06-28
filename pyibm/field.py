"""Implementation of the class for an Eulerian field variable."""

import numpy

from .grid import GridBase

class EulerianField(object):
    """Eulerian field variable."""

    def __init__(self, grid=GridBase(), ic=0.0, bc=None):
        """Initialize the field."""
        self.grid = grid
        self.size = grid.size
        self.shape = grid.shape
        if grid.size > 0:
            self.bc = bc
            self.values = ic * numpy.ones(grid.size)

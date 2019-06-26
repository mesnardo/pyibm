"""Module with the implementation of the body class."""

import numpy


class Body(object):
    """Information about the immersed body."""

    def __init__(self, x, y, z=None, grid=None):
        """Initialize the body."""
        self.ndim = 2 if z is None else 3
        self.x, self.y, self.z = x, y, z
        self.z = numpy.array(x.size * [None]) if z is None else z
        self.size = x.size
        if grid is not None:
            self.neighbors = self.get_neighbors(grid)
        self.ds = numpy.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)

    def __repr__(self):
        """Representation of the body."""
        return 'Body(ndim={}, size={})'.format(self.ndim, self.size)

    def get_neighbors(self, grid):
        """Get closest Eulerian neighbors."""
        neighbors = numpy.empty_like(self.x, dtype=numpy.int32)
        for l in range(self.size):
            i = numpy.abs(grid.x.vertices - self.x[l]).argmin()
            j = numpy.abs(grid.y.vertices - self.y[l]).argmin()
            k = 0
            if self.ndim == 3:
                k = numpy.abs(grid.z.vertices - self.z[l]).argmin()
            neighbors[l] = grid.idx(i, j, k)
        return neighbors

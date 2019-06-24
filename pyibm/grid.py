"""Module with implementation of the grid classes."""

import functools
import math
import numpy
import operator


class GridBase(object):
    """Base class for the grid."""

    def __init__(self, grid=None, config=None):
        """Initialize the grid."""
        self.x, self.y, self.z = GridLine(), GridLine(), GridLine()
        self.size = 0
        self.ndim = 0
        self.shape = ()
        if grid is not None:
            self.create_from_grid(grid)
        elif config is not None:
            self.create(config)

    def __repr__(self, ndigits=6):
        """Representation of the grid.

        Parameters
        ----------
        ndigits : integer (optional)
            Number of digits to represent floats; default: 6.

        """
        sub_reprs = [getattr(self, direction).__repr__()
                     for direction in ['x', 'y', 'z'][:self.ndim]]
        return ('Grid(size={}, shape={}, gridlines=[\n{}])'
                .format(self.size, self.shape, ',\n'.join(sub_reprs)))

    def create(self, config):
        """Create the grid.

        Parameters
        ----------
        config : dictionary
            Configuration of the grid.

        """
        lines = {}
        for direction, node in config.items():
            assert direction in ['x', 'y', 'z']
            lines[direction] = GridLine(config=node)
        shape = []
        for direction in ['z', 'y', 'x']:
            if direction in lines.keys():
                self.ndim += 1
                shape.append(lines[direction].size)
                setattr(self, direction, lines[direction])
        self.shape = tuple(shape)
        self.size = functools.reduce(operator.mul, self.shape, 1)

    def create_from_grid(self, grid):
        """Create a grid from a base grid."""
        raise NotImplementedError()

    def idx(self, i, j, k=0):
        """Return the index given directional indices."""
        lda = self.shape[-1]
        if self.ndim == 3:
            return k * (lda * self.shape[-2]) + j * lda + i
        return j * lda + i

    def ijk(self, I):
        """Return the directional indices."""
        lda = self.shape[-1]
        i, j = I % lda, I // lda
        if self.ndim == 3:
            k = I // (lda * self.shape[-2])
            return i, j, k
        return i, j


class GridCellCentered(GridBase):
    """Class for cell-centered grid."""

    def __init__(self, *args, **kwargs):
        """Call the constructor of the base class."""
        super(GridCellCentered, self).__init__(*args, **kwargs)

    def create_from_grid(self, grid):
        """Create grid from a base grid."""
        for direction in ['x', 'y', 'z'][:grid.ndim]:
            gridline = getattr(grid, direction)
            start, end = gridline.start, gridline.end
            vertices = gridline.vertices
            vertices = 0.5 * (vertices[:-1] + vertices[1:])
            setattr(self, direction, GridLine(start=start, end=end,
                                              vertices=vertices))
        self.shape = tuple(n - 1 for n in grid.shape)
        self.size = functools.reduce(operator.mul, self.shape, 1)
        self.ndim = grid.ndim


class GridFaceX(GridBase):
    """Class for x-face centered grid."""

    def __init__(self, *args, **kwargs):
        """Call the constructor of the base class."""
        super(GridFaceX, self).__init__(*args, **kwargs)

    def create_from_grid(self, grid):
        """Create grid from a base grid."""
        for direction in ['x', 'y', 'z'][:grid.ndim]:
            gridline = getattr(grid, direction)
            start, end = gridline.start, gridline.end
            vertices = gridline.vertices
            if direction is 'x':
                vertices = vertices[1:-1]
            else:
                vertices = 0.5 * (vertices[:-1] + vertices[1:])
            setattr(self, direction, GridLine(start=start, end=end,
                                              vertices=vertices))
        self.shape = tuple(line.size for line in [self.z, self.y, self.x]
                           if line.size > 0)
        self.size = functools.reduce(operator.mul, self.shape, 1)
        self.ndim = grid.ndim


class GridFaceY(GridBase):
    """Class for y-face centered grid."""

    def __init__(self, *args, **kwargs):
        """Call the constructor of the base class."""
        super(GridFaceY, self).__init__(*args, **kwargs)

    def create_from_grid(self, grid):
        """Create grid from a base grid."""
        for direction in ['x', 'y', 'z'][:grid.ndim]:
            gridline = getattr(grid, direction)
            start, end = gridline.start, gridline.end
            vertices = gridline.vertices
            if direction is 'y':
                vertices = vertices[1:-1]
            else:
                vertices = 0.5 * (vertices[:-1] + vertices[1:])
            setattr(self, direction, GridLine(start=start, end=end,
                                              vertices=vertices))
        self.shape = tuple(line.size for line in [self.z, self.y, self.x]
                           if line.size > 0)
        self.size = functools.reduce(operator.mul, self.shape, 1)
        self.ndim = grid.ndim


class GridFaceZ(GridBase):
    """Class for z-face centered grid."""

    def __init__(self, *args, **kwargs):
        """Call the constructor of the base class."""
        super(GridFaceZ, self).__init__(*args, **kwargs)

    def create_from_grid(self, grid):
        """Create grid from a base grid."""
        for direction in ['x', 'y', 'z']:
            gridline = getattr(grid, direction)
            start, end = gridline.start, gridline.end
            vertices = gridline.vertices
            if direction is 'z':
                vertices = vertices[1:-1]
            else:
                vertices = 0.5 * (vertices[:-1] + vertices[1:])
            setattr(self, direction, GridLine(start=start, end=end,
                                              vertices=vertices))
        self.shape = tuple(line.size for line in [self.z, self.y, self.x]
                           if line.size > 0)
        self.size = functools.reduce(operator.mul, self.shape, 1)
        self.ndim = grid.ndim


class GridLine():
    """Contain information about a gridline of a structured Cartesian grid."""

    def __init__(self, start=None, end=None, vertices=[], config=None):
        """Initialize the gridline.

        Parameters
        ----------
        config : dictionary (optional)
            Configuration of the gridline to create; default: None.

        """
        self.start, self.end = start, end
        self.vertices = numpy.array(vertices)
        self.size = self.vertices.size
        self.shape = self.vertices.shape
        if config is not None:
            self.create(config)

    def __repr__(self, ndigits=6):
        """Representation of the gridline.

        Parameters
        ----------
        ndigits : integer (optional)
            Number of digits to represent floats; default: 6.

        """
        return ('Gridline(start={}, end={}, size={})'
                .format(self.start, self.end, self.size))

    def create(self, config):
        """Create the gridline.

        Parameters
        ----------
        config : dictionary
            Configuration of the gridline.

        """
        start = config['start']
        segments = []
        if 'segments' not in config.keys():
            self.vertices = Segment(config=config).vertices
        else:
            for node in config['segments']:
                node['start'] = start
                end = node['end']
                segments.append(Segment(config=node).vertices)
                start = end
            config['end'] = end
            self.vertices = numpy.unique(numpy.concatenate(segments))
        self.start, self.end = config['start'], config['end']
        self.size = self.vertices.size
        self.shape = self.vertices.shape

    def get_widths(self):
        """Compute the grid spacing."""
        ghosted = numpy.concatenate(([self.start], self.vertices, [self.end]))
        return 0.5 * (ghosted[1:] - ghosted[:-1])


class Segment():
    """Contain information about a segment of a gridline."""

    def __init__(self, config=None):
        """Initialize the segment.

        Parameters
        ----------
        config : dictionary (optional)
            Configuration of the segment to create; default: None.

        """
        self.start, self.end = None, None
        self.vertices = numpy.array([])
        self.size = self.vertices.size
        self.shape = self.vertices.shape
        self.r = 1.0  # stretching ratio
        if config is not None:
            self.create(config)

    def __repr__(self, ndigits=6):
        """Representation of the segment.

        Parameters
        ----------
        ndigits : integer (optional)
            Number of digits to represent floats; default: 6.

        """
        return ('Segment(start={}, end={}, size={}, r={})'
                .format(round(self.start, ndigits=ndigits),
                        round(self.end, ndigits=ndigits),
                        self.size,
                        round(self.r, ndigits=ndigits)))

    def create(self, config):
        """Create the segment.

        Parameters
        ----------
        config : dictionary
            Configuration of the segment.

        """
        start, end = config['start'], config['end']
        length = abs(end - start)
        if 'num_cells' in config.keys():
            config['width'] = length / config['num_cells']
        width = config['width']
        r = config.get('stretching', 1.0)
        reverse = config.get('reverse', False)

        if abs(r - 1) < 1e-6:  # uniform discretization
            n_float = length / width
            n = int(round(n_float))
            assert abs(n - n_float) < 1e-6, "Length should be multple of width"
            self.vertices = numpy.linspace(start, end, num=n + 1)
        else:  # stretched discretization
            n_float = math.log(1 + length / width * (r - 1)) / math.log(r)
            n = int(round(n_float))
            width = length * (r - 1) / (r**n - 1)  # re-compute first width
            widths = [width * r**k for k in range(n)]
            cumsum = numpy.cumsum(widths)
            if reverse:
                self.vertices = numpy.concatenate(([end], end - cumsum))[::-1]
            else:
                self.vertices = numpy.concatenate(([start], start + cumsum))
        self.start, self.end, self.length = start, end, length
        self.size = self.vertices.size
        self.shape = self.vertices.shape
        self.r, self.reverse = r, reverse

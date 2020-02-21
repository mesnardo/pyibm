"""Helper functions."""

import math
from matplotlib import pyplot
import numpy
from scipy.sparse import coo_matrix

import pyibm


def plot_contourf(field, grid,
                  body=None, body_plot_kwargs={},
                  levels=None,
                  axis_lim=(None, None, None, None),
                  show_grid=False, grid_scatter_kwargs={}):
    """Plot the filled contour of the 2D field."""
    fig, ax = pyplot.subplots(figsize=(8.0, 8.0))
    X, Y = numpy.meshgrid(grid.x.vertices, grid.y.vertices)
    if show_grid:
        xmin, xmax = grid.x.start, grid.x.end
        ymin, ymax = grid.y.start, grid.y.end
        for xi in grid.x.vertices:
            ax.axvline(xi, ymin=ymin, ymax=ymax, color='grey')
        for yi in grid.y.vertices:
            ax.axhline(yi, xmin=xmin, xmax=xmax, color='grey')
        ax.scatter(X, Y, **grid_scatter_kwargs)
    if body is not None:
        ax.plot(body.x, body.y, **body_plot_kwargs)
    if levels is None:
        levels = numpy.linspace(numpy.min(field), numpy.max(field), num=51)
    if field.ndim == 1:
        field = field.reshape(grid.shape)
    contf = ax.contourf(X, Y, field,
                        levels=levels, extend='both', zorder=0)
    fig.colorbar(contf)
    ax.axis('scaled', adjustable='box')
    ax.axis(axis_lim)
    return fig, ax


def get_sub_field(u, grid, box=(-numpy.infty, numpy.infty,
                                -numpy.infty, numpy.infty)):
    """Return the solution of u inside a given box."""
    if u.ndim == 1:
        u = u.reshape(grid.shape)
    x, y = grid.x.vertices, grid.y.vertices
    xlim, ylim = box[:2], box[2:]
    mask_x = numpy.where((x >= xlim[0]) & (x <= xlim[1]))[0]
    mask_y = numpy.where((y >= ylim[0]) & (y <= ylim[1]))[0]
    return (x[mask_x], y[mask_y]), u[mask_y[0]:mask_y[-1] + 1,
                                     mask_x[0]:mask_x[-1] + 1]


def print_stats(u):
    """Print statistics about a given array."""
    print(f'Mean: {numpy.mean(u)}')
    print(f'Min, Max: {numpy.min(u)}, {numpy.max(u)}')
    print(f'L_inf: {numpy.max(numpy.abs(u))}')
    print(f'L_2: {numpy.linalg.norm(u, ord=None)}')


def plot_grids(grid, body=None, Op=None,
               axis_lim=(None, None, None, None)):
    """Plot the 2D grid."""
    pyplot.rc('font', family='serif', size=16)
    fig, ax = pyplot.subplots(figsize=(8.0, 8.0))
    xmin, xmax = grid.x.start, grid.x.end
    ymin, ymax = grid.y.start, grid.y.end
    for xi in grid.x.vertices:
        ax.axvline(xi, ymin=ymin, ymax=ymax, color='grey')
    for yi in grid.y.vertices:
        ax.axhline(yi, xmin=xmin, xmax=xmax, color='grey')
    gridc = pyibm.GridCellCentered(grid=grid)
    X, Y = numpy.meshgrid(gridc.x.vertices, gridc.y.vertices)
    ax.scatter(X, Y, label='cell-centered', marker='o', s=20)
    gridx = pyibm.GridFaceX(grid=grid)
    X, Y = numpy.meshgrid(gridx.x.vertices, gridx.y.vertices)
    ax.scatter(X, Y, label='x-face', marker='x', s=20)
    gridy = pyibm.GridFaceY(grid=grid)
    X, Y = numpy.meshgrid(gridy.x.vertices, gridy.y.vertices)
    ax.scatter(X, Y, label='y-face', marker='x', s=20)
    if body is not None:
        ax.scatter(body.x, body.y, label='body', s=40)
        neighbors = body.get_neighbors(gridc)
        xn, yn = [], []
        for xb, yb, neighbor in zip(body.x, body.y, neighbors):
            i, j, _ = gridc.ijk(neighbor)
            xi, yj = gridc.x.vertices[i], gridc.y.vertices[j]
            xn.append(xi)
            yn.append(yj)
            ax.plot([xb, xi], [yb, yj], color='black')
        ax.scatter(xn, yn, label='neighbors',
                   marker='*', s=40, color='black')
    if Op is not None:
        Op = coo_matrix(Op)
        rows = Op.row
        cols = Op.col
        for row, col in zip(rows, cols):
            if row % body.ndim == 0:
                i, j, _ = gridx.ijk(col)
                xi, yj = gridx.x.vertices[i], gridx.y.vertices[j]
                ax.scatter(xi, yj, marker='s', s=40, color='navy')
    ax.axis('scaled', adjustable='box')
    ax.axis(axis_lim)
    return fig, ax


def taylor_green_vortex(x, y, t, nu, dim=2):
    """Return the solution of the Taylor-Green vortex at given time."""
    if dim == 2:
        X, Y = numpy.meshgrid(x, y)
    else:
        X, Y = x, y
    a = 2 * numpy.pi
    u = -numpy.cos(a * X) * numpy.sin(a * Y) * numpy.exp(-2 * a**2 * nu * t)
    v = +numpy.sin(a * X) * numpy.cos(a * Y) * numpy.exp(-2 * a**2 * nu * t)
    p = (-0.25 * (numpy.cos(2 * a * X) + numpy.cos(2 * a * Y)) *
         numpy.exp(-4 * a**2 * nu * t))
    return u, v, p


def circle(R=0.5, center=(0.0, 0.0), N=None, ds=None):
    """Create the coordinates along a circle using a uniform distribution."""
    if N is None:
        if ds is None:
            raise ValueError('Provide argument N or ds')
        N = math.ceil(2 * math.pi * R / ds)
    theta = numpy.linspace(0.0, 2 * numpy.pi, num=N + 1)[:-1]
    x = center[0] + R * numpy.cos(theta)
    y = center[1] + R * numpy.sin(theta)
    return x, y

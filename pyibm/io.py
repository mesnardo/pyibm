"""Module with I/O functions."""

from matplotlib import pyplot
import os
import pathlib
from scipy.sparse import coo_matrix, csr_matrix
import sys


def load_petscmat_binary(filepath,
                         petsc_dir=os.environ.get('PETSC_DIR', None)):
    """Load a sparse PETSc Mat from binary file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path of the binary file containing the PETSc Mat.
    petsc_dir : str or pathlib.Path, optional
        Local PETSc directory; default is the environment variable `PETSC_DIR`
        or None if not set.
    
    Returns
    -------
    scipy.sparse.csr_matrix
        The sparse matrix in CSR format.

    """
    try:
        pkgdir = pathlib.Path(petsc_dir) / 'lib' / 'petsc' / 'bin'
        if pkgdir not in sys.path:
            sys.path.insert(0, str(pkgdir))
        import PetscBinaryIO
    except:
        raise ImportError('Could not import PetscBinaryIO; provide PETSC_DIR')
    io = PetscBinaryIO.PetscBinaryIO()
    A, = io.readBinaryFile(str(filepath))
    (M, N), (indptr, indices, data) = A
    return csr_matrix((data, indices, indptr), shape=(M, N))


def is_symmetric(A, tol=1e-12):
    """Return True if matrix is symmetric."""
    B = A - A.T
    B = B.multiply(abs(B) > tol)
    return B.nnz == 0


def print_matrix_info(A, name=None):
    """Print information of a given sparse matrix."""
    if name:
        print('Name: ', name)
    print('Type: ', type(A))
    print('Shape: ', A.shape)
    print('Size: ', A.data.size)
    if A.shape[0] == A.shape[1]:
        print('Symmetric: ', is_symmetric(A))
    if A.data.size > 0:
        print('Min/Max: ', A.data.min(), A.data.max())


def plot_matrix(M, figsize=(6.0, 6.0), axis_scaled=True,
                markersize=1, color='red', cmap=None,
                vmin=None, vmax=None,
                limits=[None, None, None, None]):
    """Plot the non-zero elements of a given sparse operator.

    Parameters
    ----------
    A : scipy.sparse matrix
        The sparse matrix to plot.
    figsize : tuple, optional
        Size of the Matplotlib figure.
    axis_scaled : bool, optional
        Scale axes of the graph is True; default is `True`.
    markersize : int, optional
        Size of the marker that represents on non-zero element; default: `1`.
    color : str, optional
        Marker's color to use; default is `'red'`.
    cmap : str, optional
        Colormap used to represent the value of the non-zeros;
        default is `None` (i.e., skip the colormap, use a single color
        for all non-zeros).
    vmin : float, optional
        Minimum color value; default is `None` (minimum of data).
    vmax : float, optional
        Maximum color value; default is `None` (maximum of data).
    limits : list, optional
        Axis limits to use; default is `[None, None, None, None]`.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure.
    matplotlib.axes._subplots.AxesSubplot
        Matplotlib axes.

    """
    # Convert matrix to COO format.
    if not isinstance(M, coo_matrix):
        M = coo_matrix(M)
    # Create Matplotlib figure.
    fig, ax = pyplot.subplots(figsize=figsize)
    if cmap is None:
        # Plot non-zeros elements with single marker color.
        ax.scatter(M.col, M.row, c=color, s=markersize, marker='s')
    else:
        # Otherwise, use colormap to represent elements.
        sc = ax.scatter(M.col, M.row, c=M.data, s=markersize,
                        vmin=vmin, vmax=vmax, cmap=cmap)
        fig.colorbar(sc)
    # Scale axis (if required).
    if axis_scaled:
        ax.axis('scaled', adjustable='box')
    # Set axis limits (if required).
    default_limits = [0, M.shape[0] - 1, 0, M.shape[1] - 1]
    for i, lim in enumerate(limits):
        if lim is None:
            limits[i] = default_limits[i]
    ax.set_xlim(limits[2:])
    ax.set_ylim(limits[:2])
    # Invert y-axis so that first row is at the top of the graph.
    ax.invert_yaxis()
    # Remove axis ticks.
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

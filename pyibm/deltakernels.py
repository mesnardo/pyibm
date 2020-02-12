"""Module with regularized delta kernels."""

import functools
import math
import numba


@numba.njit
def delta_roma_et_al_1999(r, dr):
    """Compute the discrete delta function from Roma et al. (1999).

    Parameters
    ----------
    r : float
        Directional distance between the source and target.
    dr : float
        Grid-spacing of the underlying Cartesian mesh.

    Returns
    -------
    Value of the discrete delta function.

    """
    x = r / dr
    if 0.5 <= x <= 1.5:
        return (5 - 3 * x - math.sqrt(-3 * (1 - x)**2 + 1)) / (6 * dr)
    elif x <= 0.5:
        return (1 + math.sqrt(-3 * x**2 + 1)) / (3 * dr)
    else:
        return 0.0


@numba.njit
def delta_peskin_2002(r, dr):
    """Compute the discrete delta function from Peskin (2002).

    Parameters
    ----------
    r : float
        Directional distance between the source and target.
    dr : float
        Grid-spacing of the underlying Cartesian mesh.

    Returns
    -------
    Value of the discrete delta function.

    """
    x = r / dr
    if 0.0 <= x <= 1.0:
        return (3 - 2 * x + math.sqrt(1 + 4 * x - 4 * x**2)) / (8 * dr)
    elif 1.0 <= x <= 2.0:
        return (5 - 2 * x - math.sqrt(-7 + 12 * x - 4 * x**2)) / (8 * dr)
    else:
        return 0.0


@numba.njit
def delta_stockie_1997(r, dr):
    """Compute the discrete delta function from Stockie (1997).

    Parameters
    ----------
    r : float
        Directional distance between the source and target.
    dr : float
        Grid-spacing of the underlying Cartesian mesh.

    Returns
    -------
    Value of the discrete delta function.

    """
    x = r / dr
    if 0.0 <= x < 1.0:
        return (61 / 112 - 11 / 42 * x - 11 / 56 * x**2 + 1 / 12 * x**3 +
                math.sqrt(3) / 336 *
                math.sqrt(243 + 1584 * x - 748 * x**2 - 1560 * x**3 +
                          500 * x**4 + 336 * x**5 - 112 * x**6)) / dr
    elif 1.0 <= x < 2.0:
        return (21 / 16 + 7 / 12 * x - 7 / 8 * x**2 + 1 / 6 * x**3 -
                3 / 2 * delta_stockie_1997(r - 1, dr)) / dr
    elif 2.0 <= x < 3:
        return (9 / 8 - 23 / 12 * x + 3 / 4 * x**2 - 1 / 12 * x**3 +
                1 / 2 * delta_stockie_1997(r - 2, dr)) / dr
    else:
        return 0.0


@numba.njit
def delta(r, dr, kernel=delta_roma_et_al_1999):
    """Compute the n-dimensional product of the discrete delta function.

    (n is the number of dimensions of the problem.)

    Parameters
    ----------
    r : float or list of floats
        Directional distance between the source and target.
    dr : float or list of floats
        Grid-spacing of the underlying Cartesian mesh.
    kernel : function (optional)
        The regularized delta function to use;
        default: delta_roma_et_al_1999.

    Returns
    -------
    d : float
        Value of the discrete delta function product.

    """
    return kernel(r, dr)

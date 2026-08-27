"""Periodic trilinear interpolation of a uniform-grid field onto arbitrary points.

Generalized out of the PLUTO tracer-file pipeline: on a fixed uniform
Cartesian grid (any code that writes one -- PLUTO, Enzo, a resampled AMR
grid, etc.), trilinear interpolation of the 8 surrounding cell-centers is
the natural analog of AREPO's SPH-kernel neighbor averaging
(:mod:`interpolate_tracers.kernels`) for a moving mesh. This module has no
dependency on any specific simulation code -- it only assumes a 3D array on
a periodic uniform grid, indexed ``field[iz, iy, ix]`` (the axis order
produced by a direct HDF5 read on most grid codes' output), and point
coordinates given as fractions of the box size in ``[0, 1)``.
"""

import numpy as np


def interpolate_field(field, x1, x2, x3, nx=None, ny=None, nz=None):
    """Trilinearly interpolate ``field`` at points (x1, x2, x3), with periodic wraparound.

    Parameters
    ----------
    field : ndarray, shape (nz, ny, nx)
        Grid values, indexed [z, y, x] (the axis order of a direct HDF5 read
        for most uniform-grid simulation output).
    x1, x2, x3 : ndarray, shape (N,)
        Point coordinates as fractions of the box size, in [0, 1). Values
        outside this range wrap around (periodic boundary conditions).
    nx, ny, nz : int, optional
        Grid resolution along each axis. Defaults to field.shape if not given.

    Returns
    -------
    ndarray, shape (N,)
        Interpolated field values at each point.
    """
    nz_f, ny_f, nx_f = field.shape
    nx = nx_f if nx is None else nx
    ny = ny_f if ny is None else ny
    nz = nz_f if nz is None else nz

    dx, dy, dz = 1.0 / nx, 1.0 / ny, 1.0 / nz

    fx = x1 / dx - 0.5
    fy = x2 / dy - 0.5
    fz = x3 / dz - 0.5

    ix0 = np.floor(fx).astype(np.int64)
    iy0 = np.floor(fy).astype(np.int64)
    iz0 = np.floor(fz).astype(np.int64)

    tx = fx - ix0
    ty = fy - iy0
    tz = fz - iz0

    ix0m, ix1m = ix0 % nx, (ix0 + 1) % nx
    iy0m, iy1m = iy0 % ny, (iy0 + 1) % ny
    iz0m, iz1m = iz0 % nz, (iz0 + 1) % nz

    c000 = field[iz0m, iy0m, ix0m]
    c100 = field[iz0m, iy0m, ix1m]
    c010 = field[iz0m, iy1m, ix0m]
    c110 = field[iz0m, iy1m, ix1m]
    c001 = field[iz1m, iy0m, ix0m]
    c101 = field[iz1m, iy0m, ix1m]
    c011 = field[iz1m, iy1m, ix0m]
    c111 = field[iz1m, iy1m, ix1m]

    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    return c0 * (1 - tz) + c1 * tz


def host_cell_index(x1, x2, x3, nx, ny, nz):
    """Flattened index (ix + nx*(iy + ny*iz)) of each point's host grid cell.

    The lower corner of the trilinear-interpolation stencil used by
    :func:`interpolate_field` -- useful as a stand-in for a matched-gas-cell
    ID on a fixed grid (there is no cell-matching step to produce one).
    """
    ix = np.floor(x1 * nx - 0.5).astype(np.int64) % nx
    iy = np.floor(x2 * ny - 0.5).astype(np.int64) % ny
    iz = np.floor(x3 * nz - 0.5).astype(np.int64) % nz
    return ix + nx * (iy + ny * iz)

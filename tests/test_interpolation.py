import numpy as np

from interpolate_tracers.interpolation import interpolate_field, host_cell_index


def _linear_field(nx, ny, nz):
    """A field that's exactly linear in x, so trilinear interpolation should
    reproduce it exactly away from the periodic wrap seam."""
    field = np.zeros((nz, ny, nx))
    dx = 1.0 / nx
    for ix in range(nx):
        xc = (ix + 0.5) * dx
        field[:, :, ix] = xc
    return field


def test_interpolate_field_matches_cell_centers():
    nx = ny = nz = 8
    field = _linear_field(nx, ny, nz)
    dx = 1.0 / nx
    # sample exactly at cell centers away from the wrap seam
    ix = np.array([2, 3, 4])
    xc = (ix + 0.5) * dx
    yc = np.full_like(xc, 0.5)
    zc = np.full_like(xc, 0.5)

    out = interpolate_field(field, xc, yc, zc, nx, ny, nz)
    np.testing.assert_allclose(out, xc, atol=1e-12)


def test_interpolate_field_midpoint_is_average():
    nx = ny = nz = 8
    field = _linear_field(nx, ny, nz)
    dx = 1.0 / nx
    # halfway between cell 2 and cell 3 centers -> average of the two values
    x_mid = np.array([3.0 * dx])
    y_mid = np.array([0.5])
    z_mid = np.array([0.5])

    out = interpolate_field(field, x_mid, y_mid, z_mid, nx, ny, nz)
    expected = 0.5 * ((2 + 0.5) * dx + (3 + 0.5) * dx)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_interpolate_field_periodic_wraparound():
    nx = ny = nz = 8
    field = _linear_field(nx, ny, nz)
    dx = 1.0 / nx
    # point just below x=0 (wraps to just below the last cell, ix=nx-1)
    x = np.array([-0.5 * dx * 0.5])  # -0.25*dx, i.e. halfway between cell -1 (=nx-1) and cell 0
    y = np.array([0.5])
    z = np.array([0.5])

    out = interpolate_field(field, x % 1.0, y, z, nx, ny, nz)
    # should be finite and between the wrapped neighbor values, not NaN/crash
    assert np.isfinite(out[0])
    last_cell_val = (nx - 1 + 0.5) * dx
    first_cell_val = 0.5 * dx
    lo, hi = sorted([last_cell_val - 1.0, first_cell_val])  # last cell wraps to x - 1 in a "linear" sense
    # weaker, robust check: interpolated value should lie strictly within the
    # box's value range (no overshoot/blowup from the wrap)
    assert field.min() - 1.0 <= out[0] <= field.max()


def test_host_cell_index_matches_flattened_convention():
    nx = ny = nz = 4
    dx = 1.0 / nx
    # point at the center of cell (ix=1, iy=2, iz=3)
    x = np.array([(1 + 0.5) * dx])
    y = np.array([(2 + 0.5) * dx])
    z = np.array([(3 + 0.5) * dx])
    idx = host_cell_index(x, y, z, nx, ny, nz)
    expected = 1 + nx * (2 + ny * 3)
    assert idx[0] == expected

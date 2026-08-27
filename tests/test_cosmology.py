import numpy as np
import astropy.units as u

from interpolate_tracers.functions_cosmo import (
    compute_cosmological_times,
    snapshot_time_redshift_map,
)


def test_snapshot_time_redshift_map_anchors_ref_snap():
    initial_snap, final_snap, ref_snap = 0, 10, 5
    z_ref = 0.5
    dt_Gyr = 0.05 * u.Gyr

    snaps, t_Gyr_arr, z_arr = snapshot_time_redshift_map(
        initial_snap, final_snap, ref_snap, z_ref, dt_Gyr
    )
    ref_idx = np.where(snaps == ref_snap)[0][0]
    assert z_arr[ref_idx] == pytest_approx(z_ref)


def pytest_approx(x, rel=1e-3):
    import pytest
    return pytest.approx(x, rel=rel)


def test_snapshot_time_redshift_map_monotonic_in_time():
    # earlier snapshots (further in the past) should have higher redshift
    initial_snap, final_snap, ref_snap = 0, 10, 5
    z_ref = 0.5
    dt_Gyr = 0.05 * u.Gyr

    snaps, t_Gyr_arr, z_arr = snapshot_time_redshift_map(
        initial_snap, final_snap, ref_snap, z_ref, dt_Gyr
    )
    # t_Gyr_arr should be increasing with snap number (later snap = later cosmic time)
    assert np.all(np.diff(t_Gyr_arr) > 0)
    # redshift should be decreasing with snap number (later snap = lower z)
    assert np.all(np.diff(z_arr) < 0)


def test_compute_cosmological_times_basic_sanity():
    time_cosmo, t_first_snap_cosmo, zz_first_snap, dt_Gyr = compute_cosmological_times(
        redshift=0.5, time_between_snapshots=0.02, initial_snap=20, final_snap=95,
    )
    assert time_cosmo.value > 0
    assert t_first_snap_cosmo.value > 0
    assert zz_first_snap > 0.5  # earlier snapshot -> higher redshift than the reference
    assert dt_Gyr.value > 0

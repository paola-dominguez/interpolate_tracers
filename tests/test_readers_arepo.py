import numpy as np
import pytest
from scipy.spatial import cKDTree

from interpolate_tracers.readers.arepo import kernel_average_at_gas_positions, select_mach


def test_kernel_average_at_gas_positions_shapes():
    rng = np.random.default_rng(0)
    n_gas = 30
    gas_coordinates = rng.random((n_gas, 3)) * 10.0
    gas_ids_all = np.arange(n_gas, dtype=np.int64)

    internal_energy_all = rng.random(n_gas)
    bfield_all = rng.random((n_gas, 3))
    density_all = rng.random(n_gas)
    velocities_all = rng.random((n_gas, 3))
    mach_all = rng.random(n_gas)
    divv_all = rng.random(n_gas)
    curlv_all = rng.random(n_gas)
    energy_diss_all = rng.random(n_gas)
    vturb_all = rng.random(n_gas)
    vturb_sol_all = rng.random(n_gas)
    vturb_comp_all = rng.random(n_gas)
    length_all = rng.random(n_gas)

    matched_gas_ids = gas_ids_all[[0, 5, 10]]
    tree_gas_all = cKDTree(gas_coordinates)

    results, gas_indices = kernel_average_at_gas_positions(
        tree_gas_all, gas_coordinates, matched_gas_ids, gas_ids_all,
        internal_energy_all, bfield_all, density_all, velocities_all,
        mach_all, divv_all, curlv_all, energy_diss_all,
        vturb_all, vturb_sol_all, vturb_comp_all, length_all,
        num_neighbors=5, kernel_type="W_M4", radius_cut=100.0,
    )

    assert gas_indices.tolist() == [0, 5, 10]
    n_tracers = len(matched_gas_ids)
    assert results["density"].shape == (n_tracers,)
    assert results["bfield"].shape == (n_tracers, 3)
    assert results["velocity"].shape == (n_tracers, 3)
    assert np.all(np.isfinite(results["density"]))


def test_select_mach_options_and_errors():
    results = {
        "simple_mach": np.array([1.0]),
        "mach": np.array([2.0]),
        "mach_8nb": np.array([3.0]),
        "median_mach": np.array([4.0]),
        "log_avg_mach": np.array([5.0]),
        "clipped_avg_mach": np.array([6.0]),
    }
    assert select_mach(results, "simple")[0] == 1.0
    assert select_mach(results, "kernel")[0] == 2.0
    assert select_mach(results, "nearest", nearest_neighbor_mach=np.array([9.0]))[0] == 9.0

    with pytest.raises(ValueError):
        select_mach(results, "not_a_method")

    with pytest.raises(ValueError):
        select_mach(results, "nearest")  # missing nearest_neighbor_mach

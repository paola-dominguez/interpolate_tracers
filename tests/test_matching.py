import numpy as np
from scipy.spatial import cKDTree

from interpolate_tracers.functions_tracers import find_matched_tracers, find_matched_tracers_nearest


def test_find_matched_tracers_threshold():
    gas_coords = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    tracer_x = np.array([0.05, 10.05, 5.0])
    tracer_y = np.array([0.0, 10.0, 5.0])
    tracer_z = np.array([0.0, 10.0, 5.0])
    tracer_ids = np.array([1, 2, 3])

    tree_gas = cKDTree(gas_coords)
    tree_tracer = cKDTree(np.vstack((tracer_x, tracer_y, tracer_z)).T)

    coincident_indices, mx, my, mz, matched_ids = find_matched_tracers(
        0.2, tree_tracer, tree_gas, tracer_x, tracer_y, tracer_z, tracer_ids
    )
    # tracers 1 and 2 are within 0.2 of a gas cell; tracer 3 (at 5,5,5) is not
    assert set(matched_ids.tolist()) == {1, 2}
    assert len(mx) == 2


def test_find_matched_tracers_nearest_matches_all():
    gas_coords = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    tracer_coords = np.array([[0.1, 0.0, 0.0], [9.9, 10.0, 10.0], [5.0, 5.0, 5.0]])
    tracer_ids = np.array([1, 2, 3])

    tree_gas = cKDTree(gas_coords)
    matched_indices, mx, my, mz, matched_ids = find_matched_tracers_nearest(
        tree_gas, tracer_coords, tracer_ids
    )
    # every tracer is "matched" (nearest neighbor always exists)
    assert len(matched_indices) == 3
    np.testing.assert_array_equal(matched_indices, [0, 1, 0])
    np.testing.assert_array_equal(matched_ids, tracer_ids)

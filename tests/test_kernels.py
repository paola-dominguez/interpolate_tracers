import numpy as np
import pytest

from interpolate_tracers import kernels as kern


@pytest.mark.parametrize("name", ["W_M4", "W_C4", "W_C6", "W_M5", "W_M6"])
def test_kernel_zero_at_support_boundary(name):
    kernel_func = getattr(kern, name)
    support = {"W_M4": 2.0, "W_C4": 2.0, "W_C6": 2.0, "W_M5": 2.5, "W_M6": 3.0}[name]
    q = np.array([support + 0.5, support + 5.0])
    w = kernel_func(q)
    assert np.all(w == 0.0)


@pytest.mark.parametrize("name", ["W_M4", "W_C4", "W_C6"])
def test_kernel_positive_and_decreasing_near_zero(name):
    kernel_func = getattr(kern, name)
    q = np.array([0.0, 0.5, 1.0, 1.5])
    w = kernel_func(q)
    assert w[0] > 0
    assert np.all(np.diff(w) <= 0), f"{name} kernel should be non-increasing with q"


def test_w_m6_nonzero_and_continuous_between_2_and_3():
    # Regression test for a fixed bug: the 2<=q<=3 branch used to be gated
    # by (q>=3)&(q<=3), which only ever matched q==3 exactly and left the
    # whole open interval 2<q<3 at 0 instead of the quintic tail (3-q)**5.
    q_mid = np.array([2.5])
    w_mid = kern.W_M6(q_mid)
    assert w_mid[0] == pytest.approx((3 - 2.5)**5)
    assert w_mid[0] > 0

    # continuous at the q=2 boundary between the 1<=q<=2 and 2<=q<=3 branches
    q_boundary = np.array([2.0])
    w_boundary = kern.W_M6(q_boundary)
    expected = (3 - 2.0)**5 - 6 * (2 - 2.0)**5
    assert w_boundary[0] == pytest.approx(expected)


def test_compute_kernel_averages_weighted_mean():
    # 1 tracer, 2 neighbors, kernel weights [1, 3] -> weighted density mean
    # should be (1*10 + 3*20) / 4 = 17.5
    kernel = np.array([[1.0, 3.0]])
    kernel_8nb = np.array([[1.0, 1.0]])

    n_tr, n_nb, n_nb8 = 1, 2, 2
    neighbor_bfields = np.zeros((n_tr, n_nb, 3))
    neighbor_velocities = np.zeros((n_tr, n_nb, 3))
    neighbor_internal_energy = np.zeros((n_tr, n_nb))
    neighbor_density = np.array([[10.0, 20.0]])
    neighbor_mach = np.array([[2.0, 4.0]])
    neighbor_mach_8nb = np.array([[2.0, 4.0]])
    neighbor_divv = np.zeros((n_tr, n_nb))
    neighbor_curlv = np.zeros((n_tr, n_nb))
    neighbor_energy_diss = np.zeros((n_tr, n_nb))
    neighbor_vturb = np.zeros((n_tr, n_nb))
    neighbor_vturb_sol = np.zeros((n_tr, n_nb))
    neighbor_vturb_comp = np.zeros((n_tr, n_nb))
    neighbor_length = np.zeros((n_tr, n_nb))
    neighbor_coords = np.zeros((n_tr, n_nb, 3))
    neighbor_coords_8nb = np.zeros((n_tr, n_nb8, 3))

    results = kern.compute_kernel_averages(
        kernel, kernel_8nb, neighbor_bfields, neighbor_velocities,
        neighbor_internal_energy, neighbor_density, neighbor_mach, neighbor_mach_8nb,
        neighbor_divv, neighbor_curlv, neighbor_energy_diss,
        neighbor_vturb, neighbor_vturb_sol, neighbor_vturb_comp, neighbor_length,
        neighbor_coords, neighbor_coords_8nb,
    )
    assert results["density"][0] == pytest.approx(17.5)
    # simple (unweighted) mach average: (2+4)/2 = 3
    assert results["simple_mach"][0] == pytest.approx(3.0)


def test_compute_kernel_averages_masked_radius_cut():
    # Same as above but one neighbor sits outside radius_cut -> mach-derived
    # quantities should be zeroed out for that tracer.
    kernel = np.array([[1.0, 1.0]])
    kernel_8nb = np.array([[1.0, 1.0]])
    n_tr, n_nb, n_nb8 = 1, 2, 2

    neighbor_bfields = np.zeros((n_tr, n_nb, 3))
    neighbor_velocities = np.zeros((n_tr, n_nb, 3))
    neighbor_internal_energy = np.zeros((n_tr, n_nb))
    neighbor_density = np.ones((n_tr, n_nb))
    neighbor_mach = np.array([[2.0, 4.0]])
    neighbor_mach_8nb = np.array([[2.0, 4.0]])
    neighbor_divv = np.zeros((n_tr, n_nb))
    neighbor_curlv = np.zeros((n_tr, n_nb))
    neighbor_energy_diss = np.zeros((n_tr, n_nb))
    neighbor_vturb = np.zeros((n_tr, n_nb))
    neighbor_vturb_sol = np.zeros((n_tr, n_nb))
    neighbor_vturb_comp = np.zeros((n_tr, n_nb))
    neighbor_length = np.zeros((n_tr, n_nb))
    # one neighbor at radius 10 -> outside radius_cut=3.5
    neighbor_coords = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    neighbor_coords_8nb = np.zeros((n_tr, n_nb8, 3))

    results = kern.compute_kernel_averages_masked(
        kernel, kernel_8nb, neighbor_bfields, neighbor_velocities,
        neighbor_internal_energy, neighbor_density, neighbor_mach, neighbor_mach_8nb,
        neighbor_divv, neighbor_curlv, neighbor_energy_diss,
        neighbor_vturb, neighbor_vturb_sol, neighbor_vturb_comp, neighbor_length,
        neighbor_coords, neighbor_coords_8nb, radius_cut=3.5,
    )
    assert results["mach"][0] == 0.0
    assert results["simple_mach"][0] == 0.0
    # density isn't masked by the radius cut in this function
    assert results["density"][0] == pytest.approx(1.0)

import numpy as np

from interpolate_tracers.functions_tracers import save_snapshot_to_hdf5, read_tracer_snapshot


def test_tracer_hdf5_round_trip(tmp_path):
    n = 5
    rng = np.random.default_rng(0)

    internal_energy = rng.random(n)
    bfield = rng.random((n, 3))
    density = rng.random(n)
    velocities = rng.random((n, 3))
    mach = rng.random(n)
    divv = rng.random(n)
    curlv = rng.random(n)
    energy_diss = rng.random(n)
    vturb = rng.random(n)
    vturb_sol = rng.random(n)
    vturb_comp = rng.random(n)
    lturb = rng.random(n)
    tracer_ids = np.arange(n, dtype=np.int64)
    gas_ids = np.arange(100, 100 + n, dtype=np.int64)
    coords = rng.random((n, 3))

    path_prefix = str(tmp_path) + "/run"
    save_snapshot_to_hdf5(
        path_prefix, 7,
        internal_energy, bfield, density, velocities,
        mach, divv, curlv, energy_diss,
        vturb, vturb_sol, vturb_comp, lturb,
        tracer_ids, gas_ids, coords,
        velocity_conversion_factor=1e5, density_conversion_factor=1e-24,
        redshift=0.5, t_sim=1.23e17, t_cosmo=4.32e17,
    )

    data = read_tracer_snapshot(path_prefix, 7)

    np.testing.assert_allclose(data["internal_energy"], internal_energy)
    np.testing.assert_allclose(data["bfield"], bfield)
    np.testing.assert_allclose(data["density"], density)
    np.testing.assert_allclose(data["velocities"], velocities)
    np.testing.assert_allclose(data["mach"], mach)
    np.testing.assert_allclose(data["div_v"], divv)
    np.testing.assert_allclose(data["curl_v"], curlv)
    np.testing.assert_allclose(data["energy_diss"], energy_diss)
    np.testing.assert_allclose(data["vturb"], vturb)
    np.testing.assert_allclose(data["vturb_sol"], vturb_sol)
    np.testing.assert_allclose(data["vturb_comp"], vturb_comp)
    np.testing.assert_allclose(data["l_turb"], lturb)
    np.testing.assert_array_equal(data["particleID"], tracer_ids)
    np.testing.assert_array_equal(data["gasID"], gas_ids)
    np.testing.assert_allclose(data["coords"], coords)
    assert data["Redshift"] == 0.5
    assert data["VelocityConversionFactor"] == 1e5
    assert data["DensityConversionFactor"] == 1e-24

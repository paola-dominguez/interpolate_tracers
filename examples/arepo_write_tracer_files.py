"""Step 2 of the AREPO tracer pipeline: kernel-average PartType0 fields onto the
tracer IDs found by arepo_find_tracer_ids.py, and write one tracer HDF5 file
per snapshot.

Adapted from a validated production run (MACS_J0018 merger tracer pipeline,
script_for_writing_tracer_files_nb32_new.py) -- the averaging math (32-neighbor
W_M4 kernel, matched-gas-cell selection via 2-NN query) is unchanged, only the
hardcoded paths/imports and the repeated field-registration/averaging block
are cleaned up (now interpolate_tracers.readers.arepo).

Requires yt (`pip install interpolate_tracers[arepo]`).
"""
import os

import astropy.units as u
import h5py
import numpy as np
import yt
from scipy.spatial import cKDTree

from interpolate_tracers.functions_cosmo import snapshot_time_redshift_map
from interpolate_tracers.functions_tracers import save_snapshot_to_hdf5
from interpolate_tracers.readers.arepo import add_derived_fields, kernel_average_at_gas_positions, select_mach

# ---------------------------- CONFIG (edit me) ----------------------------
path_in = "/path/to/your/simulation/merger_output_030/"
path_out_files = path_in + "Files_tracers/W4_nb32/"
id_run = "my_run_"

kernel_type = "W_M4"        # W_M4/W_C4/W_C6/W_M5/W_M6
num_neighbors = 32
mach_method = "simple"      # simple/kernel/kernel_8nb/median/log/clipped/nearest

initial_snap, final_snap = 20, 95
rad_filter_default = 3.5
wide_radius_snap = 85
rad_filter_wide = 4.5

# Gadget-style unit triplet -- match your own parameter file
UnitLength = 3.08568e+21
UnitMass = 1.989e+43
UnitVelocity = 100000
UnitDensity = UnitMass / UnitLength**3
UnitTime = UnitLength / UnitVelocity

# Redshift anchoring: ref_snap's real known redshift (z_ref), every other
# snapshot's redshift is mapped from it via cosmic time
ref_snap = 59
z_ref = 0.5456
TimeBetSnapshot = 0.020441  # code time units, from the parameter file
# ----------------------------------------------------------------------------

os.makedirs(path_out_files, exist_ok=True)

with h5py.File(path_out_files + f"matched_tracer_ids_all_{id_run}.h5", "r") as hdf:
    matched_tracer_ids_snap = np.sort(hdf["matched_tracer_ids"][:])

dt_Gyr = (TimeBetSnapshot * UnitTime / 3.154e16) * u.Gyr
snaps, t_Gyr_arr, z_arr = snapshot_time_redshift_map(initial_snap, final_snap, ref_snap, z_ref, dt_Gyr)

for i in range(final_snap, initial_snap, -1):
    snap = f"snapshot_{i:03d}"
    print("snapshot -->", snap)
    ds = yt.load(path_in + snap + ".hdf5")
    add_derived_fields(ds)

    t_sim = ds.current_time.in_units('s').value
    jj = np.where(snaps == i)[0][0]
    redshift = z_arr[jj]
    print(f"  t_cosmo={t_Gyr_arr[jj]:.3f} Gyr  z={redshift:.4f}")

    v_pot, c_pot = ds.find_min(('nbody', 'Potential'))
    center_x, center_y, center_z = c_pot

    ad = ds.all_data()
    tracer_ids_all = np.asarray(ad[('PartType2', 'ParticleIDs')]).astype(np.int64)
    tracer_coords_all = ad[('PartType2', 'Coordinates')]

    internal_energy_all = ad[('PartType0', 'InternalEnergy')].in_units('cm**2/s**2')
    bfield_all = ad[('PartType0', 'MagneticField')].in_units('G')
    density_all = ad[('PartType0', 'Density')].in_units('g/cm**3')
    velocities_all = ad[('PartType0', 'Velocities')].in_units('cm/s')
    mach_all = ad[('PartType0', 'Machnumber')]
    divv_all = ad[('PartType0', 'velocitydivergence')].in_units('1/s')
    curlv_all = ad[('PartType0', 'velocitycurl')].in_units('1/s')
    energy_diss_all = ad[('PartType0', 'energy_dissipation')].in_units('erg/s')
    vturb_all = ad[('PartType0', 'velocityturb')].in_units('cm/s')
    vturb_sol_all = ad[('PartType0', 'velocitysolenoidal')].in_units('cm/s')
    vturb_comp_all = ad[('PartType0', 'velocitycompressive')].in_units('cm/s')
    length_all = ad[('PartType0', 'filteringlength')].in_units('cm')

    # -- select matched tracers present in this snapshot
    id_to_index = {pid: idx for idx, pid in enumerate(tracer_ids_all)}
    indices = [id_to_index[pid] for pid in matched_tracer_ids_snap if pid in id_to_index]
    selected_ids = tracer_ids_all[indices]
    matched_coords = tracer_coords_all[indices]

    gas_x = (ad[('PartType0', 'Coordinates')][:, 0] - center_x).in_units('Mpc')
    gas_y = (ad[('PartType0', 'Coordinates')][:, 1] - center_y).in_units('Mpc')
    gas_z = (ad[('PartType0', 'Coordinates')][:, 2] - center_z).in_units('Mpc')
    gas_ids_all = ad[('PartType0', 'ParticleIDs')]
    gas_coordinates = np.vstack((gas_x, gas_y, gas_z)).T

    tree_gas_all = cKDTree(gas_coordinates, leafsize=100)

    mapped_coordinates = np.vstack((
        (matched_coords[:, 0] - center_x).in_units('Mpc'),
        (matched_coords[:, 1] - center_y).in_units('Mpc'),
        (matched_coords[:, 2] - center_z).in_units('Mpc'),
    )).T
    # nearest gas cell to each matched tracer (k=2: self + 1 nearest, since
    # tracer and gas trees are built from different point sets here k=2 isn't
    # needed, but kept for parity with the validated run this was adapted from)
    _, coincident_indices = tree_gas_all.query(mapped_coordinates, k=2)
    matched_gas_ids = np.array([gas_ids_all[idx[0]] for idx in coincident_indices]).astype(np.int64)

    rad_filter = rad_filter_wide if i >= wide_radius_snap else rad_filter_default

    results, gas_indices = kernel_average_at_gas_positions(
        tree_gas_all, gas_coordinates, matched_gas_ids, gas_ids_all,
        internal_energy_all, bfield_all, density_all, velocities_all,
        mach_all, divv_all, curlv_all, energy_diss_all,
        vturb_all, vturb_sol_all, vturb_comp_all, length_all,
        num_neighbors=num_neighbors, kernel_type=kernel_type, radius_cut=rad_filter,
    )

    nearest_mach = mach_all[gas_indices] if mach_method == "nearest" else None
    mach = select_mach(results, mach_method, nearest_neighbor_mach=nearest_mach)
    energy_diss = energy_diss_all[gas_indices] if mach_method == "nearest" else results["energy_diss"]

    save_snapshot_to_hdf5(
        path_out_files, i,
        results["internal_energy"], results["bfield"], results["density"], results["velocity"],
        mach, results["divv"], results["curlv"], energy_diss,
        results["vturb"], results["vturb_sol"], results["vturb_comp"], results["length"],
        selected_ids, matched_gas_ids, matched_coords,
        UnitVelocity, UnitDensity, redshift,
        t_sim=t_sim, t_cosmo=(t_Gyr_arr[jj] * u.Gyr).to_value(u.s),
    )

print("Done!")

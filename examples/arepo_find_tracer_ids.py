"""Step 1 of the AREPO tracer pipeline: find which tracer (PartType2) particles
sit near gas cells matching some physical condition (e.g. "in a shock"), across
a range of snapshots, and write out the matched tracer IDs.

Adapted from a validated production run (MACS_J0018 merger tracer pipeline,
script_for_finding_ids_query_ball.py) -- the matching logic and defaults are
unchanged, only the hardcoded paths/imports are cleaned up. Edit the CONFIG
block below for your own run, then see arepo_write_tracer_files.py for step 2
(kernel-averaging fields onto the matched tracers and writing tracer HDF5
files).

Requires yt (`pip install interpolate_tracers[arepo]`).
"""
import os

import h5py
import numpy as np
import yt
from scipy.spatial import cKDTree

from interpolate_tracers.functions_yt import _filtered_mach_radius, _filtered_mach_radius2
from interpolate_tracers.functions_tracers import find_matched_tracers

# ---------------------------- CONFIG (edit me) ----------------------------
path_in = "/path/to/your/simulation/merger_output_030/"
path_out_files = path_in + "Files_tracers/"

id_run = "my_run_"
initial_snap, final_snap = 20, 95   # inclusive range walked snap-by-snap, descending
rad_filter_default = 3.5            # Mpc; snapshots >= wide_radius_snap use rad_filter_wide instead
wide_radius_snap = 85
rad_filter_wide = 4.5
# ----------------------------------------------------------------------------

os.makedirs(path_out_files, exist_ok=True)

for i in range(final_snap, initial_snap, -1):
    snap = f"snapshot_{i:03d}"
    print("-->", snap)
    ds = yt.load(path_in + snap + ".hdf5")

    ds.add_field(("PartType0", "filtered_mach_radius"), function=_filtered_mach_radius,
                 units="", sampling_type="local", force_override=True)
    ds.add_field(("PartType0", "filtered_mach_radius2"), function=_filtered_mach_radius2,
                 units="", sampling_type="local", force_override=True)

    v_pot, c_pot = ds.find_min(('nbody', 'Potential'))
    center_x, center_y, center_z = c_pot

    ad = ds.all_data()
    mach_filtered = ad[('PartType0', 'filtered_mach_radius2' if i >= wide_radius_snap else 'filtered_mach_radius')]
    rad_filter = rad_filter_wide if i >= wide_radius_snap else rad_filter_default

    gas_x = (ad[('PartType0', 'Coordinates')][:, 0] - center_x).in_units('Mpc')
    gas_y = (ad[('PartType0', 'Coordinates')][:, 1] - center_y).in_units('Mpc')
    gas_z = (ad[('PartType0', 'Coordinates')][:, 2] - center_z).in_units('Mpc')
    radius = np.sqrt(gas_x**2 + gas_y**2 + gas_z**2)

    # gas cells satisfying the physical condition (here: within rad_filter AND
    # flagged by the Mach-number field) -- edit this mask for your own selection
    mask = (mach_filtered > 0) & (radius < rad_filter)
    masked_gas_coordinates = np.vstack((gas_x[mask], gas_y[mask], gas_z[mask])).T

    tracer_x = (ad[('PartType2', 'Coordinates')][:, 0] - center_x).in_units('Mpc')
    tracer_y = (ad[('PartType2', 'Coordinates')][:, 1] - center_y).in_units('Mpc')
    tracer_z = (ad[('PartType2', 'Coordinates')][:, 2] - center_z).in_units('Mpc')
    tracer_ids = ad[('PartType2', 'ParticleIDs')]
    tracer_coordinates = np.vstack((tracer_x, tracer_y, tracer_z)).T

    gas_coordinates = np.vstack((gas_x, gas_y, gas_z)).T
    tree_gas_all = cKDTree(gas_coordinates)
    distances, indices = tree_gas_all.query(gas_coordinates, k=2)
    average_nearest_neighbor_distance = np.mean(distances[:, 1])
    threshold = average_nearest_neighbor_distance / 2  # selected by hand; see README for rationale

    tree_masked_gas = cKDTree(masked_gas_coordinates)
    tree_tracer = cKDTree(tracer_coordinates)

    _, mx, my, mz, matched_tracer_ids = find_matched_tracers(
        threshold, tree_tracer, tree_masked_gas, tracer_x, tracer_y, tracer_z, tracer_ids
    )
    print(f"  matched {len(matched_tracer_ids)} tracers (threshold={threshold:.4g} Mpc)")

    filename = f"matched_tracer_ids_{id_run}snap{i:02d}.h5"
    with h5py.File(path_out_files + filename, "w") as hdf:
        hdf.create_dataset("matched_tracer_ids", data=np.sort(matched_tracer_ids))
        hdf.create_dataset("thresholds", data=threshold)

print("\nFINISHED WRITING PER-SNAPSHOT MATCHED-ID FILES!")

# --- combine every snapshot's matched IDs into one deduplicated global list ---
all_ids = []
for i in range(initial_snap + 1, final_snap):
    filename = path_out_files + f"matched_tracer_ids_{id_run}snap{i:02d}.h5"
    with h5py.File(filename, "r") as f:
        all_ids.append(f["matched_tracer_ids"][:])

unique_ids = np.unique(np.concatenate(all_ids))
print(f"Total unique tracer IDs across all snapshots: {len(unique_ids)}")

output_filename = path_out_files + f"matched_tracer_ids_all_{id_run}.h5"
with h5py.File(output_filename, "w") as hdf:
    hdf.create_dataset("matched_tracer_ids", data=unique_ids)
print("Written to:", output_filename)

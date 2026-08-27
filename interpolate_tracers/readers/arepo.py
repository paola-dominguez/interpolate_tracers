"""AREPO reader: moving-mesh tracer pipeline via yt + SPH-kernel neighbor averaging.

Factors out the block that was duplicated, near-verbatim, across every
AREPO tracer-writing script in this project (register derived fields on a
yt dataset -> KD-tree-query neighbors around each matched gas cell ->
kernel-weight and average). The underlying math is unchanged from the
validated production runs this was pulled from (e.g. the MACS_J0018 merger
tracer pipeline) -- this module only removes the copy-pasting, it doesn't
change any numbers.

AREPO tracers (PartType2) don't sit on a fixed grid the way PLUTO's do, so
"which gas cell is this tracer near" is a KD-tree nearest-neighbor search
(see :func:`interpolate_tracers.functions_tracers.find_matched_tracers` /
`find_matched_tracers_nearest`), and field values are assigned via
SPH-kernel-weighted averaging over a fixed number of neighbors
(:mod:`interpolate_tracers.kernels`), not grid interpolation.
"""

import numpy as np
from scipy.spatial import cKDTree

from .. import functions_yt as func_yt
from .. import kernels as kern

# Kernel smoothing-length divisor: h_smoothing = h_max_neighbor_distance / divisor.
# Matches the values used in the validated MACS_J0018 tracer pipeline.
KERNEL_SMOOTHING_DIVISOR = {
    "W_M4": 2, "W_C4": 2, "W_C6": 2, "W_M5": 5, "W_M6": 3,
}

# (yt field name, function in functions_yt, output units) registered on the
# dataset by add_derived_fields(). Matches every AREPO tracer script in this
# project field-for-field.
DERIVED_FIELDS = (
    ("velocitydivergence", func_yt._velocity_divergence, "1/s"),
    ("velocitycurl", func_yt._velocity_curl, "1/s"),
    ("energy_dissipation", func_yt._energy_dissipation, "erg/s"),
    ("velocityturb", func_yt._velocity_turb, "cm/s"),
    ("velocitysolenoidal", func_yt._velocity_solenoidal, "cm/s"),
    ("velocitycompressive", func_yt._velocity_compressive, "cm/s"),
    ("filteringlength", func_yt._filtering_length, "cm"),
)


def add_derived_fields(ds, fields=DERIVED_FIELDS):
    """Register the standard set of PartType0 derived fields on a yt dataset.

    Replaces the repeated ds.add_field(...) block at the top of every AREPO
    tracer script in this project.
    """
    for name, func, units in fields:
        ds.add_field(("PartType0", name), function=func, units=units,
                     sampling_type="local", force_override=True)
    return ds


def kernel_average_at_gas_positions(
    tree_gas_all, gas_coordinates, matched_gas_ids, gas_ids_all,
    internal_energy_all, bfield_all, density_all, velocities_all,
    mach_all, divv_all, curlv_all, energy_diss_all,
    vturb_all, vturb_sol_all, vturb_comp_all, length_all,
    num_neighbors=15, kernel_type="W_M4", radius_cut=3.5,
    masked=True, debug_mach_selection="no",
):
    """SPH-kernel-average PartType0 fields around each matched gas cell.

    Parameters
    ----------
    tree_gas_all : scipy.spatial.cKDTree
        KD-tree over all PartType0 gas positions (gas_coordinates).
    matched_gas_ids : array
        IDs of the gas cells to average around (one per tracer -- typically
        the nearest gas cell to each matched tracer, from
        functions_tracers.find_matched_tracers/find_matched_tracers_nearest).
    gas_ids_all, *_all : array
        Full PartType0 arrays for every field, indexed consistently with
        gas_coordinates (i.e. gas_ids_all[k] is the ID of gas_coordinates[k]).
    num_neighbors : int
        Neighbors for the main kernel average (15 or 32 have both been used
        in validated production runs -- more neighbors = smoother average).
    kernel_type : str
        One of "W_M4", "W_C4", "W_C6", "W_M5", "W_M6" (see kernels.py).
    masked : bool
        Use kernels.compute_kernel_averages_masked (masks Mach-derived
        quantities to radius_cut -- the convention for shock statistics) if
        True, else the unmasked kernels.compute_kernel_averages.

    Returns
    -------
    results : dict
        Same keys as kernels.compute_kernel_averages[_masked] returns
        (bfield, velocity, internal_energy, density, mach, divv, curlv,
        energy_diss, vturb, vturb_sol, vturb_comp, length, ...).
    gas_indices : ndarray
        Index into gas_coordinates/*_all for each matched_gas_ids entry
        (i.e. the row each tracer's averaging was centered on).
    """
    gas_ids_all = np.asarray(gas_ids_all).astype(np.int64)
    matched_gas_ids = np.asarray(matched_gas_ids).astype(np.int64)
    index_map = {gid: idx for idx, gid in enumerate(gas_ids_all)}
    gas_indices = np.array([index_map[gid] for gid in matched_gas_ids])

    query_input = gas_coordinates[gas_indices]
    distances, neighbor_indices = tree_gas_all.query(query_input, k=num_neighbors)
    distances_mach, neighbor_indices_mach = tree_gas_all.query(query_input, k=8)

    h_main = np.max(distances, axis=1)
    h_8 = np.max(distances_mach, axis=1)
    divisor = KERNEL_SMOOTHING_DIVISOR[kernel_type]
    kernel_func = getattr(kern, kernel_type)

    q = distances / (h_main[:, None] / divisor)
    kernel = kernel_func(q)
    q8 = distances_mach / (h_8[:, None] / divisor)
    kernel_8nb = kernel_func(q8)

    neighbor_coords = gas_coordinates[neighbor_indices]
    neighbor_coords_8nb = gas_coordinates[neighbor_indices_mach]

    avg_fn = kern.compute_kernel_averages_masked if masked else kern.compute_kernel_averages
    results = avg_fn(
        kernel, kernel_8nb,
        bfield_all[neighbor_indices], velocities_all[neighbor_indices],
        internal_energy_all[neighbor_indices], density_all[neighbor_indices],
        mach_all[neighbor_indices], mach_all[neighbor_indices_mach],
        divv_all[neighbor_indices], curlv_all[neighbor_indices], energy_diss_all[neighbor_indices],
        vturb_all[neighbor_indices], vturb_sol_all[neighbor_indices], vturb_comp_all[neighbor_indices],
        length_all[neighbor_indices],
        neighbor_coords, neighbor_coords_8nb,
        radius_cut=radius_cut, debug_mach_selection=debug_mach_selection,
    )
    return results, gas_indices


def select_mach(results, method, nearest_neighbor_mach=None):
    """Pick which Mach-averaging convention to use from a kernel_average_at_gas_positions result.

    method: one of "simple", "kernel", "kernel_8nb", "median", "log",
    "clipped", or "nearest" (requires nearest_neighbor_mach, the mach value
    at the single nearest gas cell -- not kernel-averaged at all).
    """
    if method == "nearest":
        if nearest_neighbor_mach is None:
            raise ValueError("method='nearest' requires nearest_neighbor_mach")
        return nearest_neighbor_mach
    options = {
        "simple": results["simple_mach"],
        "kernel": results["mach"],
        "kernel_8nb": results["mach_8nb"],
        "median": results["median_mach"],
        "log": results["log_avg_mach"],
        "clipped": results["clipped_avg_mach"],
    }
    if method not in options:
        raise ValueError(f"Unknown Mach method '{method}'. Choose from: {list(options.keys())+['nearest']}")
    return options[method]

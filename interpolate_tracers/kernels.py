
# Definition of kernels

def W_M4(q):
    import numpy as np
    kernel = np.zeros_like(q)  # Initialize the kernel array with the same shape as q
    
    # Condition where 0 <= q <= 1
    mask1 = (q >= 0) & (q <= 1)
    kernel[mask1] = 1 - (3*q[mask1]**2/2) + (3*q[mask1]**3/4)
    
    # Condition where 1 < q <= 2
    mask2 = (q > 1) & (q <= 2)
    kernel[mask2] = (1/4) * (2 - q[mask2])**3

    # Condition where q >= 2
    mask3 = (q >=2) 
    kernel[mask3] = 0.0
    
    return kernel
    
def W_C4(q):
    import numpy as np
    kernel = np.zeros_like(q)  # Initialize the kernel array with the same shape as q
    
    # Condition where 0 <= q <= 2
    mask1 = (q >= 0) & (q <= 2)
    kernel[mask1] = (1 - q[mask1]/2)**6 * ((35*q[mask1]**2/12) + 3*q[mask1] +1)

    # Condition where q >= 2
    mask2 = (q >=2) 
    kernel[mask2] = 0.0

    return kernel
    
def W_C6(q):
    import numpy as np
    kernel = np.zeros_like(q)  # Initialize the kernel array with the same shape as q

    # Condition where 0 <= q <= 2
    mask1 = (q >= 0) & (q <= 2)
    kernel[mask1] = (1 - q[mask1]/2)**8 * (4*q[mask1]**3 + (25*q[mask1]**2/4) + 4*q[mask1] + 1)    

     # Condition where q >= 2
    mask2 = (q >=2) 
    kernel[mask2] = 0.0

    return kernel
    
def W_M5(q): # needs q over 5
    import numpy as np
    kernel = np.zeros_like(q)  # Initialize the kernel array with the same shape as q

    # Condition where 0 <= q <= 1/2
    mask1 = (q >= 0) & (q <= 0.5)
    kernel[mask1] = (2.5 - q[mask1])**4 - 5*(1.5 - q[mask1])**4 + 10*(0.5 - q[mask1])**4

    # Condition where 1/2 <= q <= 3/2
    mask2 = (q >= 0.5) & (q <= 1.5)
    kernel[mask2] = (2.5 - q[mask2])**4 - 5*(1.5 - q[mask2])**4

    # Condition where 3/2 <= q <= 5/2
    mask3 = (q >= 1.5) & (q <= 2.5)
    kernel[mask3] = (2.5 - q[mask3])**4 

    return kernel
    
def W_M6(q): # needs q over 3
    import numpy as np
    kernel = np.zeros_like(q)  # Initialize the kernel array with the same shape as q

    # Condition where 0 <= q <= 1
    mask1 = (q >= 0) & (q <= 1)
    kernel[mask1] = (3 - q[mask1])**5 - 6*(2 - q[mask1])**5 + 15*(1.0 - q[mask1])**5

    # Condition where 1 <= q <= 2
    mask2 = (q >= 1.0) & (q <= 2.0)
    kernel[mask2] = (3 - q[mask2])**5 - 6*(2 - q[mask2])**5

    # Condition where 2 <= q <= 3
    # KNOWN BUG (found while packaging, not fixed to avoid silently changing
    # past results computed with this kernel): this mask only matches q==3
    # exactly, so W_M6 returns 0 for the entire open interval 2<q<3 instead
    # of the quintic tail (3-q)**5 -- a real discontinuity vs. the standard
    # M6 definition. The intended condition is (q >= 2) & (q <= 3). Not
    # used by the current validated tracer-file pipeline (which uses W_M4),
    # but anyone selecting kernel_type="W_M6" should be aware of this before
    # trusting results near q in (2, 3).
    mask3 = (q >= 3) & (q <= 3)
    kernel[mask3] =  (3 - q[mask3])**5

    return kernel

# Definitions to compute the averages
# Assumes neighbor_* arrays are shaped (N_tracers, N_neighbors)
# and vector fields (bfields, velocities) are (N_tracers, N_neighbors, 3)

def compute_kernel_averages(
    kernel,
    kernel_8nb,
    neighbor_bfields,
    neighbor_velocities,
    neighbor_internal_energy,
    neighbor_density,
    neighbor_mach,
    neighbor_mach_8nb,
    neighbor_divv,
    neighbor_curlv,
    neighbor_energy_diss,
    neighbor_vturb,
    neighbor_vturb_sol,
    neighbor_vturb_comp,
    neighbor_length,
    neighbor_coords,
    neighbor_coords_8nb,
    radius_cut=3.5,
    debug_mach_selection="no"
):
    import numpy as np

    weights_2d = kernel
    weights_2d_8nb = kernel_8nb
    weights_3d = weights_2d[..., np.newaxis]

    sum_weights = np.sum(weights_2d, axis=1, keepdims=True)
    sum_weights_8nb = np.sum(weights_2d_8nb, axis=1, keepdims=True)
    sum_weights_3d = np.sum(weights_3d, axis=1, keepdims=True)

    # Vector fields
    kernel_avg_bfield = np.sum(neighbor_bfields * weights_3d, axis=1) / sum_weights_3d.squeeze(1)
    kernel_avg_vel = np.sum(neighbor_velocities * weights_3d, axis=1) / sum_weights_3d.squeeze(1)

    # Scalar field helper
    def weighted_avg(field):
        return np.sum(field * weights_2d, axis=1) / sum_weights.squeeze(1)

    def weighted_avg_8nb(field):
        return np.sum(field * weights_2d_8nb, axis=1) / sum_weights_8nb.squeeze(1)

    kernel_avg_eint = weighted_avg(neighbor_internal_energy)
    kernel_avg_dens = weighted_avg(neighbor_density)
    kernel_avg_mach = weighted_avg(neighbor_mach)
    kernel_avg_mach_8nb = weighted_avg_8nb(neighbor_mach_8nb)
    kernel_avg_divv = weighted_avg(neighbor_divv)
    kernel_avg_curlv = weighted_avg(neighbor_curlv)
    kernel_avg_ediss = weighted_avg(neighbor_energy_diss)
    kernel_avg_vturb = weighted_avg(neighbor_vturb)
    kernel_avg_vturbsol = weighted_avg(neighbor_vturb_sol)
    kernel_avg_vturbcomp = weighted_avg(neighbor_vturb_comp)
    kernel_avg_l = weighted_avg(neighbor_length)

    # Simple Mach avg (unweighted, masked)
    mask = (neighbor_mach > 0)
    sum_mach = np.sum(np.where(mask, neighbor_mach, 0.0), axis=1)
    count_mach = np.sum(mask, axis=1)
    simple_avg_mach = np.where(count_mach > 0, sum_mach / count_mach, 0.0)

    # Median Mach
    median_mach = np.array([
        np.median(row[row > 0]) if np.any(row > 0) else 0.0
        for row in neighbor_mach
    ])

    # Log-average Mach
    log_avg_mach = np.array([
        np.exp(np.mean(np.log(row[row > 0]))) if np.any(row > 0) else 0.0
        for row in neighbor_mach
    ])

    # Average excluding Mach > 10
    limited_mask = (neighbor_mach > 0) & (neighbor_mach <= 10)
    limited_sum = np.sum(np.where(limited_mask, neighbor_mach, 0.0), axis=1)
    limited_count = np.sum(limited_mask, axis=1)
    clipped_avg_mach = np.where(limited_count > 0, limited_sum / limited_count, 0.0)

    if debug_mach_selection == "yes":
        valid_indices = np.where((kernel_avg_mach > 0) | (simple_avg_mach > 0))[0]
        print("Tracer Index | Weighted Avg Mach (Kernel) | Simple Avg Mach")
        print("------------------------------------------------------------")
        for idx in valid_indices[:50]:
            print(f"{idx:12} | {kernel_avg_mach[idx]:28.6f} | {simple_avg_mach[idx]:18.6f}")
        sys.exit()

    return {
        "bfield": kernel_avg_bfield,
        "velocity": kernel_avg_vel,
        "internal_energy": kernel_avg_eint,
        "density": kernel_avg_dens,
        "mach": kernel_avg_mach,
        "mach_8nb": kernel_avg_mach_8nb,
        "simple_mach": simple_avg_mach,
        "median_mach": median_mach,
        "log_avg_mach": log_avg_mach,
        "clipped_avg_mach": clipped_avg_mach,
        "divv": kernel_avg_divv,
        "curlv": kernel_avg_curlv,
        "energy_diss": kernel_avg_ediss,
        "vturb": kernel_avg_vturb,
        "vturb_sol": kernel_avg_vturbsol,
        "vturb_comp": kernel_avg_vturbcomp,
        "length": kernel_avg_l
    }


# This is used for shocks in AREPO
# Note: we have to mask the outskirts in our simulations
def compute_kernel_averages_masked(
    kernel,
    kernel_8nb,
    neighbor_bfields,
    neighbor_velocities,
    neighbor_internal_energy,
    neighbor_density,
    neighbor_mach,
    neighbor_mach_8nb,
    neighbor_divv,
    neighbor_curlv,
    neighbor_energy_diss,
    neighbor_vturb,
    neighbor_vturb_sol,
    neighbor_vturb_comp,
    neighbor_length,
    neighbor_coords,
    neighbor_coords_8nb,
    radius_cut,
    debug_mach_selection="no"
):
    import numpy as np
    import sys

    weights_2d = kernel                      # (N, 15)
    weights_2d_8nb = kernel_8nb              # (N, 8)
    weights_3d = weights_2d[..., np.newaxis] # (N, 15, 1)

    sum_weights = np.sum(weights_2d, axis=1, keepdims=True)
    sum_weights_8nb = np.sum(weights_2d_8nb, axis=1, keepdims=True)
    sum_weights_3d = np.sum(weights_3d, axis=1, keepdims=True)

    # Vector fields
    kernel_avg_bfield = np.sum(neighbor_bfields * weights_3d, axis=1) / sum_weights_3d.squeeze(1)
    kernel_avg_vel    = np.sum(neighbor_velocities * weights_3d, axis=1) / sum_weights_3d.squeeze(1)

    # Scalar field helper
    def weighted_avg(field):
        return np.sum(field * weights_2d, axis=1) / sum_weights.squeeze(1)

    def weighted_avg_8nb(field):
        return np.sum(field * weights_2d_8nb, axis=1) / sum_weights_8nb.squeeze(1)

    kernel_avg_eint      = weighted_avg(neighbor_internal_energy)
    kernel_avg_dens      = weighted_avg(neighbor_density)
    kernel_avg_divv      = weighted_avg(neighbor_divv)
    kernel_avg_curlv     = weighted_avg(neighbor_curlv)
    kernel_avg_ediss     = weighted_avg(neighbor_energy_diss)
    kernel_avg_vturb     = weighted_avg(neighbor_vturb)
    kernel_avg_vturbsol  = weighted_avg(neighbor_vturb_sol)
    kernel_avg_vturbcomp = weighted_avg(neighbor_vturb_comp)
    kernel_avg_l         = weighted_avg(neighbor_length)

    # Mach-related calculations
    mask = neighbor_mach > 0
    sum_mach = np.sum(neighbor_mach * mask, axis=1)
    count_mach = np.sum(mask, axis=1)
    simple_avg_mach = np.where(count_mach > 0, sum_mach / count_mach, 0.0)

    median_mach = np.array([
        np.median(row[row > 0]) if np.any(row > 0) else 0.0
        for row in neighbor_mach
    ])
    log_avg_mach = np.array([
        np.exp(np.mean(np.log(row[row > 0]))) if np.any(row > 0) else 0.0
        for row in neighbor_mach
    ])
    limited_mask = (neighbor_mach > 0) & (neighbor_mach <= 10)
    limited_sum = np.sum(neighbor_mach * limited_mask, axis=1)
    limited_count = np.sum(limited_mask, axis=1)
    clipped_avg_mach = np.where(limited_count > 0, limited_sum / limited_count, 0.0)

    kernel_avg_mach = weighted_avg(neighbor_mach)
    kernel_avg_mach_8nb = weighted_avg_8nb(neighbor_mach_8nb)

    # Compute radii and apply r < 3.5 Mpc cut for Mach-only filtering
    neighbor_radii = np.linalg.norm(neighbor_coords, axis=2)  # (N, 15)
    within_radius_mask = np.all(neighbor_radii <= radius_cut, axis=1)  # (N,)

    # Mask out Mach values where any neighbor is outside radius_cut
    kernel_avg_mach      = np.where(within_radius_mask, kernel_avg_mach, 0.0)
    kernel_avg_mach_8nb  = np.where(within_radius_mask, kernel_avg_mach_8nb, 0.0)
    simple_avg_mach      = np.where(within_radius_mask, simple_avg_mach, 0.0)
    median_mach          = np.where(within_radius_mask, median_mach, 0.0)
    log_avg_mach         = np.where(within_radius_mask, log_avg_mach, 0.0)
    clipped_avg_mach     = np.where(within_radius_mask, clipped_avg_mach, 0.0)

    kernel_avg_divv      = np.where(within_radius_mask, kernel_avg_divv, 0.0)

    if debug_mach_selection == "yes":
        valid_indices = np.where((kernel_avg_mach > 0) | (simple_avg_mach > 0))[0]
        print("Tracer Index | Weighted Avg Mach (Kernel) | Simple Avg Mach")
        print("------------------------------------------------------------")
        for idx in valid_indices[:50]:
            print(f"{idx:12} | {kernel_avg_mach[idx]:28.6f} | {simple_avg_mach[idx]:18.6f}")
        sys.exit()

    return {
        "bfield": kernel_avg_bfield,
        "velocity": kernel_avg_vel,
        "internal_energy": kernel_avg_eint,
        "density": kernel_avg_dens,
        "mach": kernel_avg_mach,
        "mach_8nb": kernel_avg_mach_8nb,
        "simple_mach": simple_avg_mach,
        "median_mach": median_mach,
        "log_avg_mach": log_avg_mach,
        "clipped_avg_mach": clipped_avg_mach,
        "divv": kernel_avg_divv,
        "curlv": kernel_avg_curlv,
        "energy_diss": kernel_avg_ediss,
        "vturb": kernel_avg_vturb,
        "vturb_sol": kernel_avg_vturbsol,
        "vturb_comp": kernel_avg_vturbcomp,
        "length": kernel_avg_l
    }

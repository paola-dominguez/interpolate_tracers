# IMPORTANT: The filtered radius should be defined here accordingly!

def _filtered_masses(field, data):
    import numpy as np
    center = data.ds.arr(data.ds.domain_center, 'code_length')
    #v_pot , c_pot  = ds.find_min(('nbody', 'Potential'))
    #center = c_pot[0], c_pot[1], c_pot[2]

    # Calculating the radius for each cell/particle from the center
    x_dist = data['PartType0', 'Coordinates'][:,0] - center[0]
    y_dist = data['PartType0', 'Coordinates'][:,1] - center[1]
    z_dist = data['PartType0', 'Coordinates'][:,2] - center[2]
    radius = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2).in_units('Mpc')
    max_radius  = 3.
    mask_radius = radius < max_radius

    # Apply the mask to the density field, setting unmasked values to 0
    #threshold_dens = 1e-29
    #mask_dens = data['PartType0', 'Density'] > threshold_dens

    threshold_value7 = 1e-7
    cr_prs  = data["gas", "cosmic_ray_pressure"]
    prs     = data["gas", "pressure"]
    chi_cr  = cr_prs/prs
    mask_ratio = chi_cr>threshold_value7

    #return data['PartType2', 'Masses'] * mask_radius * mask_ratio

    # Create a combined mask that indicates whether each cell meets all criteria
    combined_mask = mask_radius & mask_ratio

    # Return masses with np.nan where the mask is False
    masses = data['PartType2', 'Masses'].copy()  # Ensure we do not modify the original data
    masses[~combined_mask] = np.nan
    return masses

def _filtered_mach_radius(field, data):
    #0) Check the projection maps and see if the center is well capture during the whole evolution
    #1) Check also the threshold for the Mach number

    import numpy as np
    #center = data.ds.arr(data.ds.domain_center, 'code_length')
    v_pot , center  = data.ds.find_min(('nbody', 'Potential'))

    #v_gas , c_gas  = ds.find_max(('gas', 'density')) # > changed for this cluster
    #center = c_gas

    # Calculating the radius for each cell/particle from the center
    x_dist = data['PartType0', 'Coordinates'][:,0] - center[0]
    y_dist = data['PartType0', 'Coordinates'][:,1] - center[1]
    z_dist = data['PartType0', 'Coordinates'][:,2] - center[2]
    radius = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2).in_units('Mpc')
    max_radius  = 3.5 #2.5
    mask_radius = radius < max_radius
    # Set your threshold value
    threshold = 4
    # Create a mask that is True for values above the threshold
    #mask = data['PartType0', 'Machnumber'] > threshold
    threshold = 6
    # Create a mask that is True for values above the threshold
    #mask2 = data['PartType0', 'Machnumber'] < threshold
    # Apply the mask to the density field, setting unmasked values to 0
    threshold_dens = 1e-27
    mask_dens = data['PartType0', 'Density'] > threshold_dens
    return data['PartType0', 'Machnumber'] * mask_dens * mask_radius

def _filtered_mach_radius2(field, data):
    # Same as _filtered_mach_radius but with a wider radius cut (r<4.5 Mpc
    # instead of r<3.5 Mpc) and no density threshold -- used for snapshots
    # where the shock-relevant structure extends further from the center.
    import numpy as np
    v_pot, center = data.ds.find_min(('nbody', 'Potential'))

    x_dist = data['PartType0', 'Coordinates'][:,0] - center[0]
    y_dist = data['PartType0', 'Coordinates'][:,1] - center[1]
    z_dist = data['PartType0', 'Coordinates'][:,2] - center[2]
    radius = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2).in_units('Mpc')
    max_radius  = 4.5
    mask_radius = radius < max_radius

    return data['PartType0', 'Machnumber'] * mask_radius

def _filtered_velocities(field, data):
    import numpy as np
    center = data.ds.arr(data.ds.domain_center, 'code_length')
    # Calculating the radius for each cell/particle from the center
    x_dist = data['PartType0', 'Coordinates'][:,0] - center[0]
    y_dist = data['PartType0', 'Coordinates'][:,1] - center[1]
    z_dist = data['PartType0', 'Coordinates'][:,2] - center[2]
    radius = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2).in_units('Mpc')
    max_radius  = 4.8
    mask_radius = radius < max_radius

    # Apply the mask to the density field, setting unmasked values to 0
    #threshold_dens = 1e-29
    #mask_dens = data['PartType0', 'Density'] > threshold_dens

    threshold_value7 = 1e-7
    cr_prs  = data["gas", "cosmic_ray_pressure"]
    prs     = data["gas", "pressure"]
    chi_cr  = cr_prs/prs
    mask_ratio = chi_cr>threshold_value7

    return data['PartType2', 'Velocities'] * mask_radius * mask_ratio

def _position_x(field, data):
    return data.ds.arr(data["PartType2", "Coordinates"][:,0],
                       "cm")
def _position_y(field, data):
    return data.ds.arr(data["PartType2", "Coordinates"][:,1],
                       "cm")
def _position_z(field, data):
    return data.ds.arr(data["PartType2", "Coordinates"][:,2],
                       "cm")

# This is creating the field with the correct units
def _velocity_divergence(field, data):
        return data.ds.arr(data["PartType0", "VelocityDivergence"],
                       "code_velocity/code_length")
def _velocity_curl(field, data):
        return data.ds.arr(data["PartType0", "VelocityCurl"],
                       "code_velocity/code_length")
def _energy_dissipation(field, data):
        return data.ds.arr(data["PartType0", "EnergyDissipation"],
                       "code_mass/code_length*code_velocity**3")
def _entropy(field, data):
        return data.ds.arr(data["gas", "kT"]/(data["gas", "number_density"])**(2/3),
                       "keV*cm**2")

def _velocity_turb(field, data):
        return data.ds.arr(data["PartType0", "TurbulentVelocity"],
                       "code_velocity")
def _velocity_solenoidal(field, data):
        return data.ds.arr(data["PartType0", "TurbulentVelocitySolenoidal"],
                       "code_velocity")
def _velocity_compressive(field, data):
        return data.ds.arr(data["PartType0", "TurbulentVelocityCompressive"],
                       "code_velocity")
def _filtering_length(field, data):
        return data.ds.arr(data["PartType0", "TurbulentFilteringLength"],
                       "code_length")

# --- ran with new version of vortex-p

def _velocity_turb2(field, data):
        return data.ds.arr(data["PartType0", "TurbulentVelocity2"],
                       "code_velocity")
def _velocity_solenoidal2(field, data):
        return data.ds.arr(data["PartType0", "TurbulentVelocitySolenoidal2"],
                       "code_velocity")
def _velocity_compressive2(field, data):
        return data.ds.arr(data["PartType0", "TurbulentVelocityCompressive2"],
                       "code_velocity")
def _filtering_length2(field, data):
        return data.ds.arr(data["PartType0", "TurbulentFilteringLength2"],
                       "code_length")

# ---- different tolerance ----

def _velocity_turb_tol0p3(field, data):
        return data.ds.arr(data["PartType0", "TurbulentVelocity_tol0p3"],
                       "code_velocity")
def _velocity_solenoidal_tol0p3(field, data):
        return data.ds.arr(data["PartType0", "TurbulentVelocitySolenoidal_tol0p3"],
                       "code_velocity")
def _velocity_compressive_tol0p3(field, data):
        return data.ds.arr(data["PartType0", "TurbulentVelocityCompressive_tol0p3"],
                       "code_velocity")
def _filtering_length_tol0p3(field, data):
        return data.ds.arr(data["PartType0", "TurbulentFilteringLength_tol0p3"],
                       "code_length")

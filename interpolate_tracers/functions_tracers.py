def find_matched_tracers(threshold, tree_tracer, tree_masked_gas, tracer_x, tracer_y, tracer_z, tracer_ids):
    import numpy as np
     # Find all pairs of points between self and other whose distance is at most r
    # For each element self.data[i] of this tree, results[i] is a list of the indices of its neighbors in other.data.
    coincident_indices = tree_tracer.query_ball_tree(tree_masked_gas, r=threshold)
    
    # Extract the tracer coordinates that match the masked gas particles
    matched_tracer_x = []
    matched_tracer_y = []
    matched_tracer_z = []
    matched_tracer_ids = []

    # gas_indices is a list of indices of gas particles that are within the threshold distance from the tracer particle at tracer_idx
    for tracer_idx, gas_indices in enumerate(coincident_indices):
        # checking that gas_indices list is not empty because if yes, 
        # there are no gas particles within the threshold distance from the current tracer particle.
        if gas_indices:  # If there are matching gas particles (there could be more than 1)
            matched_tracer_x.append(tracer_x[tracer_idx])
            matched_tracer_y.append(tracer_y[tracer_idx])
            matched_tracer_z.append(tracer_z[tracer_idx])
            matched_tracer_ids.append(tracer_ids[tracer_idx])

    # Convert lists to arrays for further processing
    masked_tracer_x = np.array(matched_tracer_x)
    masked_tracer_y = np.array(matched_tracer_y)
    masked_tracer_z = np.array(matched_tracer_z)
    matched_tracer_ids = np.array(matched_tracer_ids)
    
    return coincident_indices, masked_tracer_x, masked_tracer_y, masked_tracer_z, matched_tracer_ids

def find_matched_tracers_nearest(tree_masked_gas, tracer_coordinates, tracer_ids):
    """Match every tracer to its single nearest point in tree_masked_gas (query, not query_ball_tree).

    Unlike find_matched_tracers (threshold-based, some tracers may have no
    match), every tracer is considered matched here -- useful when tracers
    should already be pre-filtered by radius/condition before calling this.
    """
    distances, matched_indices = tree_masked_gas.query(tracer_coordinates, k=1)

    matched_tracer_ids = tracer_ids
    matched_coords = tracer_coordinates

    return matched_indices, matched_coords[:, 0], matched_coords[:, 1], matched_coords[:, 2], matched_tracer_ids

def save_snapshot_to_hdf5(path, snapshot_number, internal_energy, bfield, density, velocities, mach, divv, curlv, energy_diss,
                          vturb, vturb_sol, vturb_comp, lturb, tracer_ids, gas_ids, coords,
                          velocity_conversion_factor, density_conversion_factor, redshift,
                          t_sim, t_cosmo):
    import h5py
    filename = path + f'_tracers_{snapshot_number:03d}.hdf5'

    print("Writing tracer file at: ",filename)

    with h5py.File(filename, 'w') as h5file:
        # Creating datasets for each data array
        h5file.create_dataset('internal_energy', data=internal_energy)
        h5file.create_dataset('Bx', data=bfield[:, 0])
        h5file.create_dataset('By', data=bfield[:, 1])
        h5file.create_dataset('Bz', data=bfield[:, 2])
        h5file.create_dataset('density', data=density)
        #h5file.create_dataset('velocities', data=velocities)
        h5file.create_dataset('mach', data=mach)
        h5file.create_dataset('div_v', data=divv)
        h5file.create_dataset('curl_v', data=curlv)
        h5file.create_dataset('energy_diss', data=energy_diss)
        h5file.create_dataset('particleID', data=tracer_ids)
        h5file.create_dataset('gasID', data=gas_ids)
        h5file.create_dataset('xcoord', data=coords[:, 0])
        h5file.create_dataset('ycoord', data=coords[:, 1])
        h5file.create_dataset('zcoord', data=coords[:, 2])
        h5file.create_dataset('xvelocity', data=velocities[:, 0])
        h5file.create_dataset('yvelocity', data=velocities[:, 1])
        h5file.create_dataset('zvelocity', data=velocities[:, 2])
        # new fields:
        h5file.create_dataset('vturb', data=vturb)
        h5file.create_dataset('vturb_sol', data=vturb_sol)
        h5file.create_dataset('vturb_comp', data=vturb_comp)
        h5file.create_dataset('l_turb', data=lturb)

        # Adding conversion factors and redshift as scalar datasets
        h5file.create_dataset('VelocityConversionFactor', data=velocity_conversion_factor)
        h5file.create_dataset('DensityConversionFactor', data=density_conversion_factor)
        #h5file.create_dataset('MagConversionFactor', data=mag_conversion_factor)
        h5file.create_dataset('Redshift', data=redshift)
        h5file.create_dataset('Time_cosmo', data=t_cosmo)
        h5file.create_dataset('Time_sim', data=t_sim)

def read_tracer_snapshot(path, snapshot_number):
    import h5py
    import numpy as np
    filename = path + f'_tracers_{snapshot_number:03d}.hdf5'

    print("Reading tracer file from: ", filename)

    data = {}

    with h5py.File(filename, 'r') as h5file:
        # Scalars
        data['VelocityConversionFactor'] = h5file['VelocityConversionFactor'][()]
        data['DensityConversionFactor'] = h5file['DensityConversionFactor'][()]
        data['Redshift'] = h5file['Redshift'][()]

        # 1D arrays
        data['internal_energy'] = h5file['internal_energy'][:]
        data['density'] = h5file['density'][:]
        data['mach'] = h5file['mach'][:]
        data['div_v'] = h5file['div_v'][:]
        data['curl_v'] = h5file['curl_v'][:]
        data['energy_diss'] = h5file['energy_diss'][:]
        data['particleID'] = h5file['particleID'][:]
        data['gasID'] = h5file['gasID'][:]
        data['vturb'] = h5file['vturb'][:]
        data['vturb_sol'] = h5file['vturb_sol'][:]
        data['vturb_comp'] = h5file['vturb_comp'][:]
        data['l_turb'] = h5file['l_turb'][:]

        # 3D vectors split into components
        data['bfield'] = np.column_stack([h5file['Bx'][:], h5file['By'][:], h5file['Bz'][:]])
        data['coords'] = np.column_stack([h5file['xcoord'][:], h5file['ycoord'][:], h5file['zcoord'][:]])
        data['velocities'] = np.column_stack([h5file['xvelocity'][:], h5file['yvelocity'][:], h5file['zvelocity'][:]])

    return data

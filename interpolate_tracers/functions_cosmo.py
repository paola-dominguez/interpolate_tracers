
def compute_cosmological_times(redshift, time_between_snapshots, initial_snap, final_snap):
    import astropy.units as u
    # Redshift information
    from astropy.cosmology import Planck13
    from astropy.cosmology import WMAP3 as cosmo #
    from astropy.cosmology import FlatLambdaCDM  # to test deSitter
    from astropy.cosmology import z_at_value
    
    gadget_default_unit_base = {
        'UnitLength_in_cm': 3.08568e+21,
        'UnitMass_in_g': 1.989e+43,
        'UnitVelocity_in_cm_per_s': 100000
    }
    
    UnitLength = gadget_default_unit_base['UnitLength_in_cm']  # in cm
    UnitMass = gadget_default_unit_base['UnitMass_in_g']  # in g
    UnitVelocity = gadget_default_unit_base['UnitVelocity_in_cm_per_s']  # in cm/s
    
    UnitTime = UnitLength / UnitVelocity  # in s
    dt_Gyr = (time_between_snapshots * UnitTime / 3.154e16) * u.Gyr  # Convert to Gyr
    
    num_snaps = final_snap - initial_snap
    time_cosmo = cosmo.age(redshift)
    
    print("Cosmology selected: ", cosmo)
    print("Cosmological time at selected snapshot: ", time_cosmo)
    print("Redshift at selected snapshot: ", redshift)
    print(" --- ")
    
    t_first_snap_cosmo = time_cosmo - num_snaps * dt_Gyr
    zz_first_snap = z_at_value(cosmo.age, t_first_snap_cosmo)
    
    print("Cosmological time at initial snapshot: ", t_first_snap_cosmo)
    print("Redshift at initial snapshot: ", zz_first_snap)
    print(" --- ")
    
    if zz_first_snap <= 0:
        print("Error with redshift computation: Negative redshift!")
        sys.exit()
    
    return time_cosmo, t_first_snap_cosmo, zz_first_snap, dt_Gyr

def snapshot_time_redshift_map(initial_snap, final_snap, ref_snap, z_ref, dt_Gyr):
    """
    Return arrays of snapshot numbers, cosmic time (Gyr), and redshift,
    assuming uniform spacing in time between snapshots.
    """
    import numpy as np
    import astropy.units as u
    # Redshift information
    from astropy.cosmology import Planck13
    from astropy.cosmology import WMAP3 as cosmo #
    from astropy.cosmology import FlatLambdaCDM  # to test deSitter
    from astropy.cosmology import z_at_value

    snaps = np.arange(initial_snap, final_snap + 1, 1)  # inclusive

    # reference time from cosmology at z_ref
    t_ref = cosmo.age(z_ref)   # Quantity in Gyr

    # time for each snapshot relative to reference
    dN = (snaps - ref_snap)    # signed offsets
    t_snaps = t_ref + dN * dt_Gyr

    # guard rails: cosmology only defined for 0 <= z < inf -> 0 < t <= age(0)
    t0 = cosmo.age(0.0)  # age of Universe today
    bad = (t_snaps <= 0*u.Gyr) | (t_snaps > t0)
    if np.any(bad):
        bad_list = snaps[bad]
        raise ValueError(f"Some snapshot times fall outside [0, age(0)]: {bad_list}")

    # invert time->redshift
    z_snaps = np.array([z_at_value(cosmo.age, tt).value for tt in t_snaps])

    return snaps, t_snaps.to_value(u.Gyr), z_snaps

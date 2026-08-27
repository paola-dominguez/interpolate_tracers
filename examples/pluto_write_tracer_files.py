"""PLUTO tracer-file pipeline: interpolate a fixed-grid PLUTO run's Eulerian
fields onto its Lagrangian tracer particles and write one tracer HDF5 file
per snapshot, in the same schema the AREPO pipeline uses
(interpolate_tracers.functions_tracers.save_snapshot_to_hdf5).

Adapted from a validated production run (a 128^3 periodic ICM-turbulence box,
T2e6_M0.7_particles) -- the interpolation math and unit conversions are
unchanged, only the hardcoded paths/grid size/unit factors are now
constructor arguments (interpolate_tracers.readers.pluto.PlutoTracerPipeline).

Prerequisite: if you want the turbulent-velocity-decomposition fields
(vturb/vturb_comp/vturb_sol) and divv/curlv, run vortex-p on your PLUTO
output first, then call add_vortex_p_fields() once (see below) before
running this script.
"""
import argparse

import numpy as np

from interpolate_tracers.functions_tracers import save_snapshot_to_hdf5
from interpolate_tracers.readers.pluto import PlutoTracerPipeline, add_vortex_p_fields

# ---------------------------- CONFIG (edit me) ----------------------------
RUN_DIR = "/path/to/your/pluto_run/"
VORTEX_OUTPUT_DIR = RUN_DIR + "vortex-p/src/output_files/"
VORTEX_PYTHON_READER_DIR = RUN_DIR + "vortex-p/python_reader/"
OUTPUT_PREFIX = RUN_DIR + "Tracer_files/_tracers"  # save_snapshot_to_hdf5 appends _NNN.hdf5

NX = NY = NZ = 128
# Non-cosmological box: anchor the last matched snapshot to z=0 and map
# earlier snapshots' redshift backwards via cosmic time (see
# PlutoTracerPipeline.compute_redshift docstring). Set ref_snap=None to skip
# redshift assignment entirely (Redshift/Time_cosmo will need to come from
# elsewhere in that case).
REF_SNAP = 162
Z_REF = 0.0
# ----------------------------------------------------------------------------


def prepare_vortex_p_fields(first, last, dry_run=False):
    """One-time prep step: append TurbulentVelocity[Compressive/Solenoidal]
    into every data.NNNN.flt.h5, from already-computed vortex-p output."""
    add_vortex_p_fields(
        RUN_DIR, VORTEX_OUTPUT_DIR, VORTEX_PYTHON_READER_DIR,
        first, last, NX, NY, NZ, dry_run=dry_run,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap', type=int, default=None)
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--last', type=int, default=162)
    parser.add_argument('--every', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    pipeline = PlutoTracerPipeline(
        run_dir=RUN_DIR,
        vortex_output_dir=VORTEX_OUTPUT_DIR,
        vortex_python_reader_dir=VORTEX_PYTHON_READER_DIR,
        nx=NX, ny=NY, nz=NZ,
        ref_snap=REF_SNAP, z_ref=Z_REF,
    )

    snaps = [args.snap] if args.snap is not None else range(args.first, args.last + 1, args.every)

    for snap in snaps:
        print(f"snapshot --> {snap:04d}")
        data = pipeline.process_snapshot(snap, dry_run=args.dry_run)
        if data is None:  # dry run
            continue

        if pipeline.ref_snap is not None:
            redshift = pipeline.compute_redshift(data['time_code'])
            t_cosmo = pipeline.compute_cosmic_age_seconds(data['time_code'])
        else:
            redshift, t_cosmo = 0.0, 0.0

        save_snapshot_to_hdf5(
            OUTPUT_PREFIX, snap,
            data['internal_energy'], np.column_stack([data['Bx'], data['By'], data['Bz']]),
            data['density'], np.column_stack([data['xvelocity'], data['yvelocity'], data['zvelocity']]),
            data['mach'], data['div_v'], data['curl_v'], data['energy_diss'],
            data['vturb'], data['vturb_sol'], data['vturb_comp'], data['l_turb'],
            data['particleID'], data['gasID'],
            np.column_stack([data['xcoord'], data['ycoord'], data['zcoord']]),
            velocity_conversion_factor=pipeline.velocity_factor,
            density_conversion_factor=pipeline.density_factor,
            redshift=redshift,
            t_sim=data['time_code'] * pipeline.time_factor,
            t_cosmo=t_cosmo,
        )

    print("Done!")


if __name__ == '__main__':
    main()

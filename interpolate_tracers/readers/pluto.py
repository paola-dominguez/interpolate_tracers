"""PLUTO reader: fixed uniform-grid tracer-file pipeline.

Generalized from a working, validated pipeline (write_tracer_files_pluto.py,
add_vortex-p_fields.py) run on a 128^3 periodic MHD turbulence box
(T2e6_M0.7_particles). The physics/interpolation logic is unchanged from
that run; what changed is that every path, grid resolution, unit-conversion
factor, and redshift-anchoring choice is now a constructor argument instead
of a hardcoded module-level constant, so this works for any PLUTO run with
Lagrangian tracer particles, not just that one.

Compared to the AREPO reader (readers/arepo.py), PLUTO tracers live on a
fixed Cartesian grid rather than a moving mesh, so "which cell is this
particle in" is a closed-form index computation (interpolation.py) rather
than a KD-tree nearest-neighbor search, and there's no SPH kernel averaging
-- trilinear interpolation of the 8 surrounding cell-centers is the direct
analog.

The turbulent-velocity decomposition fields (TurbulentVelocity[Compressive/
Solenoidal]) are expected to already be present in the grid HDF5 files --
see :func:`add_vortex_p_fields` if you need to append them from vortex-p
output first. divv/curlv are read directly from vortex-p's own output
files via its python_reader module (an external dependency shipped with
vortex-p itself, not part of this package -- pass its directory via
vortex_python_reader_dir).
"""

import os
import sys

import numpy as np
import h5py
import astropy.units as u
from astropy.cosmology import WMAP3 as _default_cosmology
from astropy.cosmology import z_at_value

from ..interpolation import interpolate_field, host_cell_index

GAMMA_DEFAULT = 5.0 / 3.0

# Default unit-conversion factors, as used for a 1-Mpc-box, Gadget-style code
# unit system (code -> cgs). Override for a different unit system.
DENSITY_FACTOR_DEFAULT = 1.66053886e-27    # code -> g/cm^3
VELOCITY_FACTOR_DEFAULT = 5.0e7            # code -> cm/s
LENGTH_FACTOR_DEFAULT = 3.0857e24          # code -> cm (1 Mpc)
B_FACTOR_G_DEFAULT = 7.2225e-6             # code -> Gauss
GYR_S = 3.15576e16                         # s per Gyr

GRID_FIELDS_DEFAULT = (
    'rho', 'prs', 'vx1', 'vx2', 'vx3', 'Bx1', 'Bx2', 'Bx3',
    'TurbulentVelocity', 'TurbulentVelocityCompressive', 'TurbulentVelocitySolenoidal',
)


def read_pluto_particles(fname):
    """Read a PLUTO binary particle file (.dbl or .flt, either precision).

    Returns a dict of 1D arrays keyed by PLUTO's own field names (id, x1,
    x2, x3, vx1, vx2, vx3, tinj, color, ...), plus scalar 'time' and
    'nparticles'.
    """
    header = {}
    with open(fname, 'rb') as f:
        while True:
            line = f.readline()
            if not line.startswith(b'#'):
                break
            parts = line.decode().split()[1:]
            if parts:
                header[parts[0]] = parts[1:]
        header_end = f.tell() - len(line)

        field_names = header['field_names']
        field_dim = np.array(header['field_dim'], dtype=int)
        nparticles = int(header['nparticles'][0])
        endian = '<' if header['endianity'][0] == 'little' else '>'
        precision = header['precision'][0]
        dtype = endian + ('f8' if precision == 'double' else 'f4')
        time = float(header['time'][0])
        tot_fdim = int(field_dim.sum())

        f.seek(header_end)
        data = np.fromfile(f, dtype=dtype, count=nparticles * tot_fdim)

    data = data.reshape(nparticles, tot_fdim)
    out = {name: data[:, i] for i, name in enumerate(field_names)}
    out['time'] = time
    out['nparticles'] = nparticles
    return out


def add_vortex_p_fields(run_dir, vortex_output_dir, vortex_python_reader_dir,
                         first, last, nx, ny, nz, size=1.0, dry_run=False):
    """Append vortex-p's turbulent-velocity decomposition into data.NNNN.flt.h5.

    Requires vortex-p to have already been run on this PLUTO output (a
    separate Fortran code, not part of this package); vortex_output_dir
    points at its output_files/ directory and vortex_python_reader_dir at
    its python_reader/ directory (added to sys.path here).

    No transpose is needed between vortex-p's vcomp/vsol arrays and PLUTO's
    own vx1/vx2/vx3 -- verified empirically (the round trip through the
    Fortran writer and Python's reshape(order='F') cancels out the naive
    transpose one might expect from the raw-HDF5-read axis convention).
    """
    if vortex_python_reader_dir not in sys.path:
        sys.path.append(vortex_python_reader_dir)
    import parameters  # noqa: E402
    import vortex_reader  # noqa: E402

    parameters.write_parameters(nx, ny, nz, 0, size, path=vortex_output_dir)

    for it in range(first, last + 1):
        vcompx, vcompy, vcompz = vortex_reader.read_vcomp(it, path=vortex_output_dir, parameters_path=vortex_output_dir)
        vsolx, vsoly, vsolz = vortex_reader.read_vsol(it, path=vortex_output_dir, parameters_path=vortex_output_dir)
        vcompx, vcompy, vcompz = vcompx[0], vcompy[0], vcompz[0]
        vsolx, vsoly, vsolz = vsolx[0], vsoly[0], vsolz[0]

        vturb_comp = np.sqrt(vcompx**2 + vcompy**2 + vcompz**2)
        vturb_sol = np.sqrt(vsolx**2 + vsoly**2 + vsolz**2)

        fn = os.path.join(run_dir, f"data.{it:04d}.flt.h5")
        print(f"-> Snapshot: {it}  ({fn})")

        with h5py.File(fn, "a") as f:
            group = f[f"Timestep_{it}/vars"]
            shape_ref = group["vx1"].shape

            vx1, vx2, vx3 = group["vx1"][:], group["vx2"][:], group["vx3"][:]
            vturb = np.sqrt(vx1**2 + vx2**2 + vx3**2)

            vrec = np.sqrt((vcompx + vsolx)**2 + (vcompy + vsoly)**2 + (vcompz + vsolz)**2)
            rel_err = np.abs(vrec - vturb) / (np.abs(vturb) + 1e-30)
            print(f"   |v| vs |vcomp+vsol| relative error: "
                  f"median={np.median(rel_err):.3e}  max={np.max(rel_err):.3e}")

            if dry_run:
                continue

            data_dict = {
                "TurbulentVelocity": np.asarray(vturb, dtype="f4"),
                "TurbulentVelocitySolenoidal": np.asarray(vturb_sol, dtype="f4"),
                "TurbulentVelocityCompressive": np.asarray(vturb_comp, dtype="f4"),
            }
            for name, data in data_dict.items():
                if data.shape != shape_ref:
                    raise ValueError(f"{name} has shape {data.shape}, expected {shape_ref}")
                if name in group:
                    del group[name]
                group.create_dataset(name, data=data, dtype="f4")


class PlutoTracerPipeline:
    """Config + methods for turning one PLUTO run's output into tracer HDF5 files.

    Parameters
    ----------
    run_dir : str
        Directory holding data.NNNN.flt.h5 and particles.NNNN_00.{flt,dbl}.
    vortex_output_dir, vortex_python_reader_dir : str, optional
        vortex-p's output_files/ and python_reader/ directories (needed only
        for divv/curlv via read_divv_curlv). Required if you call
        process_snapshot with include_divv_curl=True (the default).
    nx, ny, nz : int
        Grid resolution.
    ref_snap, z_ref : int, float, optional
        For a non-cosmological box, redshift has no meaning intrinsic to the
        simulation and must be assigned: ref_snap's snapshot is anchored to
        redshift z_ref, and every other snapshot's redshift is computed by
        mapping cosmic time backwards/forwards via `cosmology`. If ref_snap
        is None, compute_redshift()/process_snapshot() will raise.
    cosmology : astropy.cosmology instance, optional
        Defaults to WMAP3 (matches the cosmology used elsewhere in the
        original AREPO pipeline this was adapted from -- change if your
        redshift anchoring should use a different cosmology).
    density_factor, velocity_factor, length_factor, b_factor_g : float
        code -> cgs unit conversion factors. Defaults match a 1 kpc-ish
        Gadget-style unit triplet scaled to a 1 Mpc box (see module
        docstring) -- override for your own PLUTO unit system.
    """

    def __init__(self, run_dir, vortex_output_dir=None, vortex_python_reader_dir=None,
                 nx=128, ny=128, nz=128, gamma=GAMMA_DEFAULT,
                 density_factor=DENSITY_FACTOR_DEFAULT, velocity_factor=VELOCITY_FACTOR_DEFAULT,
                 length_factor=LENGTH_FACTOR_DEFAULT, b_factor_g=B_FACTOR_G_DEFAULT,
                 ref_snap=None, z_ref=0.0, cosmology=None):
        self.run_dir = run_dir.rstrip('/') + '/'
        self.vortex_output_dir = vortex_output_dir
        self.vortex_python_reader_dir = vortex_python_reader_dir
        self.nx, self.ny, self.nz = nx, ny, nz
        self.gamma = gamma
        self.density_factor = density_factor
        self.velocity_factor = velocity_factor
        self.length_factor = length_factor
        self.b_factor_g = b_factor_g
        self.time_factor = length_factor / velocity_factor

        self.ref_snap = ref_snap
        self.z_ref = z_ref
        self.cosmology = cosmology or _default_cosmology

        self._snap_times_code = None
        self._t_ref_age = None
        self._time_ref_code = None

    # -- particle / grid I/O -------------------------------------------------

    @staticmethod
    def read_pluto_particles(fname):
        return read_pluto_particles(fname)

    def read_grid_fields(self, snap, fields=GRID_FIELDS_DEFAULT):
        """Read fields from data.NNNN.flt.h5, shaped (nz, ny, nx)."""
        out = {}
        fname = self.run_dir + f'data.{snap:04d}.flt.h5'
        with h5py.File(fname, 'r') as f:
            grp = f[f'Timestep_{snap}/vars']
            for key in fields:
                out[key] = grp[key][:]
        return out

    def read_divv_curlv(self, snap):
        """Read vortex-p's divergence/curl of velocity on the base grid."""
        if self.vortex_output_dir is None or self.vortex_python_reader_dir is None:
            raise ValueError("vortex_output_dir and vortex_python_reader_dir are required for read_divv_curlv")
        if self.vortex_python_reader_dir not in sys.path:
            sys.path.append(self.vortex_python_reader_dir)
        import vortex_reader  # noqa: E402

        divv = vortex_reader.read_divv(snap, path=self.vortex_output_dir, parameters_path=self.vortex_output_dir)[0]
        curlvx, curlvy, curlvz = vortex_reader.read_curlv(snap, path=self.vortex_output_dir, parameters_path=self.vortex_output_dir)
        return divv, curlvx[0], curlvy[0], curlvz[0]

    # -- redshift assignment for a non-cosmological box ----------------------

    def load_snapshot_times(self, flt_h5_out_path=None):
        """Load the snap -> code-time map from PLUTO's flt.h5.out log.

        Required before compute_redshift()/compute_cosmic_age_seconds() --
        called automatically by them if not already loaded.
        flt_h5_out_path defaults to run_dir/flt.h5.out.
        """
        path = flt_h5_out_path or (self.run_dir + 'flt.h5.out')
        times = {}
        with open(path) as f:
            for line in f:
                parts = line.split()
                times[int(parts[0])] = float(parts[1])
        self._snap_times_code = times
        if self.ref_snap is not None:
            self._t_ref_age = self.cosmology.age(self.z_ref)
            self._time_ref_code = times[self.ref_snap]
        return times

    def compute_redshift(self, time_code):
        """Assign a redshift to a snapshot's code time, anchoring ref_snap to z_ref.

        For a non-cosmological, non-expanding box, redshift has no meaning
        intrinsic to the simulation -- this anchors ref_snap to z_ref and
        maps cosmic time backwards via the configured cosmology, matching
        functions_cosmo.snapshot_time_redshift_map's method.
        """
        if self.ref_snap is None:
            raise ValueError("ref_snap must be set to compute a redshift for this run")
        if self._time_ref_code is None:
            self.load_snapshot_times()

        delta_t_gyr = (self._time_ref_code - time_code) * self.time_factor / GYR_S
        if abs(delta_t_gyr) < 1e-6:
            # exactly the reference snapshot -- z_at_value's root search sits
            # right on its zmin boundary here and raises, so short-circuit
            return self.z_ref
        age_at_snap = self._t_ref_age - delta_t_gyr * u.Gyr
        if age_at_snap.value <= 0:
            raise ValueError(
                f"time_code={time_code}: implied cosmic age {age_at_snap} <= 0 "
                f"(more than one Hubble time before the z={self.z_ref} reference "
                f"at snap {self.ref_snap}); redshift assignment is not meaningful here.")
        return float(z_at_value(self.cosmology.age, age_at_snap, zmin=1e-8, zmax=1000).value)

    def compute_cosmic_age_seconds(self, time_code):
        """Cosmic age (seconds) implied by the same ref_snap/z_ref anchoring as compute_redshift()."""
        if self.ref_snap is None:
            raise ValueError("ref_snap must be set to compute a cosmic age for this run")
        if self._time_ref_code is None:
            self.load_snapshot_times()
        return (self._t_ref_age.to_value(u.Gyr) * GYR_S
                - (self._time_ref_code - time_code) * self.time_factor)

    # -- main per-snapshot pipeline -------------------------------------------

    def process_snapshot(self, snap, particle_suffix='_00.flt', include_divv_curl=True,
                          dry_run=False):
        """Read one snapshot's particles + grid, interpolate, return a tracer-file dict.

        Sorts particles by id first, so row i is the same physical tracer in
        every snapshot's output (PLUTO's on-disk particle order drifts over
        time as tracers migrate between MPI domains -- see functions_tracers
        module docstring / the AREPO pipeline's own use of a fixed sorted ID
        list for the same reason).

        Returns a dict of arrays with the same field names
        functions_tracers.save_snapshot_to_hdf5 expects (pass **result to
        it, or use save_tracer_snapshot below), or None if dry_run=True
        (min/max of each interpolated field is printed instead).
        """
        part_fname = self.run_dir + f'particles.{snap:04d}{particle_suffix}'
        part = self.read_pluto_particles(part_fname)

        id_order = np.argsort(part['id'])
        for key in part:
            if key not in ('time', 'nparticles'):
                part[key] = part[key][id_order]

        grid = self.read_grid_fields(snap)
        x1, x2, x3 = part['x1'], part['x2'], part['x3']
        npart = part['nparticles']

        interp = {}
        for key in ['rho', 'prs', 'Bx1', 'Bx2', 'Bx3',
                    'TurbulentVelocity', 'TurbulentVelocityCompressive', 'TurbulentVelocitySolenoidal']:
            interp[key] = interpolate_field(grid[key], x1, x2, x3, self.nx, self.ny, self.nz)

        if include_divv_curl:
            divv, curlvx, curlvy, curlvz = self.read_divv_curlv(snap)
            interp['divv'] = interpolate_field(divv, x1, x2, x3, self.nx, self.ny, self.nz)
            interp['curlvx'] = interpolate_field(curlvx, x1, x2, x3, self.nx, self.ny, self.nz)
            interp['curlvy'] = interpolate_field(curlvy, x1, x2, x3, self.nx, self.ny, self.nz)
            interp['curlvz'] = interpolate_field(curlvz, x1, x2, x3, self.nx, self.ny, self.nz)
        else:
            interp['divv'] = np.zeros(npart)
            interp['curlvx'] = interp['curlvy'] = interp['curlvz'] = np.zeros(npart)

        if dry_run:
            for key, arr in interp.items():
                print(f"  {key}: min={np.min(arr):.4e} max={np.max(arr):.4e}")
            return None

        internal_energy = interp['prs'] / ((self.gamma - 1.0) * interp['rho'])
        curl_v_mag = np.sqrt(interp['curlvx']**2 + interp['curlvy']**2 + interp['curlvz']**2)
        gas_id = host_cell_index(x1, x2, x3, self.nx, self.ny, self.nz)

        return {
            'particleID': part['id'].astype(np.int64),
            'gasID': gas_id,
            'xcoord': x1 * self.length_factor,
            'ycoord': x2 * self.length_factor,
            'zcoord': x3 * self.length_factor,
            'xvelocity': part['vx1'] * self.velocity_factor,
            'yvelocity': part['vx2'] * self.velocity_factor,
            'zvelocity': part['vx3'] * self.velocity_factor,
            'density': interp['rho'] * self.density_factor,
            'internal_energy': internal_energy * self.velocity_factor**2,
            'Bx': interp['Bx1'] * self.b_factor_g,
            'By': interp['Bx2'] * self.b_factor_g,
            'Bz': interp['Bx3'] * self.b_factor_g,
            'vturb': interp['TurbulentVelocity'] * self.velocity_factor,
            'vturb_comp': interp['TurbulentVelocityCompressive'] * self.velocity_factor,
            'vturb_sol': interp['TurbulentVelocitySolenoidal'] * self.velocity_factor,
            'div_v': interp['divv'] / self.time_factor,
            'curl_v': curl_v_mag / self.time_factor,
            # not computed unless a shock finder is enabled for this run
            'mach': np.zeros(npart, dtype=np.float32),
            'energy_diss': np.zeros(npart, dtype=np.float32),
            # no adaptive filtering -> grid resolution scale as a stand-in
            'l_turb': np.full(npart, (1.0 / self.nx) * self.length_factor, dtype=np.float32),
            'time_code': part['time'],
        }

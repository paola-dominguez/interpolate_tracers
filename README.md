# interpolate_tracers

[![Tests](https://github.com/paola-dominguez/interpolate_tracers/actions/workflows/ci.yml/badge.svg)](https://github.com/paola-dominguez/interpolate_tracers/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Uses yt](https://img.shields.io/badge/works%20with-yt-blue)](https://yt-project.org/)
[![Uses Astropy](https://img.shields.io/badge/uses-astropy-orange)](https://www.astropy.org/)
[![HDF5 with h5py](https://img.shields.io/badge/hdf5-h5py-ff69b4)](https://www.h5py.org/)

Sample Eulerian simulation fields (density, velocity, B-field, turbulence
diagnostics, ...) onto Lagrangian tracer particles, and write them out in a
single shared HDF5 schema for downstream analysis (e.g. a Fokker-Planck
cosmic-ray transport code). Built for and validated against **AREPO**
(moving-mesh) and **PLUTO** (fixed uniform grid), but the core is
code-agnostic: any simulation that writes HDF5 output can be supported by
adding one new reader module.

## Why two reading strategies

The two simulation codes this package was built against sample fields onto
tracers in genuinely different ways, and both are provided:

- **AREPO** (`interpolate_tracers.readers.arepo`): tracers move relative to
  the mesh, so field values are assigned via **SPH-kernel-weighted
  averaging** over a fixed number of nearby gas cells
  (`interpolate_tracers.kernels` -- `W_M4`/`W_C4`/`W_C6`/`W_M5`/`W_M6`
  kernels), found with a KD-tree.
- **PLUTO** (`interpolate_tracers.readers.pluto`): tracers sit on a fixed
  uniform Cartesian grid, so "which cell is this particle in" is a
  closed-form index computation, and field values are assigned via
  **periodic trilinear interpolation** of the 8 surrounding cell-centers
  (`interpolate_tracers.interpolation`).

Everything else -- the tracer-file HDF5 schema
(`interpolate_tracers.functions_tracers`), cosmological time/redshift
bookkeeping (`interpolate_tracers.functions_cosmo`), KD-tree point matching
(`find_matched_tracers`/`find_matched_tracers_nearest`), and interpolation
diagnostics (`interpolate_tracers.validate`) -- is shared between both, and
usable directly if you're adding support for a third code.

## Installation

```bash
git clone https://github.com/paola-dominguez/interpolate_tracers.git
cd interpolate_tracers
pip install -e .            # core (numpy, scipy, h5py, astropy)
pip install -e ".[arepo]"   # + yt, for the AREPO reader
pip install -e ".[dev]"     # + pytest, to run the test suite
```

## Quick start

Both readers ultimately produce a set of per-tracer arrays (density,
velocity, B-field, ...) that get written with the shared schema:

```python
from interpolate_tracers.functions_tracers import save_snapshot_to_hdf5, read_tracer_snapshot

save_snapshot_to_hdf5(
    "Tracer_files/_tracers", snapshot_number,
    internal_energy, bfield, density, velocities,
    mach, div_v, curl_v, energy_diss,
    vturb, vturb_sol, vturb_comp, l_turb,
    tracer_ids, gas_ids, coords,
    velocity_conversion_factor, density_conversion_factor, redshift,
    t_sim, t_cosmo,
)

data = read_tracer_snapshot("Tracer_files/_tracers", snapshot_number)
```

**PLUTO** (fixed grid, trilinear interpolation):

```python
from interpolate_tracers.readers.pluto import PlutoTracerPipeline

pipeline = PlutoTracerPipeline(
    run_dir="/path/to/pluto_run/",
    vortex_output_dir="/path/to/pluto_run/vortex-p/src/output_files/",
    vortex_python_reader_dir="/path/to/pluto_run/vortex-p/python_reader/",
    nx=128, ny=128, nz=128,
    ref_snap=162, z_ref=0.0,   # redshift anchoring for a non-cosmological box
)
data = pipeline.process_snapshot(snap=100)
```

**AREPO** (moving mesh, SPH-kernel averaging):

```python
from interpolate_tracers.readers.arepo import add_derived_fields, kernel_average_at_gas_positions

add_derived_fields(ds)  # registers divv/curlv/energy_diss/turbulence fields on a yt dataset
results, gas_indices = kernel_average_at_gas_positions(
    tree_gas_all, gas_coordinates, matched_gas_ids, gas_ids_all,
    internal_energy_all, bfield_all, density_all, velocities_all,
    mach_all, divv_all, curlv_all, energy_diss_all,
    vturb_all, vturb_sol_all, vturb_comp_all, length_all,
    num_neighbors=32, kernel_type="W_M4", radius_cut=3.5,
)
```

See `examples/` for complete, runnable scripts adapted from validated
production runs:

- `examples/arepo_find_tracer_ids.py` -- find tracers matching a physical
  condition (e.g. "in a shock") across a snapshot range.
- `examples/arepo_write_tracer_files.py` -- kernel-average fields onto those
  tracers and write tracer HDF5 files.
- `examples/pluto_write_tracer_files.py` -- the PLUTO equivalent, single step
  (no separate ID-matching pass needed on a fixed grid).

## Input / output schema

**AREPO input** (HDF5 snapshot): `/PartType0/{Coordinates,ParticleIDs,Density,
EnergyDissipation,InternalEnergy,MagneticField,Velocities,VelocityCurl,
VelocityDivergence}`, `/PartType2/{Coordinates,ParentID,ParticleIDs}`; optionally
`Machnumber` (shock finder) and `TurbulentVelocity[Compressive/Solenoidal]`,
`TurbulentFilteringLength` (from [vortex-p](https://github.com/dvallesp/vortex-p)).

**PLUTO input**: `data.NNNN.flt.h5` (grid: `rho,prs,vx1-3,Bx1-3`, plus
vortex-p's turbulent-velocity fields if added via
`interpolate_tracers.readers.pluto.add_vortex_p_fields`) and
`particles.NNNN_00.{flt,dbl}` (Lagrangian tracer particles, PLUTO's native
binary particle format).

**Output** (both readers, identical schema): one HDF5 file per snapshot with
`particleID, gasID, {x,y,z}coord, {x,y,z}velocity, density, internal_energy,
Bx, By, Bz, vturb, vturb_comp, vturb_sol, div_v, curl_v, mach, energy_diss,
l_turb`, plus provenance scalars (`VelocityConversionFactor`,
`DensityConversionFactor`, `Redshift`, `Time_sim`, `Time_cosmo`).

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

The test suite covers the simulation-code-agnostic core (kernels,
interpolation, cosmology, matching, HDF5 round-trip) with synthetic data --
it doesn't require real simulation output.

## License

MIT -- see [LICENSE](LICENSE).

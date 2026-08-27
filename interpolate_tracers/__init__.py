"""interpolate_tracers: sample Eulerian simulation fields onto Lagrangian tracers.

Two things this package provides, independent of which simulation code
produced the data:

- A generic core: SPH-kernel neighbor averaging (:mod:`interpolate_tracers.kernels`),
  periodic trilinear grid interpolation (:mod:`interpolate_tracers.interpolation`),
  point matching via KD-trees (:mod:`interpolate_tracers.matching`), cosmological
  time/redshift bookkeeping (:mod:`interpolate_tracers.functions_cosmo`), a shared
  tracer-file HDF5 schema (:mod:`interpolate_tracers.functions_tracers`), and
  interpolation-quality diagnostics (:mod:`interpolate_tracers.validate`).
- Per-code readers (:mod:`interpolate_tracers.readers`) that adapt a specific
  simulation code's native output into the shapes the core functions expect.
  Included: AREPO (moving-mesh, via yt + SPH-kernel neighbor averaging) and
  PLUTO (fixed uniform grid, via direct HDF5/binary I/O + trilinear
  interpolation). Adding support for another code means writing one new
  reader module, not duplicating the pipeline.
"""

__version__ = "0.1.0"

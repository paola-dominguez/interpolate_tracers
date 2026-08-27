"""Per-simulation-code readers.

Each reader adapts one code's native output into the shapes the generic
core (:mod:`interpolate_tracers.kernels`, :mod:`interpolate_tracers.interpolation`,
:mod:`interpolate_tracers.functions_tracers`) expects. Currently included:
AREPO (:mod:`interpolate_tracers.readers.arepo`, moving-mesh via yt +
SPH-kernel neighbor averaging) and PLUTO (:mod:`interpolate_tracers.readers.pluto`,
fixed uniform grid via direct HDF5/binary I/O + trilinear interpolation).
"""

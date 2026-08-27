"""Interpolation-quality diagnostics, generalized from the PLUTO pipeline's
one-off check_tracer_interpolation.py / check_tracer_statistics.py scripts.

These only assume "a grid array" and "tracer-sampled values at known
positions" -- nothing about which simulation code produced either one, and
nothing about specific file paths or snapshot numbers. Plotting helpers take
a matplotlib Axes so the caller controls figure layout/output.
"""

import numpy as np


def pdf_statistics(grid_values, tracer_values):
    """Compare a grid (Eulerian) field against its tracer-sampled (Lagrangian) values.

    Useful when tracers sample the volume roughly uniformly (e.g. ~1
    tracer/cell): the two distributions should be close to indistinguishable
    if the interpolation/averaging is unbiased.

    Returns a dict with mean/median/std for both and their fractional
    differences (grid as the reference).
    """
    grid_values = np.asarray(grid_values, dtype=np.float64).ravel()
    tracer_values = np.asarray(tracer_values, dtype=np.float64).ravel()

    gm, tm = grid_values.mean(), tracer_values.mean()
    gmed, tmed = np.median(grid_values), np.median(tracer_values)
    gs, ts = grid_values.std(), tracer_values.std()

    return {
        "grid_mean": gm, "tracer_mean": tm,
        "mean_frac_diff": (tm - gm) / gm if gm != 0 else np.nan,
        "grid_median": gmed, "tracer_median": tmed,
        "median_frac_diff": (tmed - gmed) / gmed if gmed != 0 else np.nan,
        "grid_std": gs, "tracer_std": ts,
    }


def relative_error(interpolated, reference):
    """Elementwise |interpolated - reference| / |reference|, with a floor to avoid div-by-zero."""
    interpolated = np.asarray(interpolated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return np.abs(interpolated - reference) / (np.abs(reference) + 1e-30)


def plot_pdf_comparison(ax, grid_values, tracer_values, label, bins=120,
                         grid_style=None, tracer_style=None):
    """Overlay grid (Eulerian) vs. tracer (Lagrangian) PDFs on a given Axes."""
    grid_values = np.asarray(grid_values, dtype=np.float64).ravel()
    tracer_values = np.asarray(tracer_values, dtype=np.float64).ravel()
    grid_style = grid_style or {"color": "C0", "lw": 2}
    tracer_style = tracer_style or {"color": "C1", "lw": 2, "linestyle": "--"}

    edges = np.linspace(min(grid_values.min(), tracer_values.min()),
                         max(grid_values.max(), tracer_values.max()), bins)
    ax.hist(grid_values, bins=edges, density=True, histtype='step',
            label='grid (Eulerian)', **grid_style)
    ax.hist(tracer_values, bins=edges, density=True, histtype='step',
            label='tracers (interp.)', **tracer_style)
    ax.axvline(grid_values.mean(), color=grid_style.get("color", "C0"), lw=1, alpha=0.6)
    ax.axvline(tracer_values.mean(), color=tracer_style.get("color", "C1"), lw=1, alpha=0.6,
               linestyle=tracer_style.get("linestyle", "--"))
    ax.set_xlabel(label)
    ax.set_ylabel('PDF')
    return ax


def plot_hexbin_comparison(ax, reference_values, interpolated_values, label,
                            gridsize=60, sample=None, rng=None):
    """Hexbin of (reference/host-cell value) vs. (interpolated value) with a y=x line.

    Flags systematic bias (points off the diagonal) rather than just
    reporting a summary statistic. `sample` optionally subsamples both
    arrays (same indices) for large point counts.
    """
    reference_values = np.asarray(reference_values, dtype=np.float64)
    interpolated_values = np.asarray(interpolated_values, dtype=np.float64)

    if sample is not None and sample < len(reference_values):
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(reference_values), size=sample, replace=False)
        reference_values = reference_values[idx]
        interpolated_values = interpolated_values[idx]

    vmin = min(reference_values.min(), interpolated_values.min())
    vmax = max(reference_values.max(), interpolated_values.max())
    hb = ax.hexbin(reference_values, interpolated_values, gridsize=gridsize,
                    cmap='viridis', mincnt=1, extent=[vmin, vmax, vmin, vmax])
    ax.plot([vmin, vmax], [vmin, vmax], 'r-', lw=1)
    ax.set_xlabel(label + ' (host cell)')
    ax.set_ylabel(label + ' (interpolated)')
    return hb

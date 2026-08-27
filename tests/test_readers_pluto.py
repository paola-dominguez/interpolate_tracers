import numpy as np
import pytest

from interpolate_tracers.readers.pluto import read_pluto_particles, PlutoTracerPipeline


def _write_synthetic_particle_file(path, nparticles=3, time=1.5):
    field_names = ['id', 'x1', 'x2', 'x3', 'vx1', 'vx2', 'vx3', 'tinj', 'color']
    rng = np.random.default_rng(1)
    data = rng.random((nparticles, len(field_names)))
    data[:, 0] = np.arange(nparticles)  # id column, integer-valued

    header_lines = [
        "# field_names " + " ".join(field_names),
        "# field_dim " + " ".join(["1"] * len(field_names)),
        f"# nparticles {nparticles}",
        "# endianity little",
        "# precision double",
        f"# time {time}",
    ]
    with open(path, 'wb') as f:
        for line in header_lines:
            f.write((line + "\n").encode())
        data.astype('<f8').tofile(f)
    return field_names, data, time


def test_read_pluto_particles_round_trip(tmp_path):
    path = str(tmp_path / "particles.0000_00.flt")
    field_names, data, time = _write_synthetic_particle_file(path)

    out = read_pluto_particles(path)

    assert out['nparticles'] == data.shape[0]
    assert out['time'] == pytest.approx(time)
    for i, name in enumerate(field_names):
        np.testing.assert_allclose(out[name], data[:, i])


def test_pluto_pipeline_compute_redshift_anchors_ref_snap(tmp_path):
    flt_out = tmp_path / "flt.h5.out"
    # code-time units chosen so the implied cosmic-age span is a few Gyr
    # (matches the order of magnitude used in the real run this was adapted from)
    lines = [f"{snap} {float(snap)}" for snap in range(6)]
    flt_out.write_text("\n".join(lines) + "\n")

    pipeline = PlutoTracerPipeline(run_dir=str(tmp_path), ref_snap=5, z_ref=0.0)
    pipeline.load_snapshot_times(flt_h5_out_path=str(flt_out))

    z_at_ref = pipeline.compute_redshift(5.0)
    assert z_at_ref == 0.0

    z_earlier = pipeline.compute_redshift(0.0)
    assert z_earlier > 0.0  # earlier snapshot -> higher redshift

    z_mid = pipeline.compute_redshift(2.5)
    assert 0.0 < z_mid < z_earlier


def test_pluto_pipeline_compute_redshift_requires_ref_snap(tmp_path):
    pipeline = PlutoTracerPipeline(run_dir=str(tmp_path))
    with pytest.raises(ValueError):
        pipeline.compute_redshift(0.0)

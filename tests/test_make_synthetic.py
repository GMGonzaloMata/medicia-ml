"""Pruebas de make_synthetic.py: los archivos son sintéticos y cada defecto es el que dice ser."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import make_synthetic as synthetic  # noqa: E402

TWO_MEGABYTES = 2 * 1024 * 1024


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("synthetic")
    return {path.name: path for path in synthetic.build_files(out)}


def test_writes_all_expected_files(files):
    assert set(files) == {
        "valido.npz",
        "sin-tabular.npz",
        "forma-incorrecta.npz",
        "con-nan.npz",
        "sexo-2.npz",
        "sin-estandarizar.npz",
        "con-objetos.npz",
        "con-extras.npz",
        "grande.npz",
    }


def test_valid_file_matches_the_contract(files):
    with np.load(files["valido.npz"]) as data:
        assert set(data.files) == {"waveform", "tabular"}
        wave, tab = data["waveform"], data["tabular"]
    assert wave.shape == (1, 2500, 12) and wave.dtype == np.float32
    assert tab.shape == (7,) and tab.dtype == np.float32
    assert np.isfinite(wave).all() and np.isfinite(tab).all()
    assert tab[0] in (0.0, 1.0)
    assert np.abs(wave).max() < 100 and np.abs(tab[1:]).max() < 20


def test_generation_is_deterministic(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = {p.name: p.read_bytes() for p in synthetic.build_files(tmp_path / "a")}
    second = {p.name: p.read_bytes() for p in synthetic.build_files(tmp_path / "b")}
    assert first["valido.npz"] == second["valido.npz"]


def test_each_defect_has_its_defect(files):
    with np.load(files["sin-tabular.npz"]) as data:
        assert data.files == ["waveform"]
    with np.load(files["forma-incorrecta.npz"]) as data:
        assert data["waveform"].shape == (1, 2500, 11)
    with np.load(files["con-nan.npz"]) as data:
        assert np.isnan(data["waveform"]).sum() == 1
    with np.load(files["sexo-2.npz"]) as data:
        assert data["tabular"][0] == 2
    with np.load(files["sin-estandarizar.npz"]) as data:
        assert data["tabular"][1] == 450
    with np.load(files["con-objetos.npz"], allow_pickle=True) as data:
        assert data["tabular"].dtype == object
    with np.load(files["con-extras.npz"]) as data:
        assert {"waveform", "tabular"} < set(data.files) and len(data.files) > 2


def test_oversized_file_exceeds_two_megabytes(files):
    assert files["grande.npz"].stat().st_size > TWO_MEGABYTES
    assert files["valido.npz"].stat().st_size < TWO_MEGABYTES


def test_refuses_to_write_inside_a_git_repository():
    with pytest.raises(SystemExit, match="repositorio"):
        synthetic.ensure_outside_git(Path(__file__).resolve().parent)

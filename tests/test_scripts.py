"""Pruebas de los scripts de verificación con datos sintéticos (nunca con datos reales)."""

import gzip
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import common  # noqa: E402
import make_sample  # noqa: E402
import package_artifacts as pkg  # noqa: E402


def make_model_dir(path: Path, extra: str | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name in pkg.ARTIFACT_FILES:
        (path / name).write_bytes(name.encode() * 10)
    if extra:
        (path / extra).write_bytes(b"x")
    return path


def test_package_contains_exactly_the_four_files(tmp_path):
    out = tmp_path / "out" / "model.tar.gz"
    pkg.build_archive(make_model_dir(tmp_path / "m"), "0.1.0", out, "abc123")
    with tarfile.open(out) as tar:
        assert sorted(tar.getnames()) == sorted([*pkg.ARTIFACT_FILES, "MANIFEST.json"])
        manifest = json.load(tar.extractfile("MANIFEST.json"))
    assert manifest["model_version"] == "0.1.0"
    assert manifest["code_version"] == "abc123"
    for name, digest in manifest["files"].items():
        assert digest == pkg.sha256_of(tmp_path / "m" / name)


def test_package_is_deterministic(tmp_path):
    model = make_model_dir(tmp_path / "m")
    first = pkg.build_archive(model, "0.1.0", tmp_path / "a.tar.gz", "abc")
    second = pkg.build_archive(model, "0.1.0", tmp_path / "b.tar.gz", "abc")
    assert first == second


def test_package_refuses_extra_files(tmp_path):
    with pytest.raises(SystemExit, match="no son artefactos"):
        pkg.build_archive(make_model_dir(tmp_path / "m", extra="datos.npy"), "0.1.0", tmp_path / "o.tar.gz", "abc")


def test_package_refuses_missing_files(tmp_path):
    model = make_model_dir(tmp_path / "m")
    (model / "weights.pt").unlink()
    with pytest.raises(SystemExit, match="Faltan archivos"):
        pkg.build_archive(model, "0.1.0", tmp_path / "o.tar.gz", "abc")


def test_archive_is_valid_gzip(tmp_path):
    out = tmp_path / "o.tar.gz"
    pkg.build_archive(make_model_dir(tmp_path / "m"), "0.1.0", out, "abc")
    with gzip.open(out) as handle:
        assert handle.read(1)


def test_ensure_outside_git_rejects_repositories(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(SystemExit, match="repositorio"):
        common.ensure_outside_git(tmp_path / "salida")
    common.ensure_outside_git(tmp_path.parent / "fuera")  # no hay .git arriba de tmp_path.parent en la prueba


def test_parse_indices():
    assert common.parse_indices(None, 3, 10) == [0, 1, 2]
    assert common.parse_indices("4, 7", 3, 10) == [4, 7]
    with pytest.raises(SystemExit, match="fuera de la partición"):
        common.parse_indices("10", 3, 10)
    with pytest.raises(SystemExit):
        common.parse_indices(",", 3, 10)


def write_dataset(path: Path, total: int = 4) -> None:
    rng = np.random.default_rng(0)
    np.save(path / common.WAVEFORMS_FILE, rng.normal(size=(total, 1, 2500, 12)))
    np.save(path / common.TABULAR_FILE, rng.normal(size=(total, 7)))


def write_metadata(path: Path, splits: list[str]) -> None:
    columns = ["split", *make_sample.LABEL_COLUMNS]
    lines = [",".join(columns)]
    for position, split in enumerate(splits):
        lines.append(",".join([split, *[str(position % 2)] * len(make_sample.LABEL_COLUMNS)]))
    (path / common.METADATA_FILE).write_text("\n".join(lines), encoding="utf-8")


def test_load_studies_returns_float32_rows(tmp_path):
    write_dataset(tmp_path)
    waves, tabs = common.load_studies(tmp_path, [0, 3])
    assert waves.shape == (2, 1, 2500, 12) and waves.dtype == np.float32
    assert tabs.shape == (2, 7) and tabs.dtype == np.float32


def test_load_studies_rejects_wrong_shape(tmp_path):
    np.save(tmp_path / common.WAVEFORMS_FILE, np.zeros((2, 2500, 12)))
    np.save(tmp_path / common.TABULAR_FILE, np.zeros((2, 7)))
    with pytest.raises(SystemExit, match="Forma inesperada"):
        common.load_studies(tmp_path, [0])


def test_test_split_size_checks_metadata(tmp_path):
    write_dataset(tmp_path, total=3)
    write_metadata(tmp_path, ["train", "test", "test", "val", "test"])
    assert common.test_split_size(tmp_path) == 3
    write_metadata(tmp_path, ["train", "test", "val"])
    with pytest.raises(SystemExit, match="no son los archivos de prueba"):
        common.test_split_size(tmp_path)


def test_make_sample_keeps_only_waveform_and_tabular(tmp_path, monkeypatch):
    data = tmp_path / "datos"
    data.mkdir()
    write_dataset(data, total=2)
    write_metadata(data, ["train", "test", "test"])
    out = tmp_path / "salida"
    monkeypatch.setattr(sys, "argv", ["make_sample", "--data-dir", str(data), "--index", "1", "--out-dir", str(out)])
    assert make_sample.main() == 0
    with np.load(out / "sample_1.npz", allow_pickle=False) as sample:
        assert sorted(sample.files) == ["tabular", "waveform"]
        assert sample["waveform"].shape == (1, 2500, 12)
    labels = json.loads((out / "sample_1_labels.json").read_text(encoding="utf-8"))
    assert list(labels) == list(make_sample.LABEL_COLUMNS)


FAKE_ORIGINAL = """
import torch
from medicia_ml.echonext.model import EchoNextModel


class ResNet1dWithTabular(EchoNextModel):
    # Sustituto de prueba del original: misma interfaz (recibe una tupla), sin copiar su código.
    def __init__(self, len_tabular_feature_vector, filter_size, num_classes):
        super().__init__(num_tabular=len_tabular_feature_vector, num_classes=num_classes, filter_size=filter_size)

    def forward(self, x_and_tabular):
        return super().forward(*x_and_tabular)
"""


def test_check_parity_end_to_end_with_stand_in_original(tmp_path, weights_path, monkeypatch, capsys):
    import check_parity

    data = tmp_path / "datos"
    data.mkdir()
    write_dataset(data, total=3)
    write_metadata(data, ["test", "test", "test"])
    original = tmp_path / "original"
    (original / "cradlenet" / "models").mkdir(parents=True)
    (original / "cradlenet" / "models" / "resnet1d_tabular.py").write_text(FAKE_ORIGINAL, encoding="utf-8")
    model_dir = tmp_path / "modelo"
    model_dir.mkdir()
    (model_dir / "weights.pt").write_bytes(weights_path.read_bytes())
    reference = tmp_path / "ref" / "reference.json"
    monkeypatch.setattr(check_parity, "ORIGINAL_SHA256", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_parity", "--data-dir", str(data), "--model-dir", str(model_dir),
            "--original-dir", str(original), "--count", "3", "--reference-out", str(reference),
        ],
    )
    assert check_parity.main() == 0
    output = capsys.readouterr().out
    assert "todos dentro de la tolerancia" in output
    saved = json.loads(reference.read_text(encoding="utf-8"))
    assert saved["indices"] == [0, 1, 2] and len(saved["probabilities"]) == 3 and len(saved["probabilities"][0]) == 12


def test_check_parity_fails_when_original_is_missing(tmp_path, weights_path, monkeypatch):
    import check_parity

    with pytest.raises(SystemExit, match="No encuentro"):
        check_parity.load_original(tmp_path, weights_path)


def test_check_parity_refuses_an_original_that_does_not_match_the_pinned_file(tmp_path, weights_path):
    import check_parity

    (tmp_path / "cradlenet" / "models").mkdir(parents=True)
    (tmp_path / "cradlenet" / "models" / "resnet1d_tabular.py").write_text("raise SystemExit('no debería ejecutarse')", encoding="utf-8")
    with pytest.raises(SystemExit, match="no coincide"):
        check_parity.load_original(tmp_path, weights_path)

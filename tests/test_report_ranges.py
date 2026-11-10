"""Pruebas de report_ranges.py con datos sintéticos (nunca con datos reales)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import report_ranges  # noqa: E402


def write_split(directory: Path, split: str, studies: int = 6, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    waves = rng.normal(0, 1, size=(studies, 1, 2500, 12)).astype(np.float32)
    waves[0, 0, 0, 0] = 42.5  # extremo conocido
    waves[1, 0, 5, 3] = -17.25
    tab = rng.normal(0, 1, size=(studies, 7)).astype(np.float32)
    tab[:, 0] = rng.integers(0, 2, size=studies)
    np.save(directory / f"EchoNext_{split}_waveforms.npy", waves)
    np.save(directory / f"EchoNext_{split}_tabular_features.npy", tab)


def test_waveform_extremes_are_exact(tmp_path):
    write_split(tmp_path, "test")

    aggregates = report_ranges.waveform_aggregates(tmp_path / "EchoNext_test_waveforms.npy")

    assert aggregates["max"] == pytest.approx(42.5)
    assert aggregates["min"] == pytest.approx(-17.25)
    assert aggregates["p0.1"] < 0 < aggregates["p99.9"]


def test_tabular_aggregates_cover_the_seven_variables(tmp_path):
    write_split(tmp_path, "test")

    aggregates = report_ranges.tabular_aggregates(tmp_path / "EchoNext_test_tabular_features.npy")

    assert len(aggregates) == 7
    assert aggregates[0]["min"] == 0 and aggregates[0]["max"] == 1
    assert all(a["min"] <= a["p0.1"] <= a["p99.9"] <= a["max"] for a in aggregates)


def test_report_lists_available_splits_and_skips_the_rest(tmp_path):
    write_split(tmp_path, "test")
    write_split(tmp_path, "val", seed=2)

    text = "\n".join(report_ranges.report(tmp_path))

    assert "Partición test:" in text and "Partición val:" in text
    assert "Partición train: no está" in text
    assert "QT corregido" in text


def test_report_prints_only_aggregates(tmp_path, capsys):
    write_split(tmp_path, "test", studies=4)

    print("\n".join(report_ranges.report(tmp_path)))
    output = capsys.readouterr().out

    # Una fila por agregado: sin índices de estudios ni listas de valores.
    assert output.count("\n") < 20
    assert "[" not in output and "array" not in output
    assert not list(tmp_path.glob("*.txt")) and not list(tmp_path.glob("*.csv"))


def test_report_fails_without_any_split(tmp_path):
    with pytest.raises(SystemExit, match="ninguna partición"):
        report_ranges.report(tmp_path)


def test_rejects_unexpected_shapes(tmp_path):
    np.save(tmp_path / "EchoNext_test_waveforms.npy", np.zeros((2, 1, 100, 12), dtype=np.float32))
    np.save(tmp_path / "EchoNext_test_tabular_features.npy", np.zeros((2, 7), dtype=np.float32))

    with pytest.raises(SystemExit, match="Forma inesperada"):
        report_ranges.report(tmp_path)

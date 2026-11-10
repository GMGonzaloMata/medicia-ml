"""Prueba de invoke_endpoint.py contra el servidor en proceso, con datos sintéticos."""

import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import common  # noqa: E402
import invoke_endpoint  # noqa: E402

from medicia_ml.echonext.contract import FINDINGS  # noqa: E402
from medicia_ml.serve import make_server  # noqa: E402


@pytest.fixture
def server_url(predictor):
    server = make_server(predictor, host="127.0.0.1", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/invocations"
    server.shutdown()
    server.server_close()


def write_data(path: Path) -> None:
    rng = np.random.default_rng(0)
    np.save(path / common.WAVEFORMS_FILE, rng.normal(size=(3, 1, 2500, 12)))
    np.save(path / common.TABULAR_FILE, np.column_stack([np.ones(3), rng.normal(size=(3, 6))]))


def write_reference(path: Path, data: Path, predictor, shift: float = 0.0) -> Path:
    waves, tabs = common.load_studies(data, [0, 1])
    probabilities = [
        [min(1.0, p + shift) for p in predictor.predict(waves[i, 0], tabs[i]).values()] for i in range(2)
    ]
    reference = path / "reference.json"
    reference.write_text(
        json.dumps({"indices": [0, 1], "findings": list(FINDINGS), "probabilities": probabilities}), encoding="utf-8"
    )
    return reference


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["invoke_endpoint", *args])
    return invoke_endpoint.main()


def test_matches_reference(tmp_path, predictor, server_url, monkeypatch, capsys):
    write_data(tmp_path)
    reference = write_reference(tmp_path, tmp_path, predictor)
    code = run(monkeypatch, "--data-dir", str(tmp_path), "--reference", str(reference), "--url", server_url)
    output = capsys.readouterr().out
    assert code == 0
    assert "todo correcto" in output and "entrada inválida (6 variables): estado 400" in output


def test_detects_a_difference(tmp_path, predictor, server_url, monkeypatch, capsys):
    write_data(tmp_path)
    reference = write_reference(tmp_path, tmp_path, predictor, shift=-0.01)
    code = run(monkeypatch, "--data-dir", str(tmp_path), "--reference", str(reference), "--url", server_url)
    assert code == 1
    assert "FALLA" in capsys.readouterr().out


def test_requires_exactly_one_destination(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="exactamente uno"):
        run(monkeypatch, "--data-dir", str(tmp_path))
    with pytest.raises(SystemExit, match="exactamente uno"):
        run(monkeypatch, "--data-dir", str(tmp_path), "--url", "http://x", "--endpoint-name", "y")

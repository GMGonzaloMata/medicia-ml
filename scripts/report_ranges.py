"""Imprime SOLO agregados de los valores del conjunto EchoNext-Mini, para fijar los límites de validación de la 006.

Para cada partición disponible (entrenamiento, validación y prueba) imprime, de la onda y de cada una de las siete
variables tabulares: mínimo, máximo y los percentiles 0,1 y 99,9. No escribe nada a disco ni imprime filas, índices
ni valores de estudios individuales: nada que identifique a una persona. Los percentiles de la onda se calculan sobre
una muestra fija de valores (el mínimo y el máximo son exactos).

Con esos valores, la plataforma deja un margen de al menos el doble sobre el extremo observado (ver
specs/006-carga-estudios-ecg/research.md, R4).

Uso (desde la raíz de medicia-ml, con el entorno activado):
    python scripts/report_ranges.py --data-dir "$env:USERPROFILE\\Downloads"
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TABULAR_LENGTH, WAVEFORM_SHAPE  # noqa: E402

SPLITS = ("train", "val", "test")
TABULAR_NAMES = ("sexo", "edad", "frecuencia ventricular", "frecuencia auricular", "PR", "QRS", "QT corregido")
PERCENTILES = (0.1, 99.9)
CHUNK_STUDIES = 500
SAMPLE_VALUES_PER_CHUNK = 20_000
SEED = 20260610


def waveform_aggregates(path: Path) -> dict[str, float]:
    """Mínimo y máximo exactos; percentiles sobre una muestra fija. Lee por tandas (mapeo en disco)."""
    waveforms = np.load(path, mmap_mode="r", allow_pickle=False)
    if waveforms.shape[1:] != WAVEFORM_SHAPE:
        raise SystemExit(f"Forma inesperada de las ondas: {waveforms.shape[1:]} (se esperaba {WAVEFORM_SHAPE}).")

    rng = np.random.default_rng(SEED)
    low, high = np.inf, -np.inf
    samples: list[np.ndarray] = []
    for start in range(0, len(waveforms), CHUNK_STUDIES):
        chunk = np.asarray(waveforms[start : start + CHUNK_STUDIES], dtype=np.float32).ravel()
        low, high = min(low, float(chunk.min())), max(high, float(chunk.max()))
        samples.append(rng.choice(chunk, size=min(SAMPLE_VALUES_PER_CHUNK, chunk.size), replace=False))

    sample = np.concatenate(samples)
    low_p, high_p = np.percentile(sample, PERCENTILES)

    return {"min": low, "p0.1": float(low_p), "p99.9": float(high_p), "max": high}


def tabular_aggregates(path: Path) -> list[dict[str, float]]:
    tabular = np.load(path, mmap_mode="r", allow_pickle=False)
    if tabular.shape[1:] != (TABULAR_LENGTH,):
        raise SystemExit(f"Forma inesperada de las variables: {tabular.shape[1:]} (se esperaba ({TABULAR_LENGTH},)).")

    values = np.asarray(tabular, dtype=np.float32)
    result = []
    for column in range(TABULAR_LENGTH):
        data = values[:, column]
        low_p, high_p = np.percentile(data, PERCENTILES)
        result.append({"min": float(data.min()), "p0.1": float(low_p), "p99.9": float(high_p), "max": float(data.max())})

    return result


def format_row(label: str, aggregates: dict[str, float]) -> str:
    return (
        f"  {label:<24} min {aggregates['min']:>10.4f}   p0,1 {aggregates['p0.1']:>10.4f}   "
        f"p99,9 {aggregates['p99.9']:>10.4f}   max {aggregates['max']:>10.4f}"
    )


def report(data_dir: Path) -> list[str]:
    lines: list[str] = []
    found = 0
    for split in SPLITS:
        waveforms = data_dir / f"EchoNext_{split}_waveforms.npy"
        tabular = data_dir / f"EchoNext_{split}_tabular_features.npy"
        if not (waveforms.exists() and tabular.exists()):
            lines.append(f"Partición {split}: no está en --data-dir (se omite).")
            continue

        found += 1
        lines.append(f"Partición {split}:")
        lines.append(format_row("onda (todas las derivaciones)", waveform_aggregates(waveforms)))
        for name, aggregates in zip(TABULAR_NAMES, tabular_aggregates(tabular), strict=True):
            lines.append(format_row(name, aggregates))

    if found == 0:
        raise SystemExit(f"No encontré ninguna partición en {data_dir} (EchoNext_<train|val|test>_*.npy).")

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    print("\n".join(report(args.data_dir)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

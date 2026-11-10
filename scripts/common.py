"""Utilidades compartidas por los scripts de verificación.

Los scripts corren en la máquina del responsable. Leen el conjunto de datos desde --data-dir y nunca imprimen
datos de estudios individuales: solo índices, diferencias y estados. Las salidas se escriben fuera de cualquier
repositorio de git.
"""

import csv
from pathlib import Path

import numpy as np

WAVEFORMS_FILE = "EchoNext_test_waveforms.npy"
TABULAR_FILE = "EchoNext_test_tabular_features.npy"
METADATA_FILE = "echonext_metadata_100k.csv"

WAVEFORM_SHAPE = (1, 2500, 12)
TABULAR_LENGTH = 7


def ensure_outside_git(path: Path) -> None:
    """Falla si path está dentro de un repositorio de git: las salidas nunca se versionan."""
    resolved = path.resolve()
    for parent in [resolved, *resolved.parents]:
        if (parent / ".git").exists():
            raise SystemExit(
                f"La salida {resolved} está dentro del repositorio {parent}. "
                "Elegí una carpeta fuera de los repositorios (por ejemplo C:\\datos\\echonext)."
            )


def parse_indices(text: str | None, count: int, total: int) -> list[int]:
    """--indices '0,10,200' o, por defecto, los primeros `count` estudios."""
    if text:
        indices = [int(item) for item in text.split(",") if item.strip()]
    else:
        indices = list(range(count))
    if not indices:
        raise SystemExit("Hace falta al menos un índice.")
    bad = [i for i in indices if not 0 <= i < total]
    if bad:
        raise SystemExit(f"Índices fuera de la partición de prueba (0 a {total - 1}): {bad}")
    return indices


def load_studies(data_dir: Path, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Lee solo las filas pedidas (mapeo en disco). Devuelve ondas (n, 1, 2500, 12) y variables (n, 7), float32."""
    waveforms = np.load(data_dir / WAVEFORMS_FILE, mmap_mode="r", allow_pickle=False)
    tabular = np.load(data_dir / TABULAR_FILE, mmap_mode="r", allow_pickle=False)
    if waveforms.shape[1:] != WAVEFORM_SHAPE:
        raise SystemExit(f"Forma inesperada de las ondas: {waveforms.shape[1:]} (se esperaba {WAVEFORM_SHAPE}).")
    if tabular.shape[1:] != (TABULAR_LENGTH,):
        raise SystemExit(f"Forma inesperada de las variables: {tabular.shape[1:]} (se esperaba ({TABULAR_LENGTH},)).")
    if len(waveforms) != len(tabular):
        raise SystemExit("Las ondas y las variables tienen distinta cantidad de estudios.")
    return (
        np.stack([np.asarray(waveforms[i], dtype=np.float32) for i in indices]),
        np.stack([np.asarray(tabular[i], dtype=np.float32) for i in indices]),
    )


def test_split_size(data_dir: Path) -> int:
    """Cantidad de estudios de los arreglos. Si está el CSV de metadatos, comprueba que coincida con split == test."""
    total = len(np.load(data_dir / WAVEFORMS_FILE, mmap_mode="r", allow_pickle=False))
    metadata = data_dir / METADATA_FILE
    if metadata.exists():
        with metadata.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            if rows.fieldnames is None or "split" not in rows.fieldnames:
                raise SystemExit("El CSV de metadatos no tiene la columna 'split'.")
            in_test = sum(1 for row in rows if row["split"] == "test")
        if in_test != total:
            raise SystemExit(
                f"Los arreglos tienen {total} estudios pero el CSV marca {in_test} en la partición de prueba: "
                "no son los archivos de prueba."
            )
    return total

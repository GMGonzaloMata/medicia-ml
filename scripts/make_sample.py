"""Genera un .npz de demostración con UN estudio de la partición de prueba (para pruebas manuales y la 006).

El .npz contiene SOLO la onda y las 7 variables: sin identificadores, raza ni ubicación.
    waveform: float32, forma (1, 2500, 12), como viene en el conjunto de datos.
    tabular:  float32, forma (7,), en el orden del contrato.
Las etiquetas reales (12 indicadores) van en un archivo aparte, solo para la comparación fuera de la interfaz.
Las salidas se escriben fuera de los repositorios.

Uso:
    python scripts/make_sample.py --data-dir "$env:USERPROFILE\\Downloads" --index 0 --out-dir C:\\datos\\echonext\\samples
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import METADATA_FILE, ensure_outside_git, load_studies, parse_indices, test_split_size  # noqa: E402

# Columnas de etiquetas del CSV, en el orden de las 12 salidas del modelo.
LABEL_COLUMNS = (
    "lvef_lte_45_flag",
    "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag",
    "pasp_gte_45_flag",
    "tr_max_gte_32_flag",
    "shd_moderate_or_greater_flag",
)


def read_labels(metadata: Path, test_index: int) -> list[int]:
    with metadata.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in LABEL_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Faltan columnas de etiquetas en el CSV: {missing}")
        position = -1
        for row in reader:
            if row["split"] == "test":
                position += 1
                if position == test_index:
                    return [int(float(row[c])) for c in LABEL_COLUMNS]
    raise SystemExit("No se encontró el estudio en la partición de prueba.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0, help="Índice dentro de la partición de prueba.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    ensure_outside_git(args.out_dir)
    total = test_split_size(args.data_dir)
    parse_indices(str(args.index), 1, total)
    waves, tabs = load_studies(args.data_dir, [args.index])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample = args.out_dir / f"sample_{args.index}.npz"
    np.savez(sample, waveform=waves[0], tabular=tabs[0])
    print(f"Estudio de demostración: {sample}")

    metadata = args.data_dir / METADATA_FILE
    if metadata.exists():
        labels = args.out_dir / f"sample_{args.index}_labels.json"
        labels.write_text(
            json.dumps(dict(zip(LABEL_COLUMNS, read_labels(metadata, args.index), strict=True)), indent=2),
            encoding="utf-8",
        )
        print(f"Etiquetas reales (solo para comparar fuera de la interfaz): {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compara el modelo reescrito de medicia-ml con el código original de EchoNext-Mini (SC-001, FR-014).

El original se carga desde un clon aparte de PierreElias/IntroECG fijado por commit, importando solo
cradlenet/models/resnet1d_tabular.py (depende únicamente de torch). Nada del original se copia a este repositorio.

Uso (desde la raíz de medicia-ml, con el entorno activado):
    python scripts/check_parity.py ^
        --data-dir "$env:USERPROFILE\\Downloads" ^
        --model-dir C:\\datos\\echonext\\model ^
        --original-dir "C:\\datos\\echonext\\IntroECG\\7-EchoNext Minimodel" ^
        --count 5

Imprime solo índices y diferencias máximas. Guarda las probabilidades de referencia del original en
--reference-out (fuera de los repositorios), para que el contenedor local y el punto de inferencia las comparen.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ensure_outside_git, load_studies, parse_indices, test_split_size  # noqa: E402

from medicia_ml.echonext.contract import FINDINGS  # noqa: E402
from medicia_ml.echonext.predict import Predictor, load_model  # noqa: E402

TOLERANCE = 0.001


def load_original(original_dir: Path, weights_path: Path) -> torch.nn.Module:
    source = original_dir / "cradlenet" / "models" / "resnet1d_tabular.py"
    if not source.exists():
        raise SystemExit(f"No encuentro {source}. Revisá --original-dir (clon de IntroECG en el commit fijado).")
    spec = importlib.util.spec_from_file_location("echonext_original_resnet", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ResNet1dWithTabular(len_tabular_feature_vector=7, filter_size=16, num_classes=12)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def original_probabilities(model: torch.nn.Module, wave: np.ndarray, tab: np.ndarray) -> list[float]:
    with torch.no_grad():
        logits = model((torch.from_numpy(wave).unsqueeze(0), torch.from_numpy(tab).unsqueeze(0)))
        return torch.sigmoid(logits)[0].tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Carpeta con los archivos de la partición de prueba.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Carpeta con weights.pt.")
    parser.add_argument("--original-dir", type=Path, required=True, help="Carpeta '7-EchoNext Minimodel' del clon de IntroECG.")
    parser.add_argument("--count", type=int, default=5, help="Cantidad de estudios (los primeros), de 3 a 5.")
    parser.add_argument("--indices", help="Índices separados por coma, en lugar de los primeros.")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--reference-out", type=Path, default=Path(r"C:\datos\echonext\reference.json"))
    args = parser.parse_args()

    ensure_outside_git(args.reference_out)
    total = test_split_size(args.data_dir)
    indices = parse_indices(args.indices, args.count, total)
    waves, tabs = load_studies(args.data_dir, indices)

    weights = args.model_dir / "weights.pt"
    original = load_original(args.original_dir, weights)
    mine = Predictor(load_model(weights), "parity")

    reference: list[list[float]] = []
    failures = 0
    print(f"Partición de prueba: {total} estudios. Comparando {len(indices)} (tolerancia {args.tolerance}).")
    for position, index in enumerate(indices):
        expected = original_probabilities(original, waves[position], tabs[position])
        actual = mine.predict(waves[position, 0], tabs[position])
        diffs = [abs(expected[i] - actual[name]) for i, name in enumerate(FINDINGS)]
        worst = max(diffs)
        ok = worst < args.tolerance
        failures += 0 if ok else 1
        reference.append(expected)
        print(f"  estudio {index}: diferencia máxima {worst:.2e} -> {'OK' if ok else 'FALLA'}")

    args.reference_out.parent.mkdir(parents=True, exist_ok=True)
    args.reference_out.write_text(
        json.dumps({"indices": indices, "findings": list(FINDINGS), "probabilities": reference}), encoding="utf-8"
    )
    print(f"Referencia guardada en {args.reference_out}")
    print("RESULTADO:", "todos dentro de la tolerancia" if failures == 0 else f"{failures} estudio(s) fuera de la tolerancia")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

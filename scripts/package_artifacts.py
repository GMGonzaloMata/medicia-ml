"""Arma model.tar.gz con los artefactos de EchoNext-Mini (FR-008).

Contenido exacto, en la raíz del archivo (SageMaker lo descomprime en /opt/ml/model):
    weights.pt, tabular_transformer.joblib, waveform_normalization_params.json y MANIFEST.json.
Se niega a empaquetar si la carpeta tiene otros archivos o si falta alguno, y a escribir dentro de un repositorio.

Uso:
    python scripts/package_artifacts.py --model-dir C:\\datos\\echonext\\model --version 0.1.0 --out-dir C:\\datos\\echonext\\dist
"""

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ensure_outside_git  # noqa: E402

ARTIFACT_FILES = ("weights.pt", "tabular_transformer.joblib", "waveform_normalization_params.json")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def check_model_dir(model_dir: Path) -> None:
    present = {item.name for item in model_dir.iterdir()}
    missing = sorted(set(ARTIFACT_FILES) - present)
    extra = sorted(present - set(ARTIFACT_FILES))
    if missing:
        raise SystemExit(f"Faltan archivos en {model_dir}: {missing}")
    if extra:
        raise SystemExit(f"Hay archivos que no son artefactos del modelo en {model_dir}: {extra}. No se empaqueta.")


def build_archive(model_dir: Path, version: str, out_path: Path, code: str) -> str:
    """Escribe el archivo de forma determinista (mismos bytes con las mismas entradas) y devuelve su SHA-256."""
    check_model_dir(model_dir)
    manifest = {
        "model": "echonext-mini",
        "model_version": version,
        "code_version": code,
        "files": {name: sha256_of(model_dir / name) for name in ARTIFACT_FILES},
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    def add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mtime = 0
        info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        tar.addfile(info, io.BytesIO(data))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for name in ARTIFACT_FILES:
                add(tar, name, (model_dir / name).read_bytes())
            add(tar, "MANIFEST.json", manifest_bytes)
    return sha256_of(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--version", required=True, help="Versión del modelo, por ejemplo 0.1.0.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    ensure_outside_git(args.out_dir)
    out_path = args.out_dir / "model.tar.gz"
    digest = build_archive(args.model_dir, args.version, out_path, code_version())
    print(f"Archivo: {out_path}")
    print(f"SHA-256: {digest}")
    print(f"Clave sugerida en el bucket: echonext-mini/{args.version}/model.tar.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

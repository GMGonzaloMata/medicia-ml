"""Genera archivos .npz SINTÉTICOS para probar la carga de estudios de la plataforma (006), sin leer ningún dato.

Todo se inventa con un generador aleatorio de semilla fija: la onda son latidos gaussianos repetidos con ruido y
las variables son valores plausibles ya estandarizados. No se abre ni se usa ningún archivo del conjunto EchoNext.

Escribe `valido.npz` y los archivos defectuosos que la plataforma debe rechazar o aceptar:
    sin-tabular.npz       falta el campo `tabular`            -> FIELD_MISSING
    forma-incorrecta.npz  la onda tiene 11 derivaciones        -> FIELD_SHAPE
    con-nan.npz           un NaN en la onda                    -> VALUE_NOT_FINITE
    sexo-2.npz            el sexo vale 2                       -> SEX_INVALID
    sin-estandarizar.npz  una variable sin estandarizar (450)  -> VALUE_OUT_OF_RANGE
    con-objetos.npz       un campo de objetos (`pickle`)       -> FIELD_TYPE
    con-extras.npz        campos extra: se acepta y se ignoran
    grande.npz            más de 2 MB                          -> FILE_TOO_LARGE
Las salidas se escriben fuera de los repositorios.

Uso:
    python scripts/make_synthetic.py --out-dir C:\\datos\\echonext\\synthetic
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ensure_outside_git  # noqa: E402

SAMPLES = 2500
LEADS = 12
SAMPLING_HZ = 250
BEAT_SECONDS = 0.8
SEED = 20260610
# Más de 2 MB sin comprimir: 600.000 valores float32 son 2,4 MB.
OVERSIZE_VALUES = 600_000


def synthetic_waveform(rng: np.random.Generator) -> np.ndarray:
    """Latidos gaussianos repetidos más ruido, forma (1, 2500, 12), float32."""
    time = np.arange(SAMPLES) / SAMPLING_HZ
    beat_phase = (time % BEAT_SECONDS) - BEAT_SECONDS / 2
    beat = np.exp(-(beat_phase**2) / (2 * 0.02**2))
    gains = rng.uniform(0.5, 1.5, size=LEADS) * rng.choice([-1.0, 1.0], size=LEADS, p=[0.2, 0.8])
    wave = beat[:, None] * gains[None, :] + rng.normal(0, 0.05, size=(SAMPLES, LEADS))

    return wave[None, :, :].astype(np.float32)


def synthetic_tabular(rng: np.random.Generator) -> np.ndarray:
    """Sexo (0/1) y seis variables estandarizadas plausibles, forma (7,), float32."""
    return np.concatenate([[1.0], rng.normal(0, 0.8, size=6)]).astype(np.float32)


def build_files(out_dir: Path) -> list[Path]:
    rng = np.random.default_rng(SEED)
    wave = synthetic_waveform(rng)
    tab = synthetic_tabular(rng)
    written: list[Path] = []

    def save(name: str, **arrays: np.ndarray) -> None:
        path = out_dir / name
        np.savez(path, **arrays)
        written.append(path)

    save("valido.npz", waveform=wave, tabular=tab)
    save("sin-tabular.npz", waveform=wave)
    save("forma-incorrecta.npz", waveform=wave[:, :, :11], tabular=tab)

    with_nan = wave.copy()
    with_nan[0, 100, 3] = np.nan
    save("con-nan.npz", waveform=with_nan, tabular=tab)

    bad_sex = tab.copy()
    bad_sex[0] = 2.0
    save("sexo-2.npz", waveform=wave, tabular=bad_sex)

    unstandardized = tab.copy()
    unstandardized[1] = 450.0
    save("sin-estandarizar.npz", waveform=wave, tabular=unstandardized)

    save("con-objetos.npz", waveform=wave, tabular=np.array([None] * 7, dtype=object))
    save("con-extras.npz", waveform=wave, tabular=tab, notas=np.arange(5, dtype=np.float32), extra=np.zeros(3))
    save(
        "grande.npz",
        waveform=wave,
        tabular=tab,
        relleno=rng.normal(size=OVERSIZE_VALUES).astype(np.float32),
    )

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    ensure_outside_git(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in build_files(args.out_dir):
        print(f"{path.name}: {path.stat().st_size} bytes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

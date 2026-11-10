"""Validación de la solicitud. Los mensajes nunca repiten valores del estudio."""

import json
import math

import numpy as np

from medicia_ml.echonext.contract import TABULAR_FEATURES, WAVEFORM_LEADS, WAVEFORM_SAMPLES


class InputError(ValueError):
    """Entrada inválida: se responde 400 con este mensaje."""


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_request(body: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (onda de forma (2500, 12), variables de forma (7,)), ambas float32 y finitas."""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError, RecursionError):
        # RecursionError: un JSON anidado miles de veces cabe en el tope de tamaño y rompe el analizador.
        raise InputError("El cuerpo no es un JSON válido.") from None
    if not isinstance(payload, dict):
        raise InputError("El cuerpo debe ser un objeto JSON con 'waveform' y 'tabular'.")
    for field in ("waveform", "tabular"):
        if field not in payload:
            raise InputError(f"Falta el campo '{field}'.")

    waveform = payload["waveform"]
    if (
        not isinstance(waveform, list)
        or len(waveform) != WAVEFORM_SAMPLES
        or any(not isinstance(row, list) or len(row) != WAVEFORM_LEADS for row in waveform)
    ):
        raise InputError(f"'waveform' debe ser una lista de {WAVEFORM_SAMPLES} filas de {WAVEFORM_LEADS} números.")
    if any(not _is_number(v) for row in waveform for v in row):
        raise InputError("'waveform' solo admite números.")

    tabular = payload["tabular"]
    if not isinstance(tabular, list) or len(tabular) != len(TABULAR_FEATURES):
        raise InputError(f"'tabular' debe ser una lista de {len(TABULAR_FEATURES)} números.")
    if any(not _is_number(v) for v in tabular):
        raise InputError("'tabular' solo admite números.")
    if any(isinstance(v, float) and not math.isfinite(v) for v in tabular):
        raise InputError("'tabular' contiene valores no finitos.")
    if tabular[0] not in (0, 1):
        raise InputError("El sexo (primera variable de 'tabular') debe ser 0 o 1.")

    try:
        wave_array = np.asarray(waveform, dtype=np.float32)
        tab_array = np.asarray(tabular, dtype=np.float32)
    except (OverflowError, ValueError):
        # Un entero con cientos de dígitos pasa como número pero no entra en float32.
        raise InputError("Hay valores numéricos fuera de rango.") from None
    if not np.isfinite(wave_array).all():
        raise InputError("'waveform' contiene valores no finitos.")
    if not np.isfinite(tab_array).all():
        raise InputError("'tabular' contiene valores no finitos.")
    return wave_array, tab_array

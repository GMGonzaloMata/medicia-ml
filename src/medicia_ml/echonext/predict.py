"""Carga de los pesos y predicción."""

from pathlib import Path

import numpy as np
import torch

from medicia_ml.echonext.contract import FINDINGS, MODEL_NAME
from medicia_ml.echonext.model import EchoNextModel


def load_model(weights_path: Path) -> EchoNextModel:
    """Carga weights.pt (diccionario con la clave "model") en modo evaluación.

    weights_only=True evita ejecutar código al deserializar. strict=True falla si cambia algún nombre o forma.
    """
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = EchoNextModel()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


class Predictor:
    def __init__(self, model: EchoNextModel, model_version: str):
        self._model = model
        self.model_version = model_version

    def predict(self, waveform: np.ndarray, tabular: np.ndarray) -> dict[str, float]:
        """waveform (2500, 12) y tabular (7,), float32. Devuelve las 12 probabilidades en orden fijo."""
        with torch.no_grad():
            wave = torch.from_numpy(waveform).reshape(1, 1, *waveform.shape)
            tab = torch.from_numpy(tabular).reshape(1, -1)
            probabilities = torch.sigmoid(self._model(wave, tab))[0].tolist()
        return dict(zip(FINDINGS, probabilities, strict=True))

    def response(self, waveform: np.ndarray, tabular: np.ndarray) -> dict[str, object]:
        return {
            "model": MODEL_NAME,
            "model_version": self.model_version,
            "probabilities": self.predict(waveform, tabular),
        }

import json

import numpy as np

from medicia_ml.echonext.contract import TABULAR_FEATURES, WAVEFORM_LEADS, WAVEFORM_SAMPLES


def valid_payload(seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "waveform": rng.normal(size=(WAVEFORM_SAMPLES, WAVEFORM_LEADS)).tolist(),
        "tabular": [1, *rng.normal(size=len(TABULAR_FEATURES) - 1).tolist()],
    }


def encode(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")

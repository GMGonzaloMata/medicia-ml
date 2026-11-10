import numpy as np
import pytest
import torch

from medicia_ml.echonext.contract import FINDINGS
from medicia_ml.echonext.model import EchoNextModel
from medicia_ml.echonext.predict import load_model

# Forma de cada tensor del state_dict de los pesos publicados (4,27 MB): filter_size 16, 7 variables, 12 salidas.
EXPECTED_SHAPES = {
    "conv1.weight": (16, 12, 15),
    "layer1.0.conv1.weight": (16, 16, 7),
    "layer2.0.conv1.weight": (32, 16, 7),
    "layer2.0.downsample.0.weight": (32, 16, 1),
    "layer3.5.conv2.weight": (64, 64, 7),
    "layer4.0.downsample.1.weight": (128, 1),
    "output.weight": (12, 263),
    "output.bias": (12,),
}


def test_state_dict_names_and_shapes():
    state = EchoNextModel().state_dict()
    for name, shape in EXPECTED_SHAPES.items():
        if name == "layer4.0.downsample.1.weight":
            assert tuple(state[name].shape) == (128,)
        else:
            assert tuple(state[name].shape) == shape, name
    # Etapas [3, 4, 6, 3]
    for layer, blocks in {"layer1": 3, "layer2": 4, "layer3": 6, "layer4": 3}.items():
        assert f"{layer}.{blocks - 1}.conv2.weight" in state
        assert f"{layer}.{blocks}.conv2.weight" not in state


def test_output_shape_and_range(predictor):
    rng = np.random.default_rng(0)
    result = predictor.predict(
        rng.normal(size=(2500, 12)).astype(np.float32),
        np.array([1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
    )
    assert tuple(result) == FINDINGS
    assert all(0.0 <= p <= 1.0 for p in result.values())


def test_same_input_same_output(predictor):
    rng = np.random.default_rng(3)
    wave = rng.normal(size=(2500, 12)).astype(np.float32)
    tab = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.float32)
    assert predictor.predict(wave, tab) == predictor.predict(wave, tab)


def test_load_model_is_in_eval_mode(weights_path):
    assert load_model(weights_path).training is False


def test_load_rejects_wrong_names(tmp_path):
    state = EchoNextModel().state_dict()
    state["capa_inexistente.weight"] = torch.zeros(1)
    path = tmp_path / "malo.pt"
    torch.save({"model": state}, path)
    with pytest.raises(RuntimeError):
        load_model(path)


def test_load_rejects_missing_key(tmp_path):
    path = tmp_path / "sin_clave.pt"
    torch.save({"otra": EchoNextModel().state_dict()}, path)
    with pytest.raises(KeyError):
        load_model(path)

import pytest
import torch

from medicia_ml.echonext.model import EchoNextModel
from medicia_ml.echonext.predict import Predictor, load_model


@pytest.fixture(scope="session")
def weights_path(tmp_path_factory):
    """Pesos aleatorios con el mismo formato que weights.pt: un diccionario con la clave "model"."""
    torch.manual_seed(0)
    model = EchoNextModel()
    path = tmp_path_factory.mktemp("pesos") / "weights.pt"
    torch.save({"model": model.state_dict()}, path)
    return path


@pytest.fixture(scope="session")
def predictor(weights_path):
    return Predictor(load_model(weights_path), "test")

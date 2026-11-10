import copy
import json

import pytest

from tests.helpers import encode, valid_payload
from medicia_ml.echonext.validation import InputError, parse_request


def test_valid_request_parses():
    waveform, tabular = parse_request(encode(valid_payload()))
    assert waveform.shape == (2500, 12)
    assert tabular.shape == (7,)
    assert waveform.dtype.name == "float32"


def _mutated(mutate):
    payload = copy.deepcopy(valid_payload())
    mutate(payload)
    return payload


CASES = {
    "onda con menos filas": lambda p: p.update(waveform=p["waveform"][:2499]),
    "onda transpuesta": lambda p: p.update(waveform=[list(col) for col in zip(*p["waveform"])]),
    "onda con fila corta": lambda p: p["waveform"][10].pop(),
    "onda no es lista": lambda p: p.update(waveform="hola"),
    "onda con texto": lambda p: p["waveform"][3].__setitem__(2, "1.5"),
    "onda con booleano": lambda p: p["waveform"][3].__setitem__(2, True),
    "onda con nulo": lambda p: p["waveform"][3].__setitem__(2, None),
    "onda con NaN": lambda p: p["waveform"][3].__setitem__(2, float("nan")),
    "onda con infinito": lambda p: p["waveform"][3].__setitem__(2, float("inf")),
    "onda fuera de rango float32": lambda p: p["waveform"][3].__setitem__(2, 1e300),
    "seis variables": lambda p: p["tabular"].pop(),
    "ocho variables": lambda p: p["tabular"].append(0.0),
    "variables no es lista": lambda p: p.update(tabular=5),
    "variable con texto": lambda p: p["tabular"].__setitem__(2, "x"),
    "variable con NaN": lambda p: p["tabular"].__setitem__(2, float("nan")),
    "variable con infinito": lambda p: p["tabular"].__setitem__(2, float("-inf")),
    "sexo 2": lambda p: p["tabular"].__setitem__(0, 2),
    "sexo 0,5": lambda p: p["tabular"].__setitem__(0, 0.5),
    "sexo negativo": lambda p: p["tabular"].__setitem__(0, -1),
    "falta la onda": lambda p: p.pop("waveform"),
    "faltan las variables": lambda p: p.pop("tabular"),
}


@pytest.mark.parametrize("name", CASES)
def test_invalid_input_is_rejected_without_leaking_values(name):
    body = json.dumps(_mutated(CASES[name])).encode("utf-8")
    with pytest.raises(InputError) as error:
        parse_request(body)
    message = str(error.value)
    assert message
    assert "1e+300" not in message and "nan" not in message.lower().replace("finit", "")


def test_body_not_json():
    with pytest.raises(InputError, match="JSON"):
        parse_request(b"no es json")


def test_body_not_utf8():
    with pytest.raises(InputError, match="JSON"):
        parse_request(b"\xff\xfe")


def test_body_not_an_object():
    with pytest.raises(InputError, match="objeto"):
        parse_request(b"[1, 2, 3]")


def test_sex_accepts_float_zero_and_one():
    payload = valid_payload()
    payload["tabular"][0] = 0.0
    parse_request(encode(payload))
    payload["tabular"][0] = 1.0
    parse_request(encode(payload))


def test_deeply_nested_json_is_rejected_as_invalid_input():
    body = b'{"waveform": ' + b"[" * 100000 + b"]" * 100000 + b', "tabular": []}'
    with pytest.raises(InputError, match="JSON"):
        parse_request(body)


def test_huge_integer_is_rejected_as_out_of_range():
    payload = valid_payload()
    text = json.dumps(payload).replace(json.dumps(payload["waveform"][0][0]), "9" * 400, 1)
    with pytest.raises(InputError, match="fuera de rango"):
        parse_request(text.encode("utf-8"))

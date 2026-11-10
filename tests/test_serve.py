import json
import logging
import threading
import urllib.error
import urllib.request

import pytest

from tests.helpers import encode, valid_payload
from medicia_ml.echonext.contract import FINDINGS, MAX_BODY_BYTES
from medicia_ml.serve import make_server


@pytest.fixture(scope="module")
def base_url(predictor):
    server = make_server(predictor, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def call(url, method="POST", body=None, content_type="application/json"):
    headers = {"Content-Type": content_type} if content_type else {}
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_ping(base_url):
    assert call(f"{base_url}/ping", method="GET") == (200, b"")


def test_unknown_route(base_url):
    assert call(f"{base_url}/otra", method="GET")[0] == 404
    assert call(f"{base_url}/otra", body=b"{}")[0] == 404


def test_valid_invocation_has_exact_format(base_url):
    status, raw = call(f"{base_url}/invocations", body=encode(valid_payload()))
    assert status == 200
    data = json.loads(raw)
    assert list(data) == ["model", "model_version", "probabilities"]
    assert data["model"] == "echonext-mini"
    assert data["model_version"] == "test"
    assert list(data["probabilities"]) == list(FINDINGS)
    assert all(0.0 <= p <= 1.0 for p in data["probabilities"].values())


def test_same_study_twice_gives_same_answer(base_url):
    body = encode(valid_payload(5))
    assert call(f"{base_url}/invocations", body=body) == call(f"{base_url}/invocations", body=body)


def test_invalid_input_returns_400_without_probabilities(base_url):
    payload = valid_payload()
    payload["tabular"].pop()
    status, raw = call(f"{base_url}/invocations", body=encode(payload))
    assert status == 400
    data = json.loads(raw)
    assert "probabilities" not in data and data["error"]


def test_body_too_large_returns_413(base_url):
    status, raw = call(f"{base_url}/invocations", body=b" " * (MAX_BODY_BYTES + 1))
    assert status == 413
    assert "probabilities" not in json.loads(raw)


def test_wrong_content_type_returns_415(base_url):
    status, _ = call(f"{base_url}/invocations", body=encode(valid_payload()), content_type="text/csv")
    assert status == 415


def test_internal_error_returns_500_without_details(predictor, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("detalle secreto del estudio 123")

    monkeypatch.setattr(predictor, "response", boom)
    server = make_server(predictor, host="127.0.0.1", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, raw = call(f"http://127.0.0.1:{server.server_address[1]}/invocations", body=encode(valid_payload()))
    finally:
        server.shutdown()
        server.server_close()
    assert status == 500
    assert json.loads(raw) == {"error": "error interno"}


def test_logs_do_not_contain_study_values(base_url, caplog):
    payload = valid_payload(9)
    marker = repr(payload["tabular"][1])[:8]
    with caplog.at_level(logging.INFO, logger="medicia_ml"):
        call(f"{base_url}/invocations", body=encode(payload))
    assert marker not in caplog.text


def test_nested_json_and_huge_integer_return_400(base_url):
    nested = b'{"waveform": ' + b"[" * 100000 + b"]" * 100000 + b', "tabular": []}'
    assert call(f"{base_url}/invocations", body=nested)[0] == 400
    text = json.dumps(valid_payload()).replace(json.dumps(valid_payload()["waveform"][0][0]), "9" * 400, 1)
    assert call(f"{base_url}/invocations", body=text.encode("utf-8"))[0] == 400

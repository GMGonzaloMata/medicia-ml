"""Servidor de inferencia para SageMaker: GET /ping y POST /invocations en el puerto 8080."""

import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from medicia_ml.echonext.contract import MAX_BODY_BYTES
from medicia_ml.echonext.predict import Predictor, load_model
from medicia_ml.echonext.validation import InputError, parse_request

logger = logging.getLogger("medicia_ml")


def make_server(predictor: Predictor, host: str = "0.0.0.0", port: int = 8080) -> ThreadingHTTPServer:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        # Tiempo máximo de espera de lectura: un cliente lento no ocupa un hilo indefinidamente.
        timeout = 30

        def _send(self, status: int, payload: dict[str, object] | None = None) -> None:
            body = b"" if payload is None else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            if payload is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (nombre fijado por http.server)
            if self.path == "/ping":
                self._send(200)
            else:
                self._send(404, {"error": "ruta no encontrada"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/invocations":
                self._send(404, {"error": "ruta no encontrada"})
                return
            if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
                self._send(415, {"error": "Content-Type debe ser application/json."})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send(400, {"error": "Falta Content-Length."})
                return
            if length < 0 or length > MAX_BODY_BYTES:
                self._send(413, {"error": "El cuerpo supera el tamaño máximo."})
                return
            body = self.rfile.read(length)
            try:
                waveform, tabular = parse_request(body)
            except InputError as error:
                self._send(400, {"error": str(error)})
                return
            try:
                with lock:
                    result = predictor.response(waveform, tabular)
            except Exception:  # noqa: BLE001 (nunca se devuelven detalles ni datos del estudio)
                logger.exception("Error interno al predecir")
                self._send(500, {"error": "error interno"})
                return
            self._send(200, result)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            # Solo método y ruta: nunca el contenido de la solicitud.
            logger.info("%s %s", getattr(self, "command", "-"), getattr(self, "path", "-"))

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model_dir = Path(os.environ.get("MODEL_DIR", "/opt/ml/model"))
    version = os.environ.get("MODEL_VERSION", "unknown")
    try:
        model = load_model(model_dir / "weights.pt")
    except Exception:
        logger.exception("No se pudo cargar el modelo")
        sys.exit(1)
    server = make_server(Predictor(model, version), port=int(os.environ.get("PORT", "8080")))
    logger.info("Modelo %s cargado; escuchando", version)
    server.serve_forever()


if __name__ == "__main__":
    # SageMaker agrega el argumento "serve": se ignora.
    main()

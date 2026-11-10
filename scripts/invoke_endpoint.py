"""Envía estudios al modelo y los compara con la referencia del original (SC-001, SC-002, FR-014).

Dos destinos:
  --url http://127.0.0.1:8080/invocations   contenedor local (docker run -p 8080:8080 ...)
  --endpoint-name medicia-dev-echonext-mini punto de inferencia de SageMaker (necesita boto3 y credenciales)

Lee los estudios de la partición de prueba desde --data-dir, usando los índices que dejó check_parity.py en
--reference (fuera de los repositorios). Prueba además una entrada inválida (debe dar 400, sin probabilidades).
Imprime solo índices, diferencias y estados: nunca datos de estudios.

Uso:
    python scripts/invoke_endpoint.py --data-dir "$env:USERPROFILE\\Downloads" --url http://127.0.0.1:8080/invocations
    python scripts/invoke_endpoint.py --data-dir "$env:USERPROFILE\\Downloads" --endpoint-name medicia-dev-echonext-mini
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_studies  # noqa: E402

from medicia_ml.echonext.contract import FINDINGS  # noqa: E402

TOLERANCE = 0.001


def build_payload(wave, tab) -> bytes:
    return json.dumps({"waveform": wave[0].tolist(), "tabular": tab.tolist()}).encode("utf-8")


def post_url(url: str, body: bytes) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def make_sagemaker_caller(endpoint_name: str, region: str):
    import boto3
    from botocore.exceptions import ClientError

    runtime = boto3.client("sagemaker-runtime", region_name=region)

    def call(body: bytes) -> tuple[int, dict]:
        try:
            response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType="application/json", Body=body)
            return 200, json.loads(response["Body"].read())
        except ClientError as error:
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            return status, {"error": error.response.get("Error", {}).get("Code", "error")}

    return call


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path(r"C:\datos\echonext\reference.json"))
    parser.add_argument("--url", help="URL de /invocations del contenedor local.")
    parser.add_argument("--endpoint-name", help="Nombre del punto de inferencia de SageMaker.")
    parser.add_argument("--region", default="sa-east-1")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()

    if bool(args.url) == bool(args.endpoint_name):
        raise SystemExit("Indicá exactamente uno: --url o --endpoint-name.")
    caller = (lambda body: post_url(args.url, body)) if args.url else make_sagemaker_caller(args.endpoint_name, args.region)

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if reference["findings"] != list(FINDINGS):
        raise SystemExit("La referencia tiene otro orden de salidas que el contrato.")
    indices = reference["indices"]
    waves, tabs = load_studies(args.data_dir, indices)

    failures = 0
    print(f"Comparando {len(indices)} estudios contra la referencia (tolerancia {args.tolerance}).")
    for position, index in enumerate(indices):
        status, answer = caller(build_payload(waves[position], tabs[position]))
        if status != 200 or list(answer.get("probabilities", {})) != list(FINDINGS):
            print(f"  estudio {index}: estado {status}, respuesta inesperada -> FALLA")
            failures += 1
            continue
        worst = max(abs(answer["probabilities"][name] - reference["probabilities"][position][i]) for i, name in enumerate(FINDINGS))
        ok = worst < args.tolerance
        failures += 0 if ok else 1
        print(f"  estudio {index}: diferencia máxima {worst:.2e} -> {'OK' if ok else 'FALLA'}")

    # Entrada inválida: seis variables en lugar de siete. Debe rechazarse sin devolver probabilidades.
    bad = json.dumps({"waveform": waves[0][0].tolist(), "tabular": tabs[0].tolist()[:-1]}).encode("utf-8")
    status, answer = caller(bad)
    rejected = status in (400, 424) and "probabilities" not in answer
    failures += 0 if rejected else 1
    print(f"  entrada inválida (6 variables): estado {status}, sin probabilidades: {'probabilities' not in answer} -> {'OK' if rejected else 'FALLA'}")

    print("RESULTADO:", "todo correcto" if failures == 0 else f"{failures} comprobación(es) fallaron")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

# Imagen de inferencia de EchoNext-Mini. Sin acceso a internet en ejecución: todo se instala al construir.
# Construir:  docker build --platform linux/amd64 --build-arg MODEL_VERSION=0.1.0 -t medicia-dev-echonext-mini:0.1.0 .
FROM python:3.11-slim-bookworm

ARG MODEL_VERSION=unknown
ENV MODEL_VERSION=${MODEL_VERSION} \
    MODEL_DIR=/opt/ml/model \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.lock .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock

COPY src ./src

RUN useradd --system --no-create-home appuser
USER appuser

EXPOSE 8080
# SageMaker ejecuta "docker run <imagen> serve": el argumento se agrega al ENTRYPOINT y se ignora.
ENTRYPOINT ["python", "-m", "medicia_ml.serve"]

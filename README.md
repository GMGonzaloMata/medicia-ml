# medicia-ml

Código de inferencia de los modelos de MedicIA, separado del repositorio principal (`medicia`) porque la constitución del proyecto exige que el código Python y PyTorch de los modelos viva fuera de él. Hoy contiene un solo modelo: **EchoNext-Mini**.

El repositorio principal solo referencia lo que se produce acá por versión exacta: la imagen de inferencia (ECR) y el artefacto del modelo (S3).

## Qué hace EchoNext-Mini

Recibe un electrocardiograma de 12 derivaciones ya preprocesado (onda de 2500 × 12 a 250 Hz y 7 variables tabulares) y devuelve **12 probabilidades** de posibles hallazgos de cardiopatía estructural, en un orden fijo. El contrato completo está en `specs/005-echonext-mini-sagemaker/contracts/inference-api.md` del repositorio principal.

## Aviso importante

- Las probabilidades son un **indicador relativo de riesgo de un posible hallazgo**, de apoyo clínico. **Nunca son un diagnóstico**, no hay umbrales publicados y pueden no estar calibradas.
- Es un modelo de investigación: se entrenó con adultos en un solo centro (NewYork-Presbyterian/Columbia), excluye ECG con marcapasos ventricular y de calidad pobre, tiene una sensibilidad reportada de 70,1 % y un AUROC externo reportado de 79,0 (MIMIC-IV).
- **Licencia:** el repositorio original (`PierreElias/IntroECG`) no declara licencia, Columbia tiene una solicitud de patente sobre el algoritmo y los datos están bajo la licencia de PhysioNet de datos de salud restringidos. El uso actual es **académico**. Cualquier uso comercial exige revisar todo eso antes.
- La arquitectura de este repositorio está **reescrita de forma independiente**; no se copia código del original. Se verifica contra el original con `scripts/check_parity.py`.

## Regla: nada de datos ni de pesos en este repositorio

Tres barreras: `.gitignore`; una comprobación de CI (`.github/workflows/guard.yml`) que falla si algo así llega a una propuesta de cambio, aunque se haya forzado con `git add -f`; y un gancho local opcional que se activa una vez por clon con `git config core.hooksPath .githooks`.

Los pesos (`*.pt`), los arreglos (`*.npy`, `*.npz`), el transformador (`*.joblib`), las tablas (`*.csv`) y los artefactos (`*.tar.gz`) **no se versionan nunca**; `.gitignore` los bloquea. El conjunto de datos EchoNext-Mini se usa desde la máquina del responsable y se pasa a los scripts con `--data-dir`.

## Versiones elegidas

- Python 3.11.
- PyTorch solo CPU, 2.15.0 (posterior a 2.6, que corrige el problema de seguridad de `torch.load`). La versión exacta queda en `requirements.lock`.
- NumPy 2.4.6.

## Desarrollo

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"
pytest --cov=medicia_ml --cov-fail-under=80
```

## Scripts de verificación

Corren en tu máquina y leen el conjunto de datos desde `--data-dir` (nunca desde este repositorio). Imprimen solo índices, diferencias y estados: nunca datos de estudios. Se niegan a escribir sus salidas dentro de un repositorio de git.

- `scripts/check_parity.py`: compara el modelo de este repositorio con el original (un clon aparte de `PierreElias/IntroECG` en el commit `15233e9392e9dc10c136a5624abf10199207bce9`, que no se copia acá) sobre los primeros N estudios de la partición de prueba. Guarda la referencia en `--reference-out`.
- `scripts/package_artifacts.py`: arma `model.tar.gz` (pesos, transformador de variables, parámetros de normalización y un manifiesto con huellas SHA-256). Se niega a empaquetar si sobra o falta algún archivo.
- `scripts/invoke_endpoint.py`: envía estudios al contenedor local (`--url`) o al punto de inferencia de SageMaker (`--endpoint-name`, necesita `pip install -e ".[verify]"` y el perfil de AWS) y los compara con la referencia. Prueba también una entrada inválida.
- `scripts/make_sample.py`: genera un `.npz` de demostración con solo la onda y las 7 variables de un estudio de prueba, y sus etiquetas reales en un archivo aparte.

## Construir y probar la imagen

```powershell
docker build --platform linux/amd64 --provenance=false --sbom=false --build-arg MODEL_VERSION=0.1.0 -t medicia-dev-echonext-mini:0.1.0 .
docker run -d --name eco-local -p 127.0.0.1:8080:8080 -v C:\datos\echonext\model:/opt/ml/model:ro medicia-dev-echonext-mini:0.1.0 serve
```

`--provenance=false --sbom=false` es obligatorio: sin eso, Docker sube un índice de manifiestos en lugar de una imagen simple. Para subirla a ECR y encender el modelo, ver `infra/README.md`, sección "Modelo EchoNext-Mini", del repositorio `medicia`.

## Qué contiene el artefacto

`tabular_transformer.joblib` es un archivo `pickle` y viaja en el artefacto solo para que la plataforma (006) lo use en otro lado. **El servidor no lo carga nunca**: cargar un `pickle` de origen desconocido ejecuta código. Nada de este repositorio debe usar `joblib.load` ni `pickle.load` sobre los artefactos.

## Contrato

La entrada, la salida y los errores están en `specs/005-echonext-mini-sagemaker/contracts/inference-api.md` del repositorio `medicia`. Por `InvokeEndpoint` de SageMaker, un 400 del contenedor llega como error 424.

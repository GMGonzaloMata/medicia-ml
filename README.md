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

"""Contrato de entrada y salida de EchoNext-Mini.

Es la única fuente del orden de las variables y de las salidas. Debe coincidir con
specs/005-echonext-mini-sagemaker/contracts/inference-api.md del repositorio medicia.
"""

MODEL_NAME = "echonext-mini"

# Onda ya filtrada y normalizada: 2500 muestras (250 Hz, 10 s) por 12 derivaciones.
WAVEFORM_SAMPLES = 2500
WAVEFORM_LEADS = 12

# Orden fijo de las 7 variables tabulares. Todas estandarizadas salvo el sexo (0 mujer, 1 hombre).
TABULAR_FEATURES = (
    "sex",
    "age",
    "ventricular_rate",
    "atrial_rate",
    "pr_interval",
    "qrs_duration",
    "qt_corrected",
)

# Orden fijo de las 12 salidas. La última es el hallazgo compuesto.
FINDINGS = (
    "lvef_lte_45",
    "lvwt_gte_13",
    "aortic_stenosis",
    "aortic_regurgitation",
    "mitral_regurgitation",
    "tricuspid_regurgitation",
    "pulmonary_regurgitation",
    "rv_systolic_dysfunction",
    "pericardial_effusion",
    "pasp_gte_45",
    "tr_max_gte_32",
    "shd",
)

# Tope del cuerpo de una solicitud. La típica pesa cerca de 0,6 MB.
MAX_BODY_BYTES = 2 * 1024 * 1024

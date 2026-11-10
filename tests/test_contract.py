from medicia_ml.echonext import contract


def test_findings_order_is_fixed():
    assert contract.FINDINGS == (
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


def test_composite_is_last():
    assert contract.FINDINGS[-1] == "shd"
    assert len(set(contract.FINDINGS)) == 12


def test_tabular_order_is_fixed():
    assert contract.TABULAR_FEATURES == (
        "sex",
        "age",
        "ventricular_rate",
        "atrial_rate",
        "pr_interval",
        "qrs_duration",
        "qt_corrected",
    )


def test_waveform_shape():
    assert (contract.WAVEFORM_SAMPLES, contract.WAVEFORM_LEADS) == (2500, 12)

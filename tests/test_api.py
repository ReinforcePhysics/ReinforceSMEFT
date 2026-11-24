import json
from pathlib import Path

import pytest

from reinforcesmeft import ReinforceSMEFT


# ============================================================
# TEST INPUTS
# ============================================================

OPERATORS = [
    "lq1_1111",
    "lq3_1111",
    "phil3_11",
]

OBSERVABLES = [
    "GammaZ",
    "m_W",
    "R(W->cX)",
    "AFB(Z->bb)",
]


def toy_data(scale=1000.0):
    """
    Small synthetic theory dataset using real SMEFT operator and
    flavio observable names.

    The numerical values are intentionally synthetic so the unit tests
    remain fast and independent of full flavio calculations.
    """

    return {
        "metadata": {
            "eft": "SMEFT",
            "basis": "Warsaw",
            "np_scale_gev": scale,
            "normalize": scale**2,
            "operators": OPERATORS,
        },
        "observables": {
            "GammaZ": {
                "sm_prediction": 1.0,
                "sm_uncertainty": 0.0,
                "coefficients": {
                    "lq1_1111": 1.0,
                    "lq3_1111": 0.2,
                },
            },
            "m_W": {
                "sm_prediction": 1.0,
                "sm_uncertainty": 0.0,
                "coefficients": {
                    "lq3_1111": -1.0,
                    "phil3_11": 0.4,
                },
            },
            "R(W->cX)": {
                "sm_prediction": 1.0,
                "sm_uncertainty": 0.0,
                "coefficients": {
                    "lq1_1111": 0.3,
                    "phil3_11": 1.0,
                },
            },
            "AFB(Z->bb)": {
                "sm_prediction": 1.0,
                "sm_uncertainty": 0.0,
                "coefficients": {
                    "lq1_1111": -0.5,
                    "lq3_1111": 0.5,
                    "phil3_11": 0.5,
                },
            },
        },
    }


def small_hyper():
    return {
        "n_batches": 2,
        "rollouts_per_batch": 3,
        "max_steps": 2,
        "min_step_delta_chi2": 0.0,
        "novelty_enabled": False,
        "verbose": False,
        "seed": 3,
    }


# ============================================================
# SEARCH / RESULT TESTS
# ============================================================

def test_search_and_filters():
    measurements = {
        "GammaZ": (1.5, 0.1),
        "m_W": (0.6, 0.1),
        "R(W->cX)": (1.4, 0.1),
        "AFB(Z->bb)": (1.1, 0.1),
    }

    aa = ReinforceSMEFT(
        wc_list=OPERATORS,
        EFT="SMEFT",
        Basis="Warsaw",
        NP_scale=1000.0,
        observables=OBSERVABLES,
        measurements=measurements,
        th_predictions=toy_data(),
        hyper_para=small_hyper(),
    )

    assert aa.trainer.operator_vocab == OPERATORS

    assert aa.dict == sorted(
        aa.dict,
        key=lambda s: s["delta_chi2"],
        reverse=True,
    )

    assert aa.best is aa.dict[0]

    assert all(
        s["delta_chi2"] >= 0
        for s in aa.delta_chi2_gt(0)
    )

    assert all(
        s["n_operators"] == 2
        for s in aa.with_n_operators(2)
    )

    refit = aa.refit(["lq1_1111"])

    assert refit["n_operators"] == 1
    assert 0.0 <= refit["p_value"] <= 1.0

    assert refit["fitted_wc"]["lq1_1111"] == pytest.approx(
        refit["wc"]["lq1_1111"]
    )


# ============================================================
# WC LIST VALIDATION
# ============================================================

def test_wc_list_rejects_mapping():
    with pytest.raises(TypeError, match="Python list"):
        ReinforceSMEFT(
            wc_list={"lq1_1111": 1e-6},
            EFT="SMEFT",
            Basis="Warsaw",
            NP_scale=1000.0,
            observables=["GammaZ"],
            measurements={
                "GammaZ": (1.0, 0.1),
            },
            th_predictions=toy_data(),
            hyper_para={
                "n_batches": 1,
                "rollouts_per_batch": 1,
                "max_steps": 1,
                "verbose": False,
            },
        )


def test_wc_list_rejects_tuple():
    with pytest.raises(TypeError, match="Python list"):
        ReinforceSMEFT(
            wc_list=(
                "lq1_1111",
                "lq3_1111",
            ),
            EFT="SMEFT",
            Basis="Warsaw",
            NP_scale=1000.0,
            observables=["GammaZ"],
            measurements={
                "GammaZ": (1.0, 0.1),
            },
            th_predictions=toy_data(),
            hyper_para={
                "n_batches": 1,
                "rollouts_per_batch": 1,
                "max_steps": 1,
                "verbose": False,
            },
        )


# ============================================================
# NORMALIZATION VALIDATION
# ============================================================

def test_np_scale_normalization_is_validated():
    bad = toy_data(scale=1000.0)

    bad["metadata"]["normalize"] = 123.0

    with pytest.raises(
        ValueError,
        match=r"NORMALIZE=Lambda\^2",
    ):
        ReinforceSMEFT(
            wc_list=["lq1_1111"],
            EFT="SMEFT",
            Basis="Warsaw",
            NP_scale=1000.0,
            observables=["GammaZ"],
            measurements={
                "GammaZ": (1.2, 0.1),
            },
            th_predictions=bad,
            hyper_para={
                "n_batches": 1,
                "rollouts_per_batch": 1,
                "max_steps": 1,
                "verbose": False,
            },
        )


# ============================================================
# MEASUREMENT FALLBACK
# ============================================================

def test_measurement_fallback(monkeypatch):
    import reinforcesmeft.data as data_module

    monkeypatch.setattr(
        data_module,
        "_measurement_from_flavio",
        lambda obs: (1.1, 0.1),
    )

    aa = ReinforceSMEFT(
        wc_list=["lq1_1111"],
        EFT="SMEFT",
        Basis="Warsaw",
        NP_scale=1000.0,
        observables=["GammaZ"],
        measurements=None,
        th_predictions=toy_data(),
        hyper_para={
            "n_batches": 1,
            "rollouts_per_batch": 1,
            "max_steps": 1,
            "min_step_delta_chi2": -1.0,
            "novelty_enabled": False,
            "verbose": False,
        },
    )

    assert aa.dataset.exp_central["GammaZ"] == pytest.approx(
        1.1
    )


# ============================================================
# AUTOMATIC THEORY GENERATION
# ============================================================

def test_missing_theory_predictions_triggers_generation(
    monkeypatch,
    tmp_path,
):
    import reinforcesmeft.api as api_module

    called = {}

    def fake_generate(
        observables,
        output_file,
        *,
        operators,
        eft,
        basis,
        scale,
        **kwargs,
    ):
        called["observables"] = list(observables)
        called["operators"] = list(operators)
        called["scale"] = scale

        payload = toy_data(scale)

        payload["metadata"]["operators"] = list(
            operators
        )

        selected_observables = {}

        for observable in observables:
            selected_observables[observable] = (
                payload["observables"][observable]
            )

        payload["observables"] = selected_observables

        Path(output_file).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        return payload

    monkeypatch.setattr(
        api_module,
        "generate_linear_coefficients",
        fake_generate,
    )

    monkeypatch.chdir(tmp_path)

    aa = ReinforceSMEFT(
        wc_list=[
            "lq1_1111",
            "lq3_1111",
        ],
        EFT="SMEFT",
        Basis="Warsaw",
        NP_scale=2000.0,
        observables=["GammaZ"],
        measurements={
            "GammaZ": (1.2, 0.1),
        },
        th_predictions=None,
        hyper_para={
            "n_batches": 1,
            "rollouts_per_batch": 1,
            "max_steps": 1,
            "min_step_delta_chi2": -1.0,
            "novelty_enabled": False,
            "verbose": False,
        },
    )

    assert called == {
        "observables": ["GammaZ"],
        "operators": [
            "lq1_1111",
            "lq3_1111",
        ],
        "scale": 2000.0,
    }

    assert aa.generated_theory_path is not None
    assert aa.generated_theory_path.exists()

    assert aa.dataset.normalize == pytest.approx(
        4_000_000.0
    )


# ============================================================
# SUPPLIED THEORY DATA
# ============================================================

def test_supplied_theory_predictions_do_not_generate(
    monkeypatch,
):
    import reinforcesmeft.api as api_module

    def should_not_run(*args, **kwargs):
        raise AssertionError(
            "coefficient generation should not run"
        )

    monkeypatch.setattr(
        api_module,
        "generate_linear_coefficients",
        should_not_run,
    )

    ReinforceSMEFT(
        wc_list=["lq1_1111"],
        EFT="SMEFT",
        Basis="Warsaw",
        NP_scale=1000.0,
        observables=["GammaZ"],
        measurements={
            "GammaZ": (1.2, 0.1),
        },
        th_predictions=toy_data(),
        hyper_para={
            "n_batches": 1,
            "rollouts_per_batch": 1,
            "max_steps": 1,
            "min_step_delta_chi2": -1.0,
            "novelty_enabled": False,
            "verbose": False,
        },
    )

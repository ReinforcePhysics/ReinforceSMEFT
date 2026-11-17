import json
from pathlib import Path

import pytest

from reinforcesmeft import ReinforceSMEFT


def toy_data(scale=1000.0):
    return {
        "metadata": {
            "eft": "SMEFT",
            "basis": "Warsaw",
            "np_scale_gev": scale,
            "normalize": scale**2,
            "operators": ["op1", "op2", "op3"],
        },
        "observables": {
            "O1": {"sm_prediction": 1.0, "sm_uncertainty": 0.0, "coefficients": {"op1": 1.0, "op2": 0.2}},
            "O2": {"sm_prediction": 1.0, "sm_uncertainty": 0.0, "coefficients": {"op2": -1.0, "op3": 0.4}},
            "O3": {"sm_prediction": 1.0, "sm_uncertainty": 0.0, "coefficients": {"op1": 0.3, "op3": 1.0}},
            "O4": {"sm_prediction": 1.0, "sm_uncertainty": 0.0, "coefficients": {"op1": -0.5, "op2": 0.5, "op3": 0.5}},
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


def test_search_and_filters():
    measurements = {
        "O1": (1.5, 0.1),
        "O2": (0.6, 0.1),
        "O3": (1.4, 0.1),
        "O4": (1.1, 0.1),
    }
    aa = reinforcesmeft(
        wc_list=["op1", "op2", "op3"],
        EFT="SMEFT",
        Basis="Warsaw",
        NP_scale=1000.0,
        observables=["O1", "O2", "O3", "O4"],
        measurements=measurements,
        th_predictions=toy_data(),
        hyper_para=small_hyper(),
    )
    assert aa.trainer.operator_vocab == ["op1", "op2", "op3"]
    assert aa.dict == sorted(aa.dict, key=lambda s: s["delta_chi2"], reverse=True)
    assert aa.best is aa.dict[0]
    assert all(s["delta_chi2"] >= 0 for s in aa.delta_chi2_gt(0))
    assert all(s["n_operators"] == 2 for s in aa.with_n_operators(2))
    refit = aa.refit(["op1"])
    assert refit["n_operators"] == 1
    assert 0.0 <= refit["p_value"] <= 1.0
    assert refit["fitted_wc"]["op1"] == pytest.approx(refit["wc"]["op1"])


def test_wc_list_rejects_mapping():
    with pytest.raises(TypeError, match="Python list"):
        reinforcesmeft(
            wc_list={"op1": 1e-6},
            EFT="SMEFT",
            Basis="Warsaw",
            NP_scale=1000.0,
            observables=["O1"],
            measurements={"O1": (1.0, 0.1)},
            th_predictions=toy_data(),
            hyper_para={"n_batches": 1, "rollouts_per_batch": 1, "max_steps": 1, "verbose": False},
        )


def test_wc_list_rejects_tuple():
    with pytest.raises(TypeError, match="Python list"):
        reinforcesmeft(
            wc_list=("op1", "op2"),
            EFT="SMEFT",
            Basis="Warsaw",
            NP_scale=1000.0,
            observables=["O1"],
            measurements={"O1": (1.0, 0.1)},
            th_predictions=toy_data(),
            hyper_para={"n_batches": 1, "rollouts_per_batch": 1, "max_steps": 1, "verbose": False},
        )


def test_np_scale_normalization_is_validated():
    bad = toy_data(scale=1000.0)
    bad["metadata"]["normalize"] = 123.0
    with pytest.raises(ValueError, match="NORMALIZE=Lambda\\^2"):
        reinforcesmeft(
            wc_list=["op1"],
            EFT="SMEFT",
            Basis="Warsaw",
            NP_scale=1000.0,
            observables=["O1"],
            measurements={"O1": (1.2, 0.1)},
            th_predictions=bad,
            hyper_para={"n_batches": 1, "rollouts_per_batch": 1, "max_steps": 1, "verbose": False},
        )


def test_measurement_fallback(monkeypatch):
    import reinforcsmeft.data as data_module

    monkeypatch.setattr(data_module, "_measurement_from_flavio", lambda obs: (1.1, 0.1))
    aa = reinforcesmeft(
        wc_list=["op1"],
        EFT="SMEFT",
        Basis="Warsaw",
        NP_scale=1000.0,
        observables=["O1"],
        measurements=None,
        th_predictions=toy_data(),
        hyper_para={"n_batches": 1, "rollouts_per_batch": 1, "max_steps": 1, "min_step_delta_chi2": -1.0, "novelty_enabled": False, "verbose": False},
    )
    assert aa.dataset.exp_central["O1"] == pytest.approx(1.1)


def test_missing_theory_predictions_triggers_generation(monkeypatch, tmp_path):
    import reinforcesmeft.api as api_module

    called = {}

    def fake_generate(observables, output_file, *, operators, eft, basis, scale, **kwargs):
        called["observables"] = list(observables)
        called["operators"] = list(operators)
        called["scale"] = scale
        payload = toy_data(scale)
        payload["metadata"]["operators"] = list(operators)
        payload["observables"] = {"O1": payload["observables"]["O1"]}
        Path(output_file).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(api_module, "generate_linear_coefficients", fake_generate)
    monkeypatch.chdir(tmp_path)

    aa = reinforcesmeft(
        wc_list=["op1", "op2"],
        EFT="SMEFT",
        Basis="Warsaw",
        NP_scale=2000.0,
        observables=["O1"],
        measurements={"O1": (1.2, 0.1)},
        th_predictions=None,
        hyper_para={"n_batches": 1, "rollouts_per_batch": 1, "max_steps": 1, "min_step_delta_chi2": -1.0, "novelty_enabled": False, "verbose": False},
    )
    assert called == {"observables": ["O1"], "operators": ["op1", "op2"], "scale": 2000.0}
    assert aa.generated_theory_path is not None
    assert aa.generated_theory_path.exists()
    assert aa.dataset.normalize == pytest.approx(4_000_000.0)


def test_supplied_theory_predictions_do_not_generate(monkeypatch):
    import reinforcesmeft.api as api_module

    def should_not_run(*args, **kwargs):
        raise AssertionError("coefficient generation should not run")

    monkeypatch.setattr(api_module, "generate_linear_coefficients", should_not_run)
    reinforcesmeft(
        wc_list=["op1"],
        EFT="SMEFT",
        Basis="Warsaw",
        NP_scale=1000.0,
        observables=["O1"],
        measurements={"O1": (1.2, 0.1)},
        th_predictions=toy_data(),
        hyper_para={"n_batches": 1, "rollouts_per_batch": 1, "max_steps": 1, "min_step_delta_chi2": -1.0, "novelty_enabled": False, "verbose": False},
    )

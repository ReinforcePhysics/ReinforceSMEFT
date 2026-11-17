from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import HyperParameters
from .data import LinearDataset, observable_key
from .fitting import LinearFitter
from .linearize import generate_linear_coefficients
from .rl import RLTrainer


class ReinforceSMEFT:
    """High-level RL search API.

    Parameters
    ----------
    wc_list:
        List of Wilson-coefficient names that defines the RL action vocabulary.
    EFT, Basis:
        WCxf EFT and basis names, e.g. ``"SMEFT"`` and ``"Warsaw"``.
    NP_scale:
        New-physics scale in GeV.
    observables:
        List of flavio observables. Argument-dependent observables can be tuples,
        e.g. ``("<P5p>(B0->K*mumu)", 4, 6)``.
    measurements:
        Optional mapping of measurements. Missing entries are filled from flavio's
        default/combined measurements. If ``None``, flavio measurements are used for
        every requested observable.
    th_predictions:
        Existing linear-theory JSON path or mapping. If omitted, the package first
        computes the requested linear coefficients with flavio/wilson, saves them to
        a JSON file, and then trains on that file.
    hyper_para:
        Training hyperparameters.
    """

    def __init__(
        self,
        wc_list: Sequence[str],
        EFT: str,
        Basis: str,
        NP_scale: float,
        observables: Sequence[Any],
        measurements: Mapping[str, Any] | None = None,
        th_predictions: str | Path | Mapping[str, Any] | None = None,
        hyper_para: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(wc_list, list):
            raise TypeError(
                "wc_list must be a Python list of Wilson-coefficient names, e.g. "
                "['lq1_1111', 'lq3_1122']; coefficient values are fitted internally."
            )
        self.wc_list = list(dict.fromkeys(str(wc) for wc in wc_list))
        if not self.wc_list:
            raise ValueError("wc_list must contain at least one Wilson coefficient name")

        self.EFT = str(EFT)
        self.Basis = str(Basis)
        self.NP_scale = float(NP_scale)
        self.observables = list(observables)
        if not self.observables:
            raise ValueError("observables must contain at least one observable")
        self.hyper = HyperParameters.from_dict(hyper_para)

        self.generated_theory_path: Path | None = None
        if th_predictions is None:
            self.generated_theory_path = self._default_theory_path()
            generate_linear_coefficients(
                self.observables,
                self.generated_theory_path,
                operators=self.wc_list,
                eft=self.EFT,
                basis=self.Basis,
                scale=self.NP_scale,
            )
            theory_source: str | Path | Mapping[str, Any] = self.generated_theory_path
        else:
            # Important: supplied theory predictions are never recomputed.
            theory_source = th_predictions

        self.dataset = LinearDataset(
            theory_source,
            measurements,
            eft=self.EFT,
            basis=self.Basis,
            scale=self.NP_scale,
            observables=self.observables,
            operator_vocab=self.wc_list,
        )
        self.fitter = LinearFitter(self.dataset, self.hyper)
        self.trainer = RLTrainer(self.fitter, self.hyper)
        self.raw_solutions, self.metrics = self.trainer.train()
        self.dict = self._deduplicate_and_sort(self.raw_solutions)

    def _default_theory_path(self) -> Path:
        safe_eft = self.EFT.replace("/", "_").replace(" ", "_")
        safe_basis = self.Basis.replace("/", "_").replace(" ", "_")
        scale_text = f"{self.NP_scale:g}".replace(".", "p")
        return Path.cwd() / f"linear_coefficients_{safe_eft}_{safe_basis}_{scale_text}GeV.json"

    @staticmethod
    def _deduplicate_and_sort(solutions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        best: dict[tuple[str, ...], dict[str, Any]] = {}
        for solution in solutions:
            key = tuple(sorted(solution["operators"]))
            previous = best.get(key)
            if previous is None or solution["chi2"] < previous["chi2"]:
                best[key] = dict(solution)
        return sorted(best.values(), key=lambda item: item["delta_chi2"], reverse=True)

    @property
    def solutions(self) -> list[dict[str, Any]]:
        return self.dict

    @property
    def best(self) -> dict[str, Any] | None:
        return self.dict[0] if self.dict else None

    def top(self, n: int = 10) -> list[dict[str, Any]]:
        if n < 0:
            raise ValueError("n must be >= 0")
        return self.dict[:n]

    def filter(
        self,
        *,
        min_delta_chi2: float | None = None,
        max_delta_chi2: float | None = None,
        min_p_value: float | None = None,
        max_p_value: float | None = None,
        min_chi2: float | None = None,
        max_chi2: float | None = None,
        n_operators: int | None = None,
        min_operators: int | None = None,
        max_operators: int | None = None,
        contains: Sequence[str] | None = None,
        exact_operators: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        contains_set = set(contains or [])
        exact_set = set(exact_operators) if exact_operators is not None else None
        result = []
        for solution in self.dict:
            if min_delta_chi2 is not None and solution["delta_chi2"] < min_delta_chi2:
                continue
            if max_delta_chi2 is not None and solution["delta_chi2"] > max_delta_chi2:
                continue
            if min_p_value is not None and solution["p_value"] < min_p_value:
                continue
            if max_p_value is not None and solution["p_value"] > max_p_value:
                continue
            if min_chi2 is not None and solution["chi2"] < min_chi2:
                continue
            if max_chi2 is not None and solution["chi2"] > max_chi2:
                continue
            if n_operators is not None and solution["n_operators"] != n_operators:
                continue
            if min_operators is not None and solution["n_operators"] < min_operators:
                continue
            if max_operators is not None and solution["n_operators"] > max_operators:
                continue
            support = set(solution["operators"])
            if contains_set and not contains_set.issubset(support):
                continue
            if exact_set is not None and support != exact_set:
                continue
            result.append(solution)
        return result

    def delta_chi2_gt(self, x: float) -> list[dict[str, Any]]:
        return self.filter(min_delta_chi2=float(x))

    def p_value_gt(self, y: float) -> list[dict[str, Any]]:
        return self.filter(min_p_value=float(y))

    def chi2_below(self, x: float) -> list[dict[str, Any]]:
        return self.filter(max_chi2=float(x))

    def with_n_operators(self, n: int) -> list[dict[str, Any]]:
        return self.filter(n_operators=int(n))

    def near_chi2(self, target: float, tolerance: float = 1.0) -> list[dict[str, Any]]:
        target, tolerance = float(target), abs(float(tolerance))
        return self.filter(min_chi2=target - tolerance, max_chi2=target + tolerance)

    def near_p_value(self, target: float, tolerance: float = 0.01) -> list[dict[str, Any]]:
        target, tolerance = float(target), abs(float(tolerance))
        return self.filter(min_p_value=target - tolerance, max_p_value=target + tolerance)

    def refit(self, operators: Sequence[str]) -> dict[str, Any]:
        fit = self.fitter.fit(operators)
        return {
            "operators": list(fit.operators),
            "n_operators": len(fit.operators),
            "wc": dict(fit.wc),
            "fitted_wc": dict(fit.fitted_wc),
            "chi2": fit.chi2,
            "chi2_min": fit.chi2,
            "delta_chi2": self.dataset.sm_chi2 - fit.chi2,
            "delta_chi2_sm": self.dataset.sm_chi2 - fit.chi2,
            "p_value": fit.p_value,
            "n_dof": fit.n_dof,
            "pulls": [float(x) for x in fit.pulls],
            "predictions": dict(fit.predictions),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "eft": self.EFT,
            "basis": self.Basis,
            "np_scale_gev": self.NP_scale,
            "wc_action_vocabulary": list(self.wc_list),
            "requested_observables": [observable_key(obs) for obs in self.observables],
            "retained_observables": list(self.dataset.observables),
            "n_observables": len(self.dataset.observables),
            "n_candidate_operators": len(self.trainer.operator_vocab),
            "n_raw_rollout_solutions": len(self.raw_solutions),
            "n_distinct_solutions": len(self.dict),
            "sm_chi2": self.dataset.sm_chi2,
            "normalization": self.dataset.normalize,
            "normalization_definition": "x_i = Lambda^2 * C_i",
            "generated_theory_path": (
                str(self.generated_theory_path) if self.generated_theory_path else None
            ),
            "best": self.best,
            "skipped_observables": dict(self.dataset.skipped),
            "hyperparameters": self.hyper.to_dict(),
        }

    def save(self, path: str | Path, *, include_raw: bool = False) -> Path:
        path = Path(path).expanduser()
        payload: dict[str, Any] = {
            "summary": self.summary(),
            "solutions": self.dict,
            "metrics": self.metrics,
        }
        if include_raw:
            payload["raw_solutions"] = self.raw_solutions
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path


# Lowercase constructor retained as the concise public API.
#anomalies = Anomalies

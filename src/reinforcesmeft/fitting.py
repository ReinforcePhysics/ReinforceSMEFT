from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import chi2 as chi2_distribution

from .config import HyperParameters
from .data import LinearDataset


@dataclass(slots=True)
class FitResult:
    operators: list[str]
    chi2: float
    pulls: np.ndarray
    predictions: dict[str, float]
    wc: dict[str, float]
    fitted_wc: dict[str, float]
    p_value: float
    n_dof: int


class LinearFitter:
    """Bounded linear least-squares backend for selected operator supports."""

    def __init__(self, dataset: LinearDataset, hyper: HyperParameters) -> None:
        self.dataset = dataset
        self.hyper = hyper
        self._baseline_pulls = dataset.pulls(dataset.baseline_predictions)

    def fit(self, operators: Sequence[str]) -> FitResult:
        operators = list(dict.fromkeys(str(op) for op in operators))
        unknown = sorted(set(operators) - set(self.dataset.operator_vocab))
        if unknown:
            raise ValueError("Unknown operator(s): " + ", ".join(unknown))

        if not operators:
            predictions = dict(self.dataset.baseline_predictions)
            pulls = self.dataset.pulls(predictions)
            chi2 = float(np.dot(pulls, pulls))
            return self._result([], chi2, pulls, predictions, {}, {})

        n_obs = len(self.dataset.observables)
        A = np.zeros((n_obs, len(operators)), dtype=np.float64)
        for row, obs in enumerate(self.dataset.observables):
            sm = self.dataset.sm_prediction[obs]
            sigma = self.dataset.total_sigma[obs]
            coeffs = self.dataset.entries[obs].get("coefficients", {})
            for col, op in enumerate(operators):
                A[row, col] = sm * float(coeffs.get(op, 0.0)) / sigma

        result = lsq_linear(
            A,
            -self._baseline_pulls,
            bounds=(self.hyper.fit_lower, self.hyper.fit_upper),
            method="trf",
            tol=self.hyper.fit_tol,
            lsmr_tol=self.hyper.fit_tol,
            max_iter=self.hyper.fit_max_iter,
            verbose=0,
        )
        if not result.success:
            result = lsq_linear(
                A,
                -self._baseline_pulls,
                bounds=(self.hyper.fit_lower, self.hyper.fit_upper),
                method="bvls",
                tol=self.hyper.fit_tol,
                max_iter=max(self.hyper.fit_max_iter, 5000),
                verbose=0,
            )
        if not result.success:
            raise RuntimeError(f"Linear least-squares fit failed: {result.message}")

        fitted_wc = {
            op: float(value) / self.dataset.normalize
            for op, value in zip(operators, np.asarray(result.x, dtype=np.float64))
        }
        total_wc = dict(fitted_wc)
        predictions = self.dataset.predict(total_wc)
        pulls = self.dataset.pulls(predictions)
        chi2 = float(np.dot(pulls, pulls))
        return self._result(operators, chi2, pulls, predictions, total_wc, fitted_wc)

    def _result(
        self,
        operators: list[str],
        chi2: float,
        pulls: np.ndarray,
        predictions: dict[str, float],
        wc: dict[str, float],
        fitted_wc: dict[str, float],
    ) -> FitResult:
        n_dof = max(len(self.dataset.observables) - len(operators), 1)
        p_value = float(chi2_distribution.sf(chi2, n_dof))
        return FitResult(
            operators=operators,
            chi2=float(chi2),
            pulls=np.asarray(pulls, dtype=np.float64),
            predictions=predictions,
            wc=wc,
            fitted_wc=fitted_wc,
            p_value=p_value,
            n_dof=n_dof,
        )

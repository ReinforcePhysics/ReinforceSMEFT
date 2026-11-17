from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def observable_key(obs: Any) -> str:
    """Return the JSON key used for a scalar or argument-dependent flavio observable."""
    if isinstance(obs, tuple):
        if not obs:
            raise ValueError("Observable tuple cannot be empty")
        return f"{obs[0]}({', '.join(map(str, obs[1:]))})"
    return str(obs)


def parse_observable_key(key: Any) -> Any:
    """Convert a JSON observable key back to flavio's scalar/tuple representation."""
    if not isinstance(key, str) or not key.endswith(")"):
        return key
    last_open = key.rfind("(")
    if last_open < 0:
        return key
    arg_text = key[last_open + 1 : -1].strip()
    if not arg_text:
        return key
    parts = [part.strip() for part in arg_text.split(",")]
    try:
        args = tuple(float(part) for part in parts)
    except ValueError:
        return key
    return (key[:last_open], *args)


def load_theory_predictions(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        data = dict(source)
    else:
        path = Path(source).expanduser()
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    if "observables" not in data or not isinstance(data["observables"], Mapping):
        raise ValueError("th_predictions must contain an 'observables' mapping")
    return data


def _coerce_measurement(value: Any) -> tuple[float, float]:
    if isinstance(value, Mapping):
        central = value.get("central", value.get("value", value.get("mean")))
        sigma = value.get(
            "uncertainty", value.get("sigma", value.get("error", value.get("std")))
        )
        if central is None or sigma is None:
            raise ValueError("measurement dict needs central/value and uncertainty/sigma")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        central, sigma = value
    else:
        raise ValueError(
            "measurement must be {'central': value, 'uncertainty': sigma} "
            "or a two-element (value, sigma) sequence"
        )
    central, sigma = float(central), float(sigma)
    if not np.isfinite(central) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("measurement central value must be finite and uncertainty > 0")
    return central, sigma


def _component(value: Any, idx: int) -> Any:
    return value[idx] if isinstance(value, (list, tuple, np.ndarray)) else value


def _distribution_central_sigma(distribution: Any, idx: int = 0) -> tuple[float, float]:
    """Extract an effective central value and 1-sigma error from a flavio distribution."""
    if hasattr(distribution, "covariance"):
        cv = np.asarray(distribution.central_value)
        cov = np.asarray(distribution.covariance)
        return float(cv[idx]), float(np.sqrt(cov[idx, idx]))
    if hasattr(distribution, "standard_deviation"):
        return (
            float(_component(distribution.central_value, idx)),
            float(_component(distribution.standard_deviation, idx)),
        )
    if hasattr(distribution, "ppf"):
        q16 = float(_component(distribution.ppf(0.158655254), idx))
        q50 = float(_component(distribution.ppf(0.5), idx))
        q84 = float(_component(distribution.ppf(0.841344746), idx))
        return q50, 0.5 * ((q50 - q16) + (q84 - q50))
    raise ValueError("Unsupported flavio probability distribution")


def _measurement_from_flavio(obs_key: str) -> tuple[float, float]:
    """Return flavio's default/combined measurement for an observable.

    We first use flavio.combine_measurements when available. A registry-based
    fallback is retained for flavio versions or observable types for which the
    combination helper is unavailable.
    """
    try:
        import flavio
    except ImportError as exc:
        raise ImportError(
            "flavio is required when measurements=None. Reinstall the package with "
            "its dependencies (normal `pip install reinforcesmeft` installs flavio)."
        ) from exc

    target = parse_observable_key(obs_key)

    combine = getattr(flavio, "combine_measurements", None)
    if combine is not None:
        try:
            distribution = combine(target)
            central, sigma = _distribution_central_sigma(distribution, 0)
            if np.isfinite(central) and np.isfinite(sigma) and sigma > 0:
                return central, sigma
        except Exception:
            pass

    candidates: list[tuple[float, float]] = []
    for measurement in flavio.Measurement.instances.values():
        for distribution, parameters in measurement._constraints:
            if target not in parameters or "UpperLimit" in type(distribution).__name__:
                continue
            idx = parameters.index(target)
            try:
                central, sigma = _distribution_central_sigma(distribution, idx)
            except Exception:
                continue
            if np.isfinite(central) and np.isfinite(sigma) and sigma > 0:
                candidates.append((central, sigma))

    if not candidates:
        raise ValueError(f"No usable central-value flavio measurement found for {obs_key}")
    return candidates[0]


class LinearDataset:
    """Validated linearized-observable dataset used by fitting and RL."""

    def __init__(
        self,
        theory_predictions: str | Path | Mapping[str, Any],
        measurements: Mapping[str, Any] | None,
        *,
        eft: str,
        basis: str,
        scale: float,
        observables: Sequence[Any],
        operator_vocab: Sequence[str],
    ) -> None:
        self.raw = load_theory_predictions(theory_predictions)
        self.metadata = dict(self.raw.get("metadata", {}))
        self.eft = str(eft)
        self.basis = str(basis)
        self.scale = float(scale)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("NP_scale must be a positive finite number in GeV")

        # Dimension-six convention: x_i = Lambda^2 C_i, with Lambda in GeV.
        self.expected_normalize = self.scale**2
        self.normalize = float(self.metadata.get("normalize", self.expected_normalize))
        self._validate_metadata()

        self.requested_observables = [observable_key(obs) for obs in observables]
        if not self.requested_observables:
            raise ValueError("observables must contain at least one observable")
        if len(set(self.requested_observables)) != len(self.requested_observables):
            raise ValueError("observables contains duplicate entries")

        self.operator_vocab = list(dict.fromkeys(str(op) for op in operator_vocab))
        if not self.operator_vocab:
            raise ValueError("wc_list must contain at least one Wilson coefficient name")
        self._validate_operator_vocab()

        explicit_measurements = {
            observable_key(key): value for key, value in (measurements or {}).items()
        }
        self.observables: list[str] = []
        self.entries: dict[str, dict[str, Any]] = {}
        self.exp_central: dict[str, float] = {}
        self.exp_sigma: dict[str, float] = {}
        self.theory_sigma: dict[str, float] = {}
        self.total_sigma: dict[str, float] = {}
        self.sm_prediction: dict[str, float] = {}
        self.skipped: dict[str, str] = {}

        raw_observables = self.raw["observables"]
        missing = [obs for obs in self.requested_observables if obs not in raw_observables]
        if missing:
            raise ValueError(
                "Requested observables are absent from th_predictions: " + ", ".join(missing)
            )

        for obs in self.requested_observables:
            entry = raw_observables[obs]
            try:
                self._add_observable(obs, entry, explicit_measurements)
            except Exception as exc:
                self.skipped[obs] = str(exc)

        if not self.observables:
            details = "; ".join(f"{k}: {v}" for k, v in self.skipped.items())
            raise ValueError("No usable observables remain after validation. " + details)

        self.sm_predictions = self.predict({})
        self.sm_pulls = self.pulls(self.sm_predictions)
        self.sm_chi2 = self.chi2(self.sm_predictions)
        # The RL search always starts from the SM in the revised API.
        self.baseline_predictions = dict(self.sm_predictions)
        self.baseline_pulls = self.sm_pulls.copy()
        self.baseline_chi2 = self.sm_chi2

    def _validate_metadata(self) -> None:
        for key, requested in (("eft", self.eft), ("basis", self.basis)):
            if key in self.metadata and str(self.metadata[key]).lower() != str(requested).lower():
                raise ValueError(
                    f"Requested {key}={requested!r}, but th_predictions metadata "
                    f"contains {key}={self.metadata[key]!r}"
                )

        stored_scale = self.metadata.get("np_scale_gev", self.metadata.get("scale"))
        if stored_scale is not None and not math.isclose(
            float(stored_scale), self.scale, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(
                f"Requested NP scale={self.scale} GeV, but th_predictions were "
                f"generated at {float(stored_scale)} GeV"
            )

        if not np.isfinite(self.normalize) or self.normalize <= 0:
            raise ValueError("metadata['normalize'] must be a positive finite number")
        if not math.isclose(
            self.normalize, self.expected_normalize, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(
                "Inconsistent Wilson-coefficient normalization: for the requested "
                f"NP scale Lambda={self.scale:g} GeV, the package requires "
                f"NORMALIZE=Lambda^2={self.expected_normalize:g}, but the JSON contains "
                f"{self.normalize:g}. Regenerate the theory predictions at this NP scale."
            )

    def _validate_operator_vocab(self) -> None:
        metadata_ops = self.metadata.get("operators")
        if isinstance(metadata_ops, list) and metadata_ops:
            known = {str(op) for op in metadata_ops}
            missing = [op for op in self.operator_vocab if op not in known]
            if missing:
                raise ValueError(
                    "wc_list contains coefficients absent from th_predictions metadata: "
                    + ", ".join(missing)
                )
            return

        known = {
            str(op)
            for entry in self.raw["observables"].values()
            if isinstance(entry, Mapping)
            for op in entry.get("coefficients", {})
        }
        missing = [op for op in self.operator_vocab if op not in known]
        if missing:
            raise ValueError(
                "wc_list contains coefficients not found in th_predictions: "
                + ", ".join(missing)
            )

    def _add_observable(
        self,
        obs: str,
        entry: Any,
        explicit_measurements: Mapping[str, Any],
    ) -> None:
        if not isinstance(entry, Mapping):
            raise ValueError("invalid theory-prediction entry")
        if "error" in entry:
            raise ValueError(f"linearization error: {entry['error']}")
        if "sm_prediction" not in entry or "coefficients" not in entry:
            raise ValueError("entry needs sm_prediction and coefficients")

        sm = float(entry["sm_prediction"])
        theory_sigma = float(entry.get("sm_uncertainty", 0.0))
        if not np.isfinite(sm) or not np.isfinite(theory_sigma) or theory_sigma < 0:
            raise ValueError("invalid SM prediction or uncertainty")

        if obs in explicit_measurements:
            central, exp_sigma = _coerce_measurement(explicit_measurements[obs])
        else:
            central, exp_sigma = _measurement_from_flavio(obs)

        total_sigma = math.sqrt(theory_sigma**2 + exp_sigma**2)
        if not np.isfinite(total_sigma) or total_sigma <= 0:
            raise ValueError("non-positive combined uncertainty")

        self.observables.append(obs)
        self.entries[obs] = dict(entry)
        self.exp_central[obs] = central
        self.exp_sigma[obs] = exp_sigma
        self.theory_sigma[obs] = theory_sigma
        self.total_sigma[obs] = total_sigma
        self.sm_prediction[obs] = sm

    def predict(self, wc_values: Mapping[str, float]) -> dict[str, float]:
        """Linear prediction for physical Wilson coefficients C_i in GeV^-2."""
        result: dict[str, float] = {}
        for obs in self.observables:
            coeffs = self.entries[obs].get("coefficients", {})
            shift = sum(
                float(coeffs.get(wc, 0.0)) * self.normalize * float(value)
                for wc, value in wc_values.items()
            )
            result[obs] = self.sm_prediction[obs] * (1.0 + shift)
        return result

    def pulls(self, predictions: Mapping[str, float]) -> np.ndarray:
        return np.asarray(
            [
                (float(predictions[obs]) - self.exp_central[obs]) / self.total_sigma[obs]
                for obs in self.observables
            ],
            dtype=np.float64,
        )

    def chi2(self, predictions: Mapping[str, float]) -> float:
        pulls = self.pulls(predictions)
        return float(np.dot(pulls, pulls))

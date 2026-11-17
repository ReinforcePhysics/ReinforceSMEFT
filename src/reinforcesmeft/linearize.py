from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .data import observable_key


def _unpack(obs: Any) -> tuple[str, tuple[Any, ...]]:
    return (str(obs[0]), tuple(obs[1:])) if isinstance(obs, tuple) else (str(obs), ())


def _validate_operators(operators: Sequence[str], eft: str, basis: str) -> list[str]:
    try:
        from wilson import wcxf
    except ImportError as exc:
        raise ImportError(
            "wilson is required for coefficient generation. A normal package installation "
            "installs it automatically."
        ) from exc

    requested = list(dict.fromkeys(str(op) for op in operators))
    if not requested:
        raise ValueError("wc_list must contain at least one Wilson coefficient name")

    basis_obj = wcxf.Basis[eft, basis]
    known = {
        str(wc)
        for sector_data in basis_obj.sectors.values()
        for wc in sector_data.keys()
    }
    missing = [op for op in requested if op not in known]
    if missing:
        raise ValueError(
            f"Unknown Wilson coefficient(s) for {eft}/{basis}: " + ", ".join(missing)
        )
    return requested


def _one_observable(
    obs: Any,
    operators: Sequence[str],
    *,
    scale: float,
    eft: str,
    basis: str,
    normalize: float,
    fd_steps: Sequence[float],
    coefficient_threshold: float,
) -> dict[str, Any]:
    try:
        import flavio
        import numdifftools as nd
        from wilson import Wilson
    except ImportError as exc:
        raise ImportError(
            "flavio, wilson, and numdifftools are required for coefficient generation. "
            "A normal package installation installs them automatically."
        ) from exc

    name, args = _unpack(obs)
    sm = float(flavio.sm_prediction(name, *args))
    sm_unc = float(flavio.sm_uncertainty(name, *args))
    if not np.isfinite(sm) or abs(sm) < 1e-300:
        raise ValueError(f"Invalid SM prediction for {observable_key(obs)}: {sm}")

    def function(x: np.ndarray) -> float:
        # x_i = Lambda^2 C_i, hence C_i = x_i / Lambda^2.
        wc = {op: float(x[i]) / normalize for i, op in enumerate(operators)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prediction = float(
                flavio.np_prediction(name, Wilson(wc, scale, eft, basis), *args)
            )
        if not np.isfinite(prediction):
            raise ValueError("Non-finite flavio prediction")
        return prediction / sm

    x0 = np.zeros(len(operators), dtype=float)
    last_error: Exception | None = None
    for step in fd_steps:
        try:
            gradient = np.asarray(
                nd.Gradient(function, method="central", step=step, order=2)(x0),
                dtype=float,
            )
            if gradient.shape != (len(operators),) or not np.all(np.isfinite(gradient)):
                raise ValueError("Invalid finite-difference gradient")
            coefficients = {
                op: float(value)
                for op, value in sorted(
                    zip(operators, gradient), key=lambda item: abs(item[1]), reverse=True
                )
                if abs(float(value)) > coefficient_threshold
            }
            return {
                "sm_prediction": sm,
                "sm_uncertainty": sm_unc,
                "finite_difference_x_step": float(step),
                "finite_difference_wc_step": float(step / normalize),
                "coefficients": coefficients,
            }
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not linearize {observable_key(obs)}: {last_error}")


def generate_linear_coefficients(
    observables: Iterable[Any],
    output_file: str | Path,
    *,
    operators: Sequence[str],
    eft: str = "SMEFT",
    basis: str = "Warsaw",
    scale: float = 1000.0,
    fd_steps: Sequence[float] = (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
    coefficient_threshold: float = 1e-12,
) -> dict[str, Any]:
    """Generate the linear theory JSON used by :class:`reinforcesmeft.ReinforceSMEFT`.

    Wilson coefficients are normalized with the dimension-six convention
    x_i = Lambda^2 C_i, where Lambda is ``scale`` in GeV. Therefore a 1 TeV
    scale gives NORMALIZE=1e6, while another NP scale is handled automatically.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number in GeV")
    normalize = scale**2
    obs_list = list(observables)
    if not obs_list:
        raise ValueError("observables must contain at least one observable")
    operators = _validate_operators(operators, eft, basis)

    results: dict[str, Any] = {}
    for obs in obs_list:
        key = observable_key(obs)
        try:
            results[key] = _one_observable(
                obs,
                operators,
                scale=scale,
                eft=eft,
                basis=basis,
                normalize=normalize,
                fd_steps=fd_steps,
                coefficient_threshold=coefficient_threshold,
            )
        except Exception as exc:
            results[key] = {"error": str(exc), "error_type": type(exc).__name__}

    output = {
        "metadata": {
            "scale": scale,
            "np_scale_gev": scale,
            "eft": str(eft),
            "basis": str(basis),
            "normalize": normalize,
            "normalization_definition": "x_i = Lambda^2 * C_i",
            "wilson_coefficient_units": "GeV^-2",
            "finite_difference_method": "central difference with adaptive global step reduction",
            "finite_difference_x_steps_tried": [float(x) for x in fd_steps],
            "finite_difference_wc_steps_tried": [float(x / normalize) for x in fd_steps],
            "coefficient_threshold": float(coefficient_threshold),
            "coefficient_ordering": "descending absolute coefficient magnitude",
            "operators": list(operators),
            "observables": [observable_key(obs) for obs in obs_list],
            "formula": "O/O_SM = 1 + sum_i coefficient_i * (Lambda^2 * C_i)",
        },
        "observables": results,
    }

    path = Path(output_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    return output

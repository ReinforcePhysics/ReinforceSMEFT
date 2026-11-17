from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass(slots=True)
class HyperParameters:
    """Training, optimization, and search settings."""

    n_batches: int = 300
    rollouts_per_batch: int = 25
    max_steps: int = 10
    seed: int = 42

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    gamma: float = 0.99
    entropy_start: float = 1e-2
    entropy_end: float = 1e-4

    use_critic: bool = False
    critic_loss_coef: float = 0.5
    normalize_advantage: bool = True
    return_norm_momentum: float = 0.01

    embed_dim: int = 32
    n_heads: int = 4
    alpha_init: float = 1.0

    min_step_delta_chi2: float = 2.0
    step_cost: float = 0.1
    dead_step_penalty: float = 0.5
    min_reward_delta_chi2: float = 1e-3
    solved_bonus: float = 10.0
    solved_chi2_threshold: float = 1e-6

    novelty_enabled: bool = True
    novelty_weight: float = 80.0
    novelty_archive_delta_chi2_min: float = 100.0

    # Bounds apply to the dimensionless variables x_i = Lambda^2 C_i.
    fit_lower: float = -1.0
    fit_upper: float = 1.0
    fit_tol: float = 1e-10
    fit_max_iter: int = 2000

    deterministic: bool = False
    verbose: bool = True
    device: str = "auto"

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> "HyperParameters":
        if values is None:
            return cls()
        valid = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - valid)
        if unknown:
            raise ValueError(
                "Unknown hyperparameter(s): " + ", ".join(unknown)
                + ". Valid keys are: " + ", ".join(sorted(valid))
            )
        result = cls(**dict(values))
        if result.n_batches < 1 or result.rollouts_per_batch < 1 or result.max_steps < 1:
            raise ValueError("n_batches, rollouts_per_batch, and max_steps must be >= 1")
        if result.embed_dim < 1 or result.n_heads < 1 or result.embed_dim % result.n_heads != 0:
            raise ValueError("embed_dim must be positive and divisible by n_heads")
        if result.fit_lower >= result.fit_upper:
            raise ValueError("fit_lower must be smaller than fit_upper")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

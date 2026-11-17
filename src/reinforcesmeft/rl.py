from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HyperParameters
from .fitting import FitResult, LinearFitter


@dataclass
class RLState:
    pulls: np.ndarray
    chi2: float
    obs_names: list[str]
    active_ops: list[str] = field(default_factory=list)
    step: int = 0

    def copy(self) -> "RLState":
        return RLState(
            pulls=np.asarray(self.pulls, dtype=np.float32).copy(),
            chi2=float(self.chi2),
            obs_names=list(self.obs_names),
            active_ops=list(self.active_ops),
            step=int(self.step),
        )


class SMEFTActorCritic(nn.Module):
    """Actor-critic policy using pulls, chi2, operator masks and step index."""

    def __init__(
        self,
        op_vocab: Sequence[str],
        obs_names: Sequence[str],
        chi2_reference: float,
        hyper: HyperParameters,
    ) -> None:
        super().__init__()
        self.op_vocab = list(op_vocab)
        self.obs_names = list(obs_names)
        self.num_ops = len(self.op_vocab)
        self.num_obs = len(self.obs_names)
        self.embed_dim = hyper.embed_dim
        self.max_steps = hyper.max_steps
        self.op_to_idx = {op: idx for idx, op in enumerate(self.op_vocab)}

        d = hyper.embed_dim
        self.chi2_reference = max(abs(float(chi2_reference)), 1.0)
        self.op_embed = nn.Embedding(self.num_ops, d)
        self.pull_proj = nn.Linear(1, d)
        self.active_embed = nn.Embedding(2, d)
        self.fired_embed = nn.Embedding(2, d)
        self.cross_attn = nn.MultiheadAttention(d, hyper.n_heads, batch_first=True)
        self.global_state_encoder = nn.Sequential(nn.Linear(2, d), nn.Tanh(), nn.Linear(d, d))
        self.actor_head = nn.Sequential(nn.Linear(d * 5, d), nn.Tanh(), nn.Linear(d, 1))
        self.critic_head = nn.Sequential(
            nn.Linear(d * 6, d), nn.ReLU(), nn.Linear(d, max(d // 2, 1)), nn.ReLU(), nn.Linear(max(d // 2, 1), 1)
        )
        self.alpha = nn.Parameter(torch.tensor(float(hyper.alpha_init), dtype=torch.float32))

    def encode_state(self, state: RLState, fired_ops: set[int] | None = None):
        device = self.alpha.device
        pulls = torch.as_tensor(state.pulls, dtype=torch.float32, device=device).flatten()
        if pulls.numel() != self.num_obs:
            raise ValueError(f"Expected {self.num_obs} pulls, got {pulls.numel()}")

        op_ids = torch.arange(self.num_ops, device=device)
        op_repr = self.op_embed(op_ids)
        pull_tokens = self.pull_proj(pulls.unsqueeze(-1)).unsqueeze(0)
        attn_out, attn_weights = self.cross_attn(
            query=op_repr.unsqueeze(0), key=pull_tokens, value=pull_tokens
        )
        attn_out = attn_out.squeeze(0)

        active_mask = torch.zeros(self.num_ops, dtype=torch.long, device=device)
        for op in state.active_ops:
            idx = self.op_to_idx.get(op)
            if idx is not None:
                active_mask[idx] = 1
        fired_mask = torch.zeros(self.num_ops, dtype=torch.long, device=device)
        for idx in fired_ops or set():
            if 0 <= int(idx) < self.num_ops:
                fired_mask[int(idx)] = 1

        global_scalar = torch.tensor(
            [state.chi2 / self.chi2_reference, state.step / max(self.max_steps, 1)],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        global_repr = self.global_state_encoder(global_scalar)
        operator_state = torch.cat(
            [
                op_repr,
                attn_out,
                self.active_embed(active_mask),
                self.fired_embed(fired_mask),
                global_repr.expand(self.num_ops, -1),
            ],
            dim=-1,
        )
        return operator_state, global_repr, attn_weights.squeeze(0), fired_mask

    def forward(self, state: RLState, fired_ops: set[int] | None = None):
        operator_state, global_repr, attention, fired_mask = self.encode_state(state, fired_ops)
        logits = self.actor_head(operator_state).squeeze(-1) * self.alpha
        logits = logits.masked_fill(fired_mask.bool(), -1e9)
        pooled = operator_state.mean(dim=0, keepdim=True)
        value = self.critic_head(torch.cat([pooled, global_repr], dim=-1)).squeeze()
        return logits, value, attention


class RunningReturnNormalizer:
    def __init__(self, momentum: float = 0.01, eps: float = 1e-6) -> None:
        self.momentum = momentum
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def update_and_normalize(self, values: torch.Tensor) -> torch.Tensor:
        batch_mean = float(values.mean().item())
        batch_var = float(values.var(unbiased=False).item())
        if not self.initialized:
            self.mean = batch_mean
            self.var = max(batch_var, self.eps)
            self.initialized = True
        else:
            m = self.momentum
            self.mean = (1 - m) * self.mean + m * batch_mean
            self.var = max((1 - m) * self.var + m * batch_var, self.eps)
        return (values - self.mean) / (self.var**0.5 + self.eps)


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    aa, bb = set(a), set(b)
    union = aa | bb
    return 1.0 if not union else len(aa & bb) / len(union)


def _returns_to_go(rewards: Sequence[float], gamma: float) -> list[float]:
    result = [0.0] * len(rewards)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = float(rewards[i]) + gamma * running
        result[i] = running
    return result


class RLTrainer:
    def __init__(self, fitter: LinearFitter, hyper: HyperParameters) -> None:
        self.fitter = fitter
        self.dataset = fitter.dataset
        self.hyper = hyper
        self.device = self._device(hyper.device)
        vocab = list(self.dataset.operator_vocab)
        if not vocab:
            raise ValueError("No candidate operators remain for RL training")
        self.operator_vocab = vocab
        self.model = SMEFTActorCritic(
            vocab, self.dataset.observables, self.dataset.baseline_chi2, hyper
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=hyper.learning_rate, weight_decay=hyper.weight_decay
        )
        self.return_normalizer = RunningReturnNormalizer(hyper.return_norm_momentum)
        self.rng = np.random.default_rng(hyper.seed)
        torch.manual_seed(hyper.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(hyper.seed)
        self.novelty_archive: list[frozenset[str]] = []
        self.metrics: list[dict[str, float]] = []

    @staticmethod
    def _device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _rollout(self, batch_idx: int, rollout_idx: int) -> dict[str, Any]:
        h = self.hyper
        state = RLState(
            pulls=np.asarray(self.dataset.baseline_pulls, dtype=np.float32),
            chi2=self.dataset.baseline_chi2,
            obs_names=list(self.dataset.observables),
        )
        fired: set[int] = set()
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        history: list[dict[str, Any]] = []

        for step_idx in range(min(h.max_steps, len(self.operator_vocab))):
            logits, value, _ = self.model(state, fired)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits) if h.deterministic else dist.sample()
            idx = int(action.item())
            fired.add(idx)
            op = self.operator_vocab[idx]
            trial_ops = [*state.active_ops, op]
            fit = self.fitter.fit(trial_ops)
            reduction = state.chi2 - fit.chi2
            accepted = reduction > h.min_step_delta_chi2

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            values.append(value)
            old_chi2 = state.chi2
            if accepted:
                state = RLState(
                    pulls=fit.pulls.astype(np.float32),
                    chi2=fit.chi2,
                    obs_names=list(self.dataset.observables),
                    active_ops=trial_ops,
                    step=step_idx + 1,
                )
            else:
                state.step = step_idx + 1

            history.append(
                {
                    "step": step_idx,
                    "proposed_operator": op,
                    "accepted": accepted,
                    "old_chi2": float(old_chi2),
                    "trial_chi2": float(fit.chi2),
                    "chi2_reduction": float(reduction),
                }
            )

        final_fit = self.fitter.fit(state.active_ops)
        delta_chi2 = self.dataset.baseline_chi2 - final_fit.chi2
        physics_reward = delta_chi2 - h.step_cost * len(state.active_ops)
        dead = sum(item["chi2_reduction"] < h.min_reward_delta_chi2 for item in history)
        physics_reward -= h.dead_step_penalty * dead
        if final_fit.chi2 < h.solved_chi2_threshold:
            physics_reward += h.solved_bonus

        jmax = 0.0
        novelty_bonus = 0.0
        if h.novelty_enabled:
            if self.novelty_archive:
                jmax = max(_jaccard(state.active_ops, list(s)) for s in self.novelty_archive)
            novelty_bonus = h.novelty_weight * (1.0 - jmax)
        total_reward = physics_reward + novelty_bonus

        step_rewards = []
        for item in history:
            reward = item["chi2_reduction"] - h.step_cost
            if item["chi2_reduction"] < h.min_reward_delta_chi2:
                reward -= h.dead_step_penalty
            step_rewards.append(float(reward))
        if step_rewards:
            step_rewards[-1] += novelty_bonus
            if final_fit.chi2 < h.solved_chi2_threshold:
                step_rewards[-1] += h.solved_bonus
        returns = _returns_to_go(step_rewards, h.gamma)

        support = frozenset(state.active_ops)
        if (
            h.novelty_enabled
            and delta_chi2 > h.novelty_archive_delta_chi2_min
            and support
            and support not in self.novelty_archive
        ):
            self.novelty_archive.append(support)

        return {
            "batch": batch_idx,
            "rollout": rollout_idx,
            "fit": final_fit,
            "delta_chi2": float(delta_chi2),
            "physics_reward": float(physics_reward),
            "novelty_bonus": float(novelty_bonus),
            "traj_reward": float(total_reward),
            "archive_jaccard": float(jmax),
            "history": history,
            "log_probs": torch.stack(log_probs),
            "entropies": torch.stack(entropies),
            "values": torch.stack(values),
            "returns": torch.tensor(returns, dtype=torch.float32, device=self.device),
        }

    def train(self) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
        h = self.hyper
        archive: list[dict[str, Any]] = []

        for batch in range(h.n_batches):
            self.model.train()
            progress = batch / max(h.n_batches, 1)
            entropy_coef = h.entropy_start * (1.0 - progress) + h.entropy_end * progress
            rollouts = [self._rollout(batch, r) for r in range(h.rollouts_per_batch)]

            flat_log_probs = torch.cat([r["log_probs"] for r in rollouts])
            flat_entropies = torch.cat([r["entropies"] for r in rollouts])
            flat_values = torch.cat([r["values"] for r in rollouts])
            flat_returns = torch.cat([r["returns"] for r in rollouts])
            norm_returns = self.return_normalizer.update_and_normalize(flat_returns)
            raw_advantages = (
                norm_returns - flat_values.detach() if h.use_critic else norm_returns
            )
            advantages = raw_advantages
            if h.normalize_advantage and advantages.numel() > 1 and advantages.std().item() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_loss = -(advantages.detach() * flat_log_probs).mean()
            critic_loss = F.mse_loss(flat_values, norm_returns.detach())
            entropy_loss = -entropy_coef * flat_entropies.mean()
            loss = actor_loss + entropy_loss
            if h.use_critic:
                loss = loss + h.critic_loss_coef * critic_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), h.grad_clip).item())
            self.optimizer.step()

            for item in rollouts:
                fit: FitResult = item["fit"]
                archive.append(
                    {
                        "batch": int(item["batch"]),
                        "rollout": int(item["rollout"]),
                        "operators": list(fit.operators),
                        "n_operators": len(fit.operators),
                        "wc": dict(fit.wc),
                        "fitted_wc": dict(fit.fitted_wc),
                        "chi2": float(fit.chi2),
                        "chi2_min": float(fit.chi2),
                        "delta_chi2": float(item["delta_chi2"]),
                        "delta_chi2_sm": float(self.dataset.sm_chi2 - fit.chi2),
                        "p_value": float(fit.p_value),
                        "n_dof": int(fit.n_dof),
                        "pulls": [float(x) for x in fit.pulls],
                        "predictions": {k: float(v) for k, v in fit.predictions.items()},
                        "traj_reward": float(item["traj_reward"]),
                        "physics_reward": float(item["physics_reward"]),
                        "novelty_bonus": float(item["novelty_bonus"]),
                        "archive_jaccard": float(item["archive_jaccard"]),
                        "history": item["history"],
                    }
                )

            metric = {
                "batch": float(batch),
                "mean_delta_chi2": float(np.mean([r["delta_chi2"] for r in rollouts])),
                "max_delta_chi2": float(np.max([r["delta_chi2"] for r in rollouts])),
                "mean_reward": float(np.mean([r["traj_reward"] for r in rollouts])),
                "actor_loss": float(actor_loss.item()),
                "critic_loss": float(critic_loss.item()),
                "entropy_loss": float(entropy_loss.item()),
                "total_loss": float(loss.item()),
                "grad_norm": grad_norm,
            }
            self.metrics.append(metric)
            if h.verbose:
                print(
                    f"batch={batch:04d} mean_delta_chi2={metric['mean_delta_chi2']:.3f} "
                    f"max_delta_chi2={metric['max_delta_chi2']:.3f} "
                    f"mean_reward={metric['mean_reward']:.3f}"
                )

        return archive, self.metrics

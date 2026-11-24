"""
Genesis-v9 Data-Efficiency Levers (`colab_gpu_trainer/v9_data_efficiency.py`).
Implements the core data-efficiency training stack for 100B-300B token TPU TRC / Modal B200 runs:
  1. `Rho1ExcessLossMasker`: NeurIPS 2024 Selective LM keeping top `keep_ratio` excess-loss tokens.
  2. `compute_ouro_mtp_loss`: Ouro/LoopLM multi-depth expected CE + exit entropy + MTP (k=3) loss.
  3. `WSDScheduler`: Warmup-Stable-Decay LR schedule with explicit annealing phase signaling.
  4. `RegMixProxyScorer`: ICLR 2025 Ridge-Regression domain mixture optimizer over proxy runs.
  5. `OfflineOPDKLLoss`: MobileLLM-R1.5-style offline teacher full-vocab & top-K KL distillation.
  6. `MultiDomainValEvaluator`: Separate EN/VI/Code/Math/SFT-CoT validation + exact BPB tracking.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_LN2 = math.log(2.0)


@dataclass
class Rho1Result:
    kept_frac: float
    mean_excess: float
    kept_loss: float
    full_loss: float
    mask: torch.Tensor


class Rho1ExcessLossMasker:
    """
    Rho-1 Selective Language Modeling (Lin et al., NeurIPS 2024):
    Retains the top `keep_ratio` tokens ranked by excess loss `(L_model - L_ref)` using
    `F.cross_entropy(..., reduction='none')`. Supports offline reference losses or online
    per-token-ID EMA reference loss tracking when `ref_losses` is not precomputed.
    """

    def __init__(
        self, keep_ratio: float = 0.60, vocab_size: int | None = None,
        ema_decay: float = 0.99, floor_weight: float = 0.0,
    ) -> None:
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.keep_ratio = float(keep_ratio)
        self.ema_decay = float(ema_decay)
        self.floor_weight = float(floor_weight)
        self.ref_ema: torch.Tensor | None = (
            torch.full((vocab_size,), math.log(max(2, vocab_size)), dtype=torch.float32)
            if vocab_size is not None else None
        )

    @torch.no_grad()
    def _online_ref_and_update(self, targets: torch.Tensor, tok_loss: torch.Tensor) -> torch.Tensor:
        if self.ref_ema is None:
            return tok_loss.detach().mean().expand_as(tok_loss)
        if self.ref_ema.device != targets.device:
            self.ref_ema = self.ref_ema.to(targets.device)
        flat_t = targets.reshape(-1).clamp(0, self.ref_ema.numel() - 1)
        flat_l = tok_loss.detach().float().reshape(-1)
        ref = self.ref_ema[flat_t].view_as(tok_loss)
        # Scatter-mean online EMA update per unique token ID in batch
        uniq, inv = torch.unique(flat_t, return_inverse=True)
        sums = torch.zeros_like(uniq, dtype=torch.float32).scatter_add_(0, inv, flat_l)
        cnts = torch.zeros_like(uniq, dtype=torch.float32).scatter_add_(0, inv, torch.ones_like(flat_l))
        means = sums / cnts.clamp(min=1.0)
        self.ref_ema[uniq] = self.ema_decay * self.ref_ema[uniq] + (1.0 - self.ema_decay) * means
        return ref

    def select_mask(
        self, per_tok_loss: torch.Tensor, targets: torch.Tensor | None = None,
        ref_losses: torch.Tensor | None = None, valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Compute binary/weighted Rho-1 token mask `(B, T)` keeping top `keep_ratio` valid tokens."""
        loss_det = per_tok_loss.detach().float()
        if ref_losses is not None:
            ref = ref_losses.detach().float()
        elif targets is not None:
            ref = self._online_ref_and_update(targets, loss_det)
        else:
            ref = loss_det.mean().expand_as(loss_det)
        excess = loss_det - ref
        valid = torch.ones_like(loss_det, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
        flat_v = valid.reshape(-1)
        valid_idx = torch.nonzero(flat_v, as_tuple=False).squeeze(-1)
        n_valid = int(valid_idx.numel())
        if n_valid == 0:
            return torch.zeros_like(loss_det), 0.0
        k = max(1, min(n_valid, int(round(n_valid * self.keep_ratio))))
        v_exc = excess.reshape(-1)[valid_idx]
        _, top_rel = torch.topk(v_exc, k, sorted=False)
        flat_w = torch.zeros_like(loss_det.reshape(-1))
        if self.floor_weight > 0.0:
            flat_w[valid_idx] = self.floor_weight
        flat_w[valid_idx[top_rel]] = 1.0
        mean_exc = float(v_exc[top_rel].mean().item())
        return flat_w.view_as(loss_det), mean_exc

    def __call__(
        self, logits: torch.Tensor, targets: torch.Tensor,
        ref_losses: torch.Tensor | None = None, valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Rho1Result]:
        B, T, V = logits.shape
        per_tok = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none").view(B, T)
        w_mask, mean_exc = self.select_mask(per_tok, targets=targets, ref_losses=ref_losses, valid_mask=valid_mask)
        kept_loss = (per_tok * w_mask).sum() / w_mask.sum().clamp(min=1.0)
        v_f = torch.ones_like(per_tok) if valid_mask is None else valid_mask.float()
        full_loss = float(((per_tok.detach() * v_f).sum() / v_f.sum().clamp(min=1.0)).item())
        kept_frac = float((w_mask > 0.5).float().sum().item() / max(1.0, v_f.sum().item()))
        return kept_loss, Rho1Result(kept_frac, mean_exc, float(kept_loss.detach().item()), full_loss, w_mask)


def compute_ouro_mtp_loss(
    out, targets: torch.Tensor, token_mask: torch.Tensor | None = None,
    entropy_beta: float = 0.05, uniform_floor: float = 0.15, mtp_weight: float = 0.30,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Ouro/LoopLM multi-depth expected CE + exit-gate entropy bonus + MTP (k=3) loss."""
    B, T, V = out.logits.shape
    m = torch.ones((B, T), device=targets.device, dtype=torch.float32) if token_mask is None else token_mask.float()
    denom = m.sum().clamp(min=1.0)
    per_depth = torch.stack([
        F.cross_entropy(dl.reshape(-1, V), targets.reshape(-1), reduction="none").view(B, T)
        for dl in out.depth_logits
    ], dim=-1)  # (B, T, R)
    p = out.exit_probs.clamp(min=1e-6)
    ouro_ce = ((((1.0 - uniform_floor) * (per_depth * p).sum(dim=-1) + uniform_floor * per_depth.mean(dim=-1))) * m).sum() / denom
    ent = (-(p * p.log()).sum(dim=-1) * m).sum() / denom
    mtp_loss = targets.new_zeros((), dtype=torch.float32)
    if out.mtp_logits:
        parts = []
        for idx, ml in enumerate(out.mtp_logits):
            off, L = idx + 1, ml.shape[1]
            ce = F.cross_entropy(ml.reshape(-1, V), targets[:, off : off + L].reshape(-1), reduction="none").view(B, L)
            mm = m[:, off : off + L]
            parts.append((ce * mm).sum() / mm.sum().clamp(min=1.0))
        mtp_loss = torch.stack(parts).mean()
    total = ouro_ce - entropy_beta * ent + mtp_weight * mtp_loss
    return total, {
        "ouro_ce": float(ouro_ce.detach().item()),
        "exit_entropy": float(ent.detach().item()),
        "mtp_loss": float(mtp_loss.detach().item()),
    }


class WSDScheduler:
    """Warmup-Stable-Decay LR scheduler (MiniCPM / SmolLM3 / MobileLLM-R1)."""

    def __init__(
        self, total_steps: int, lr_max: float, warmup_steps: int = 200,
        decay_ratio: float = 0.15, min_lr_ratio: float = 0.05, decay_style: str = "cosine",
    ) -> None:
        self.total_steps = max(1, int(total_steps))
        self.lr_max = float(lr_max)
        self.warmup_steps = max(1, min(int(warmup_steps), self.total_steps))
        self.decay_steps = max(1, int(round(self.total_steps * float(decay_ratio))))
        self.stable_end = max(self.warmup_steps, self.total_steps - self.decay_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.decay_style = decay_style

    def phase_at(self, step: int) -> str:
        if step <= self.warmup_steps:
            return "warmup"
        if step <= self.stable_end:
            return "stable"
        return "decay"

    def lr_at(self, step: int) -> float:
        s = max(1, min(int(step), self.total_steps))
        if s <= self.warmup_steps:
            return self.lr_max * (s / float(self.warmup_steps))
        if s <= self.stable_end:
            return self.lr_max
        prog = (s - self.stable_end) / float(max(1, self.total_steps - self.stable_end))
        if self.decay_style == "sqrt":
            fac = 1.0 - math.sqrt(min(1.0, max(0.0, prog)))
        elif self.decay_style == "linear":
            fac = 1.0 - prog
        else:
            fac = 0.5 * (1.0 + math.cos(math.pi * prog))
        return self.lr_max * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * fac)

    def step_optimizer(self, opt, step: int) -> float:
        lr = self.lr_at(step)
        if hasattr(opt, "set_lr"):
            opt.set_lr(lr)
        else:
            for g in opt.param_groups:
                g["lr"] = lr
        return lr


@dataclass
class RegMixResult:
    weights: dict[str, float]
    predicted_loss: float
    r2_fit: float
    coeffs: np.ndarray


class RegMixProxyScorer:
    """
    RegMix (Liu et al., ICLR 2025) lightweight Ridge-Regression mixture optimizer:
    Fits a surrogate predicting target validation loss from proxy domain mixtures, then
    optimizes weights on the constrained probability simplex.
    """

    def __init__(self, alpha: float = 1e-2, min_weight: float = 0.02, max_weight: float = 0.70) -> None:
        self.alpha = float(alpha)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

    @staticmethod
    def sample_dirichlet_mixtures(
        domain_names: Sequence[str], n_trials: int, concentration: float = 1.0, seed: int = 42,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.dirichlet(np.full(len(domain_names), float(concentration)), size=n_trials)

    def fit_and_optimize(
        self, domain_names: Sequence[str], mix_weights: np.ndarray, target_losses: np.ndarray,
        n_candidates: int = 4096, seed: int = 42,
    ) -> RegMixResult:
        X = np.asarray(mix_weights, dtype=np.float64)
        y = np.asarray(target_losses, dtype=np.float64).reshape(-1)
        M, D = X.shape
        if len(domain_names) != D or len(y) != M:
            raise ValueError(f"Shape mismatch: domains={len(domain_names)}, X={X.shape}, y={y.shape}")
        # Design matrix with unpenalized intercept + linear mixture features
        A = np.hstack([np.ones((M, 1), dtype=np.float64), X])
        reg = np.eye(D + 1, dtype=np.float64) * self.alpha
        reg[0, 0] = 0.0
        theta = np.linalg.solve(A.T @ A + reg, A.T @ y)
        preds = A @ theta
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

        # Box-constrained simplex projection: clip(v - lam, lo, hi) summing to 1.0
        lo = min(self.min_weight, 1.0 / D)
        hi = max(self.max_weight, 1.0 / D)

        def _proj_box_simplex(V: np.ndarray) -> np.ndarray:
            a = np.min(V - hi, axis=-1, keepdims=True)
            b = np.max(V - lo, axis=-1, keepdims=True)
            for _ in range(32):
                mid = 0.5 * (a + b)
                s = np.clip(V - mid, lo, hi).sum(axis=-1, keepdims=True)
                a = np.where(s > 1.0, mid, a)
                b = np.where(s <= 1.0, mid, b)
            return np.clip(V - 0.5 * (a + b), lo, hi)

        rng = np.random.default_rng(seed)
        cand = _proj_box_simplex(rng.dirichlet(np.ones(D), size=n_candidates))
        greedy = np.full(D, lo, dtype=np.float64)
        rem = 1.0 - greedy.sum()
        for idx in np.argsort(theta[1:]):
            add = min(hi - greedy[idx], rem)
            greedy[idx] += max(0.0, add)
            rem -= max(0.0, add)
        all_c = np.vstack([cand, greedy[None, :], _proj_box_simplex(X)])
        all_p = theta[0] + all_c @ theta[1:]
        best_i = int(np.argmin(all_p))
        best_w = all_c[best_i]
        return RegMixResult(
            weights={name: round(float(best_w[i]), 4) for i, name in enumerate(domain_names)},
            predicted_loss=float(all_p[best_i]),
            r2_fit=round(r2, 4),
            coeffs=theta,
        )


class OfflineOPDKLLoss:
    """
    Offline Teacher On-Policy Distillation KL helper (MobileLLM-R1.5 style, 0% inference cost):
    Supports full-vocab teacher logits or memory-compact cached Top-K teacher log-probs.
    """

    def __init__(self, temperature: float = 1.5, mode: str = "reverse", eps: float = 1e-8) -> None:
        if mode not in ("reverse", "forward", "jsd"):
            raise ValueError(f"Unsupported KL mode: {mode}")
        self.temperature = float(temperature)
        self.mode = mode
        self.eps = float(eps)

    def _kl_from_logp(self, log_ps: torch.Tensor, log_pt: torch.Tensor) -> torch.Tensor:
        if self.mode == "forward":
            return (log_pt.exp() * (log_pt - log_ps)).sum(dim=-1)
        if self.mode == "reverse":
            return (log_ps.exp() * (log_ps - log_pt)).sum(dim=-1)
        log_m = torch.logaddexp(log_ps, log_pt) - _LN2
        return 0.5 * ((log_pt.exp() * (log_pt - log_m)).sum(dim=-1) + (log_ps.exp() * (log_ps - log_m)).sum(dim=-1))

    def forward_full(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        T = self.temperature
        log_ps = F.log_softmax(student_logits.float() / T, dim=-1)
        log_pt = F.log_softmax(teacher_logits.detach().float() / T, dim=-1)
        kl = self._kl_from_logp(log_ps, log_pt) * (T * T)
        if valid_mask is None:
            return kl.mean()
        m = valid_mask.float()
        return (kl * m).sum() / m.sum().clamp(min=1.0)

    def forward_topk(
        self, student_logits: torch.Tensor, teacher_topk_ids: torch.Tensor,
        teacher_topk_logprobs: torch.Tensor, valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exact KL over cached Top-K teacher log-probs (T=1) plus probability-conserving tail bucket."""
        T = self.temperature
        log_ps_1 = F.log_softmax(student_logits.float(), dim=-1)
        s_topk = torch.gather(log_ps_1, dim=-1, index=teacher_topk_ids.long())
        t_topk = teacher_topk_logprobs.detach().float()
        t_tail = (1.0 - t_topk.exp().sum(dim=-1, keepdim=True).clamp(max=1.0 - self.eps)).clamp(min=self.eps).log()
        s_tail = (1.0 - s_topk.exp().sum(dim=-1, keepdim=True).clamp(max=1.0 - self.eps)).clamp(min=self.eps).log()
        t_aug = F.log_softmax(torch.cat([t_topk, t_tail], dim=-1) / T, dim=-1)
        s_aug = F.log_softmax(torch.cat([s_topk, s_tail], dim=-1) / T, dim=-1)
        kl = self._kl_from_logp(s_aug, t_aug) * (T * T)
        if valid_mask is None:
            return kl.mean()
        m = valid_mask.float()
        return (kl * m).sum() / m.sum().clamp(min=1.0)


@dataclass
class DomainValMetric:
    loss_nats: float
    ppl: float
    bpb: float


class MultiDomainValEvaluator:
    """
    Separate EN/VI/Code/Math/SFT-CoT validation evaluator with tokenizer-independent BPB.
    Prevents the v8 single-shard validation bug where SFT/CoT best checkpoints froze.
    """

    def __init__(
        self,
        domain_batches: Mapping[str, torch.Tensor],
        bytes_per_token: Mapping[str, float] | None = None,
        domain_masks: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self.batches = {k: v for k, v in domain_batches.items()}
        self.bpt = {k: float((bytes_per_token or {}).get(k, 3.2)) for k in self.batches}
        self.masks = dict(domain_masks or {})

    @torch.no_grad()
    def evaluate(
        self, model: nn.Module, recur_depths: Sequence[int] = (1, 2),
        device: str = "cpu", use_amp_bf16: bool = False,
    ) -> dict[str, dict[str, DomainValMetric]]:
        was_training = model.training
        model.eval()
        results: dict[str, dict[str, DomainValMetric]] = {}
        amp_dev = "cuda" if str(device).startswith("cuda") else "cpu"
        for r in recur_depths:
            r_key = f"R{r}"
            dom_out: dict[str, DomainValMetric] = {}
            loss_sum, bpb_sum = 0.0, 0.0
            for dom, batch in self.batches.items():
                bt = batch.to(device)
                x, y = bt[:, :-1], bt[:, 1:]
                if use_amp_bf16 and amp_dev == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = model(x, recur_steps=r)
                else:
                    logits, _ = model(x, recur_steps=r)
                ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view_as(y)
                if dom in self.masks:
                    mm = self.masks[dom].to(device).float()
                    lv = float(((ce * mm).sum() / mm.sum().clamp(min=1.0)).item())
                else:
                    lv = float(ce.mean().item())
                bpb = lv / (_LN2 * max(self.bpt[dom], 1e-3))
                dom_out[dom] = DomainValMetric(round(lv, 4), round(math.exp(min(lv, 15.0)), 2), round(bpb, 4))
                loss_sum += lv
                bpb_sum += bpb
            n_dom = max(1, len(dom_out))
            avg_l = loss_sum / n_dom
            dom_out["overall"] = DomainValMetric(round(avg_l, 4), round(math.exp(min(avg_l, 15.0)), 2), round(bpb_sum / n_dom, 4))
            results[r_key] = dom_out
        if was_training:
            model.train()
        return results

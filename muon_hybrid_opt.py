"""
Muon + AdamW Hybrid Optimizer for Genesis-v9 (`colab_gpu_trainer/muon_hybrid_opt.py`).
Implements Keller Jordan / Moonshot AI `Moonlight-16B` (02/2025) Newton-Schulz-5 Muon:
  - 5-step quintic Newton-Schulz orthogonalization (`a=3.4445, b=-4.7750, c=2.0315`),
    supporting 2D `(m, n)` hidden matrices and batched 3D `(E, m, n)` expert tensors.
  - RMS alignment scale `0.2 * sqrt(max(m, n))` matching AdamW update RMS + decoupled weight decay.
  - Paired with AdamW (`wd=0.0`) for `emb`, `step_emb`, `engram.tables`, `PEER` (`keys_*`, `u_exp`, `v_exp`),
    and 1D norms/biases/scalars.
  - Explicit ablation flag `muon_on_bitlinear: bool = False` (routes STE BitLinear to AdamW by default).
"""
from __future__ import annotations

import math
from typing import Iterable
import torch
import torch.nn as nn

_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315
_SPARSE_NO_DECAY_KEYS = ("engram.tables", "keys_a", "keys_b", "u_exp", "v_exp")
_NO_DECAY_EXACT_KEYS = ("emb.weight", "step_emb.weight", "engram.conv", "exit_gate", "recur_inj", "q_norm_")


def zeropower_via_newtonschulz5(
    G: torch.Tensor, steps: int = 5, eps: float = 1e-7, use_bf16: bool | None = None,
) -> torch.Tensor:
    """5-step quintic Newton-Schulz orthogonalization on last two dims `(..., m, n)`."""
    if G.ndim < 2:
        raise ValueError(f"Expected >=2D tensor for Newton-Schulz, got shape {tuple(G.shape)}")
    bf16 = G.is_cuda if use_bf16 is None else use_bf16
    X = G.to(torch.bfloat16 if bf16 else torch.float32)
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.transpose(-1, -2)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = _NS_B * A + _NS_C * (A @ A)
        X = _NS_A * X + B @ X
    if transposed:
        X = X.transpose(-1, -2)
    return X.to(G.dtype)


def muon_rms_scale(shape: tuple[int, ...]) -> float:
    """Moonlight-16B RMS matching factor `0.2 * sqrt(max(m, n))` for `(..., m, n)`."""
    return 0.2 * math.sqrt(float(max(shape[-2], shape[-1])))


class Muon(torch.optim.Optimizer):
    """Newton-Schulz-5 Orthogonalized Momentum Optimizer with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | list[dict],
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.05,
        ns_steps: int = 5,
        nesterov: bool = True,
        eps: float = 1e-7,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            ns_steps=ns_steps, nesterov=nesterov, eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = float(group["lr"])
            beta = float(group["momentum"])
            wd = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            nesterov = bool(group["nesterov"])
            eps = float(group["eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2 or min(g.shape[-2], g.shape[-1]) <= 1:
                    raise ValueError(f"Muon requires >=2D matrix with min(m,n)>1, got {tuple(p.shape)}")
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.lerp_(g, 1.0 - beta)
                g_in = g.lerp(buf, beta) if nesterov else buf
                orth = zeropower_via_newtonschulz5(g_in, steps=ns_steps, eps=eps)
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(orth.to(p.dtype), alpha=-lr * muon_rms_scale(tuple(p.shape)))
        return loss


def is_bitlinear_param(name: str) -> bool:
    """Identifies BitNet b1.58 STE parameters (e.g. `moe.bit_w_up`, `moe.bit_w_down`, `*bitlinear*`)."""
    low = name.lower()
    return ("bit_w_" in low) or ("bitlinear" in low)


def partition_v9_params(
    model: nn.Module, muon_on_bitlinear: bool = False,
) -> dict[str, list[torch.nn.Parameter]]:
    """
    Partition trainable Genesis-v9 parameters with strict zero-WD protection on sparse tables & 1D norms:
      - `muon_2d`: hidden 2D matrices (DeltaNet QKVO, Attn QKVO/FFN, Engram Wk/Wv, PEER Wq, Router, MTP)
                   plus STE BitLinear 3D expert tensors ONLY when `muon_on_bitlinear=True`.
      - `adamw_bitlinear`: STE BitLinear tensors when `muon_on_bitlinear=False` (decayed AdamW).
      - `adamw_decay`: any remaining >=2D dense weights not eligible for Muon.
      - `adamw_sparse_nodecay`: `engram.tables`, `PEER keys_*/u_exp/v_exp` (`wd=0.0`, higher LR).
      - `adamw_nodecay`: `emb`, `step_emb`, `engram.conv`, `exit_gate`, `recur_inj`, norms, biases (`wd=0.0`).
    """
    groups: dict[str, list[torch.nn.Parameter]] = {
        "muon_2d": [], "adamw_bitlinear": [], "adamw_decay": [],
        "adamw_sparse_nodecay": [], "adamw_nodecay": [],
    }
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if any(k in name for k in _SPARSE_NO_DECAY_KEYS):
            groups["adamw_sparse_nodecay"].append(p)
        elif p.ndim <= 1 or name.endswith(".bias") or any(k in name for k in _NO_DECAY_EXACT_KEYS):
            groups["adamw_nodecay"].append(p)
        elif is_bitlinear_param(name):
            if muon_on_bitlinear and p.ndim >= 2 and min(p.shape[-2], p.shape[-1]) > 1:
                groups["muon_2d"].append(p)
            else:
                groups["adamw_bitlinear"].append(p)
        elif p.ndim == 2 and min(p.shape[0], p.shape[1]) > 1:
            groups["muon_2d"].append(p)
        else:
            groups["adamw_decay"].append(p)
    return groups


class MuonHybridOptimizer:
    """
    Production Genesis-v9 optimizer pairing Newton-Schulz-5 Muon (2D hidden matrices) with
    AdamW (`wd=0.0` on embeddings, Engram hash tables, PEER keys/experts, and 1D norms).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1.2e-3,
        weight_decay: float = 0.05,
        muon_lr_mult: float = 1.0,
        sparse_lr_mult: float = 1.5,
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        muon_on_bitlinear: bool = False,
        use_fused_adamw: bool | None = None,
    ) -> None:
        self.base_lr = float(lr)
        self.muon_lr_mult = float(muon_lr_mult)
        self.sparse_lr_mult = float(sparse_lr_mult)
        self.muon_on_bitlinear = bool(muon_on_bitlinear)
        self.groups = partition_v9_params(model, muon_on_bitlinear=self.muon_on_bitlinear)

        self.muon: Muon | None = None
        if self.groups["muon_2d"]:
            self.muon = Muon(
                [{"params": self.groups["muon_2d"], "lr": lr * muon_lr_mult, "weight_decay": weight_decay}],
                lr=lr * muon_lr_mult, momentum=muon_momentum, weight_decay=weight_decay, ns_steps=muon_ns_steps,
            )

        adam_groups: list[dict] = []
        if self.groups["adamw_bitlinear"]:
            adam_groups.append({"name": "adamw_bitlinear", "params": self.groups["adamw_bitlinear"], "lr": lr, "weight_decay": weight_decay})
        if self.groups["adamw_decay"]:
            adam_groups.append({"name": "adamw_decay", "params": self.groups["adamw_decay"], "lr": lr, "weight_decay": weight_decay})
        if self.groups["adamw_sparse_nodecay"]:
            adam_groups.append({"name": "adamw_sparse_nodecay", "params": self.groups["adamw_sparse_nodecay"], "lr": lr * sparse_lr_mult, "weight_decay": 0.0})
        if self.groups["adamw_nodecay"]:
            adam_groups.append({"name": "adamw_nodecay", "params": self.groups["adamw_nodecay"], "lr": lr, "weight_decay": 0.0})

        all_cuda = all(p.is_cuda for g in adam_groups for p in g["params"]) if adam_groups else False
        fused = all_cuda if use_fused_adamw is None else bool(use_fused_adamw and all_cuda)
        self.adamw: torch.optim.AdamW | None = (
            torch.optim.AdamW(adam_groups, lr=lr, betas=adam_betas, eps=adam_eps, fused=fused) if adam_groups else None
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()

    def set_lr(self, lr: float) -> None:
        self.base_lr = float(lr)
        if self.muon is not None:
            for g in self.muon.param_groups:
                g["lr"] = self.base_lr * self.muon_lr_mult
        if self.adamw is not None:
            for g in self.adamw.param_groups:
                mult = self.sparse_lr_mult if g.get("name") == "adamw_sparse_nodecay" else 1.0
                g["lr"] = self.base_lr * mult

    def group_summary(self) -> dict[str, int]:
        out = {k: sum(p.numel() for p in ps) for k, ps in self.groups.items()}
        out["total"] = sum(out.values())
        return out

    def state_dict(self) -> dict:
        return {
            "base_lr": self.base_lr,
            "muon_on_bitlinear": self.muon_on_bitlinear,
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        if self.muon is not None and state.get("muon") is not None:
            self.muon.load_state_dict(state["muon"])
        if self.adamw is not None and state.get("adamw") is not None:
            self.adamw.load_state_dict(state["adamw"])
        if "base_lr" in state:
            self.set_lr(float(state["base_lr"]))

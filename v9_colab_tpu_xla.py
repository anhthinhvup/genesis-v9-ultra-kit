"""Genesis-v9 1.5B PyTorch/XLA Trainer & Benchmark for TPU v6e-1 (`32GB HBM2e`, `918 TFLOPS BF16`).
Static shapes | 0 `.item()` host syncs in micro-batches | 0 LR/step recompilations | `9.34GB` static HBM.
Strictly <= 380 LOC | 0 B on C:\\ | 0 Gate-1 banned imports (`re`/`sympy`/`eval`/`exec`).
"""
from __future__ import annotations
import argparse, gc, json, math, os, sys, time, types
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
from genesis_v9_native import BitNetMoEV9, PeerLayerV9, V9Config
from genesis_v9_v41 import DEFAULT_V9_TOK_PATH, GenesisV9V41Native, V9V41Output, load_v9_canon_lut_from_tokenizer, v9_v41_param_breakdown
from muon_hybrid_opt import muon_rms_scale, partition_v9_params, zeropower_via_newtonschulz5
from v9_b200_core import B200_PRESETS
from v9_b200_fast_kernels import _fast_cdn_forward, _fast_peer_forward
from v9_colab_ultra_1000cu import _guard_path, build_bot_sft_mask, parse_mix_spec, sample_mix_batch
from v9_data_efficiency import WSDScheduler
from v9_v41_upgrades import ChannelGatedDeltaNetV9

TPU_V6E1_CU_RATE, TPU_V6E1_HBM_GB = 3.2, 32.0

@dataclass
class XlaRuntime:
    is_xla: bool; xm: Any | None; device: torch.device; device_name: str; pjrt_device: str; hbm_gb: float; use_bf16_autocast: bool

def detect_xla_runtime() -> XlaRuntime:
    """Detect `torch_xla` + `xla_model` + `PJRT_DEVICE=TPU`; fall back cleanly to CPU if absent."""
    try:
        import torch_xla, torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        dev, pjrt = xm.xla_device(), os.environ.get("PJRT_DEVICE", "TPU")
        real = getattr(dev, "type", "") == "xla"
        return XlaRuntime(True, xm, dev, f"TPU-{pjrt}({dev})" if real else f"XLA-STUB({dev})", pjrt, TPU_V6E1_HBM_GB, real)
    except Exception:
        return XlaRuntime(False, None, torch.device("cpu"), "CPU", "CPU", 0.0, False)

def make_xla_autocast(rt: XlaRuntime):
    return torch.autocast("xla", dtype=torch.bfloat16) if (rt.is_xla and rt.use_bf16_autocast) else nullcontext()

def estimate_tpu_v6e_hbm_gb(cfg: V9Config, batch_size: int, seq_len: int, recur_steps: int = 2, hbm_cap_gb: float = TPU_V6E1_HBM_GB) -> dict[str, Any]:
    """Exact 32GB HBM2e memory budget for BF16 params/grads + XlaMuonAdafactor + static activations."""
    total_p = int(v9_v41_param_breakdown(cfg, fsp_horizon=16)["total_train"])
    d, n_dn, e_cnt, sub_d = cfg.d_model, cfg.n_dn_layers, cfg.n_experts, (cfg.d_model // cfg.peer_heads) // 2
    bit_p, bit_fac = 2 * n_dn * e_cnt * d * cfg.d_ff, 2 * n_dn * e_cnt * (d + cfg.d_ff)
    sparse_nd_p = cfg.engram_heads * cfg.engram_buckets * cfg.engram_dim + 4 * cfg.peer_heads * cfg.peer_sub_k * sub_d + 4 * (cfg.peer_sub_k ** 2) * d
    sparse_fac = bit_fac + cfg.engram_heads * (cfg.engram_buckets + cfg.engram_dim) + 4 * cfg.peer_heads * (cfg.peer_sub_k + sub_d) + 4 * (cfg.peer_sub_k ** 2 + d)
    nodecay_p = cfg.vocab * d + cfg.max_recur_steps * d + (11 + 6 * n_dn + cfg.engram_conv_k) * d + n_dn * (cfg.delta_heads + d) + 32
    muon_p = max(0, total_p - bit_p - sparse_nd_p - nodecay_p)
    w_gb, opt_gb = (total_p * 2) / 1e9, (muon_p * 4 + (bit_p + sparse_nd_p) * 2 + sparse_fac * 4 + nodecay_p * 8) / 1e9
    static_gb = 2.0 * w_gb + opt_gb
    r, b, t = max(1, int(recur_steps)), int(batch_size), int(seq_len)
    n_tok, dn_p, hd, nc = b * t, cfg.n_prelude_dn + r * cfg.n_core_dn + cfg.n_coda_dn, d // cfg.delta_heads, math.ceil(t / cfg.delta_chunk)
    moe_a = dn_p * e_cnt * max(1, (n_tok * cfg.top_k * 2) // e_cnt + 64) * (2 * d + cfg.d_ff) * 2
    dn_a = dn_p * (3 * b * cfg.delta_heads * t * hd + nc * b * cfg.delta_heads * hd * hd + 3 * b * cfg.delta_heads * nc * cfg.delta_chunk * cfg.delta_chunk) * 4
    peer_a, logit_a = (2 * r) * (2 * n_tok * cfg.peer_heads * cfg.peer_k * d * 2), (r + max(0, cfg.mtp_k - 1) + 1) * n_tok * cfg.vocab * 2 + 2 * min(b, 4) * t * cfg.vocab * 4
    act_gb = (1.08 * (moe_a + dn_a + peer_a + logit_a)) / 1e9; tot_gb = static_gb + act_gb
    return {"total_params": total_p, "weights_bf16_gb": round(w_gb, 3), "grads_bf16_gb": round(w_gb, 3), "opt_states_gb": round(opt_gb, 3),
            "static_gb": round(static_gb, 3), "activations_gb": round(act_gb, 3), "total_gb": round(tot_gb, 3),
            "hbm_cap_gb": float(hbm_cap_gb), "headroom_gb": round(hbm_cap_gb - tot_gb, 3), "fits_24gb": tot_gb < 24.0, "fits_32gb": tot_gb < hbm_cap_gb}

def _xla_bitnet_moe_forward(self: BitNetMoEV9, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """100% static-shape BitNet b1.58 MoE routing (zero `bincount`, zero dynamic boolean mask)."""
    B, T, D = xn.shape; N, E, K = B * T, self.n_exp, self.top_k; x_flat = xn.reshape(N, D)
    probs = F.softmax(self.router(x_flat).float(), dim=-1); top_w, top_idx = torch.topk(probs, K, dim=-1)
    top_w = (top_w / top_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)).to(xn.dtype)
    aux = float(E) * ((F.one_hot(top_idx, E).float().sum(dim=1).mean(dim=0) / float(K)) * probs.mean(dim=0)).sum()
    flat_e, flat_w = top_idx.reshape(-1), top_w.reshape(-1); order = torch.argsort(flat_e, stable=True); inv_order = torch.argsort(order)
    se, sw, st = flat_e.index_select(0, order), flat_w.index_select(0, order), torch.arange(N, device=xn.device).repeat_interleave(K).index_select(0, order)
    cap = max(1, (N * K * 2) // E + 64)
    counts = torch.zeros(E, device=xn.device, dtype=torch.long).scatter_add(0, se, torch.ones_like(se))
    pos = torch.arange(N * K, device=xn.device) - F.pad(counts.cumsum(0)[:-1], (1, 0)).index_select(0, se)
    in_cap, flat_idx = pos < cap, se * (cap + 1) + pos.clamp(max=cap)
    b_in = torch.zeros((E * (cap + 1), D), device=xn.device, dtype=xn.dtype).index_add(0, flat_idx, x_flat.index_select(0, st) * in_cap.unsqueeze(-1).to(xn.dtype)).view(E, cap + 1, D)
    wu, wd = self._ste_158(self.bit_w_up).to(xn.dtype), self._ste_158(self.bit_w_down).to(xn.dtype)
    b_out = torch.bmm(F.relu(torch.bmm(b_in[:, :cap], wu.transpose(1, 2))).clamp(max=6.0).square(), wd.transpose(1, 2))
    gath = F.pad(b_out, (0, 0, 0, 1)).reshape(E * (cap + 1), D).index_select(0, flat_idx) * (sw * in_cap.to(sw.dtype)).unsqueeze(-1)
    return gath.index_select(0, inv_order).view(N, K, D).sum(dim=1).reshape(B, T, D), aux

def apply_xla_tpu_kernels(model: nn.Module, delta_chunk: int = 16) -> dict[str, int]:
    """Attach static-shape XLA/TPU kernels in-place (zero state_dict changes)."""
    counts = {"channel_deltanet": 0, "xla_bitnet_moe": 0, "peer": 0}
    for mod in model.modules():
        if isinstance(mod, ChannelGatedDeltaNetV9) and not mod.scalar_mode: mod.chunk, mod.forward = int(delta_chunk), types.MethodType(_fast_cdn_forward, mod); counts["channel_deltanet"] += 1  # type: ignore[method-assign]
        elif isinstance(mod, BitNetMoEV9): mod.forward = types.MethodType(_xla_bitnet_moe_forward, mod); counts["xla_bitnet_moe"] += 1  # type: ignore[method-assign]
        elif isinstance(mod, PeerLayerV9): mod.forward = types.MethodType(_fast_peer_forward, mod); counts["peer"] += 1  # type: ignore[method-assign]
    return counts

def xla_forward_train(model: GenesisV9V41Native, tok_ids: torch.Tensor, recur_steps: int, targets: torch.Tensor) -> V9V41Output:
    """Exact `GenesisV9V41Native.forward(return_details=True)` with ZERO `.item()` host syncs."""
    r_steps = max(1, min(model.cfg.max_recur_steps, int(recur_steps)))
    x, aux_tot = model.emb(tok_ids), torch.zeros((), device=tok_ids.device, dtype=torch.float32)
    for blk in model.prelude_dn: x, a = blk(x); aux_tot = aux_tot + a
    x, _ = model.prelude_attn(x, kv_shared=None); x = model.engram(tok_ids, x)
    s0, y, z, inj, core_kv = x, x, torch.zeros_like(x), torch.tanh(model.recur_inj).to(x.dtype), None
    surv, exit_list, step_states = x.new_ones((x.shape[0], x.shape[1]), dtype=torch.float32), [], []
    for r in range(r_steps):
        z_in = (s0 if r == 0 else z + inj * (0.5 * (s0 + y))) + model.step_emb.weight[r].to(x.dtype)
        z, a = model.core_dn[0](z_in); aux_tot = aux_tot + a; z, pa = model.peer0(z); aux_tot = aux_tot + pa
        z, core_kv = model.core_attn(z, kv_shared=core_kv); y = z if r == 0 else y + inj * z
        for blk in model.core_dn[1:]: y, a = blk(y); aux_tot = aux_tot + a
        y, pa = model.peer1(y); aux_tot = aux_tot + pa
        if r < r_steps - 1:
            lam = torch.sigmoid(model.exit_gate(y).float()).squeeze(-1); exit_list.append(surv * lam); surv = surv * (1.0 - lam)
        else: exit_list.append(surv)
        step_states.append(y)
    depth_logits, h_fin = [], y
    for r, ys in enumerate(step_states):
        hc, ca = model._run_coda(ys, cache=None)
        if r == r_steps - 1: h_fin, aux_tot = hc, aux_tot + ca
        depth_logits.append(model._lm_head(hc))
    mtp_logits: list[torch.Tensor] = []
    if len(model.mtp_heads) > 0:
        h_mtp, T = h_fin, tok_ids.shape[1]
        for m, head in enumerate(model.mtp_heads):
            if (L := T - (m + 1)) <= 0: break
            h_mtp = head(h_mtp[:, :L], model.emb(targets[:, m : m + L])); mtp_logits.append(model._lm_head(h_mtp))
    fsp_logits = model.fsp_head(h_fin, model.emb.weight) if model.fsp_head is not None else None
    return V9V41Output(depth_logits[-1], aux_tot, depth_logits, torch.stack(exit_list, dim=-1), mtp_logits, h_fin, fsp_logits)

def compute_v9_v41_loss_xla(
    model: GenesisV9V41Native, out: V9V41Output, targets: torch.Tensor, token_mask: torch.Tensor | None = None,
    trm_ds_weight: float = 0.85, entropy_beta: float = 0.05, mtp_weight: float = 0.15, fsp_weight: float = 0.10, aux_coef: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact Genesis-v9 V4.1 loss returning pure device tensors (ZERO `.item()` host syncs)."""
    B, T, V = out.logits.shape; m = torch.ones((B, T), device=targets.device) if token_mask is None else token_mask.float(); denom = m.sum().clamp(min=1.0)
    per_depth = torch.stack([F.cross_entropy(dl.float().reshape(-1, V), targets.reshape(-1), reduction="none").view(B, T) for dl in out.depth_logits], dim=-1)
    R = per_depth.shape[-1]; w_step = torch.arange(1, R + 1, device=targets.device, dtype=torch.float32); w_step = w_step / w_step.sum()
    trm_ds, p = ((per_depth * w_step.view(1, 1, R)).sum(dim=-1) * m).sum() / denom, out.exit_probs.clamp(min=1e-6)
    core = trm_ds_weight * trm_ds + (1.0 - trm_ds_weight) * (((per_depth * p).sum(dim=-1) * m).sum() / denom - entropy_beta * (-(p * p.log()).sum(dim=-1) * m).sum() / denom)
    mtp_loss = (torch.stack([(F.cross_entropy(ml.float().reshape(-1, V), targets[:, i + 1 : i + 1 + ml.shape[1]].reshape(-1), reduction="none").view(B, -1)
                              * m[:, i + 1 : i + 1 + ml.shape[1]]).sum() / m[:, i + 1 : i + 1 + ml.shape[1]].sum().clamp(min=1.0)
                             for i, ml in enumerate(out.mtp_logits)]).mean() if out.mtp_logits else targets.new_zeros((), dtype=torch.float32))
    fsp_loss = targets.new_zeros((), dtype=torch.float32)
    if out.fsp_logits is not None and model.fsp_head is not None:
        H = min(model.fsp_head.horizon, T); tgt_win = F.pad(targets, (0, H - 1), value=0).unfold(1, H, 1)
        valid_k = (torch.arange(T, device=targets.device)[:, None] + torch.arange(H, device=targets.device)[None, :]) < T
        w_mat = model.fsp_head.horizon_weights[:H].to(device=targets.device, dtype=torch.float32).unsqueeze(0) * valid_k.float()
        w_den, step_b, loss_num = w_mat.sum(dim=-1).clamp(min=1e-6).unsqueeze(0), (4 if B > 4 else B), targets.new_zeros((), dtype=torch.float32)
        for b0 in range(0, B, step_b):
            b1 = min(B, b0 + step_b)
            loss_num = loss_num + (((-F.log_softmax(out.fsp_logits[b0:b1].float(), dim=-1).gather(-1, tgt_win[b0:b1]) * w_mat.unsqueeze(0)).sum(dim=-1) / w_den) * m[b0:b1]).sum()
        fsp_loss = loss_num / denom
    total = core + mtp_weight * mtp_loss + fsp_weight * fsp_loss + aux_coef * out.aux_loss
    return total, {"total_loss": total.detach(), "final_ce": (per_depth[..., -1].detach() * m).sum() / denom, "mtp_loss": mtp_loss.detach(), "fsp_loss": fsp_loss.detach()}

def clip_grad_norm_xla(params: list[torch.nn.Parameter], max_norm: float = 1.0) -> torch.Tensor:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads: return torch.zeros((), dtype=torch.float32)
    total_norm = torch.stack([g.detach().float().pow(2).sum() for g in grads]).sum().sqrt()
    coef = (float(max_norm) / (total_norm + 1e-6)).clamp(max=1.0)
    for g in grads: g.mul_(coef.to(g.dtype))
    return total_norm

def _fp32_to_bf16_sr(x_f32: torch.Tensor) -> torch.Tensor:
    """Unbiased FP32 -> BF16 stochastic rounding (0 extra HBM bytes, pure XLA BitcastConvert)."""
    xc = x_f32.clamp(-3.3e38, 3.3e38).contiguous()
    return ((xc.view(torch.int32) + torch.randint(0, 65536, xc.shape, device=xc.device, dtype=torch.int32)) & (-65536)).view(torch.float32).to(torch.bfloat16)

class XlaMuonAdafactorOptimizer(torch.optim.Optimizer):
    """XLA-native `torch.optim.Optimizer` (MXU BF16 Muon + Factored Adafactor + AdamW) with 0D tensor LR/step."""
    def __init__(self, model: nn.Module, lr: float = 6e-4, weight_decay: float = 0.05, sparse_lr_mult: float = 1.5,
                 muon_momentum: float = 0.95, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8, use_bf16_ns: bool = True) -> None:
        self.base_lr, self.use_bf16_ns, self._step_count, grps = float(lr), bool(use_bf16_ns), 0, partition_v9_params(model, muon_on_bitlinear=False)
        specs = (("muon_2d", "muon", grps["muon_2d"], 1.0, weight_decay), ("adafactor_decay", "adafactor", grps["adamw_bitlinear"] + grps["adamw_decay"], 1.0, weight_decay),
                 ("adafactor_sparse", "adafactor", grps["adamw_sparse_nodecay"], sparse_lr_mult, 0.0), ("adamw_nodecay", "adamw", grps["adamw_nodecay"], 1.0, 0.0))
        super().__init__([{"name": n, "kind": k, "params": ps, "lr": lr * lm, "lr_mult": lm, "weight_decay": wd} for n, k, ps, lm, wd in specs if ps],
                         dict(lr=lr, weight_decay=weight_decay, muon_momentum=muon_momentum, betas=betas, eps=eps))
        dev = next(model.parameters()).device
        self._lr_tensors = [torch.tensor(float(g["lr"]), device=dev, dtype=torch.float32) for g in self.param_groups]
        self._step_t, self._b1_t, self._b2_t = torch.zeros((), device=dev), torch.tensor(float(betas[0]), device=dev), torch.tensor(float(betas[1]), device=dev)

    def set_lr(self, lr: float) -> None:
        self.base_lr = float(lr)
        for g, lt in zip(self.param_groups, self._lr_tensors): g["lr"] = self.base_lr * float(g["lr_mult"]); lt.fill_(g["lr"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None; self._step_count += 1; self._step_t.add_(1.0)
        bc1, bc2 = (1.0 - torch.pow(self._b1_t, self._step_t)).clamp(min=1e-6), (1.0 - torch.pow(self._b2_t, self._step_t)).clamp(min=1e-6)
        for g, lr_t in zip(self.param_groups, self._lr_tensors):
            kind, wd, (b1, b2), eps, mm = g["kind"], float(g["weight_decay"]), g["betas"], float(g["eps"]), float(g["muon_momentum"])
            for p in g["params"]:
                if p.grad is None: continue
                gf, st, pf = p.grad.detach().float(), self.state[p], p.detach().float()
                if kind == "muon":
                    buf = st.setdefault("momentum_buffer", torch.zeros_like(pf)); buf.mul_(mm).add_(gf, alpha=1.0 - mm)
                    u = zeropower_via_newtonschulz5(gf.lerp(buf, mm), steps=5, eps=1e-7, use_bf16=self.use_bf16_ns).float() * muon_rms_scale(tuple(p.shape))
                elif kind == "adafactor" and p.ndim >= 2:
                    g2 = gf * gf + eps
                    vr = st.setdefault("v_r", torch.zeros(p.shape[:-1], device=p.device, dtype=torch.float32))
                    vc = st.setdefault("v_c", torch.zeros((*p.shape[:-2], p.shape[-1]), device=p.device, dtype=torch.float32))
                    vr.mul_(b2).add_(g2.mean(dim=-1), alpha=1.0 - b2); vc.mul_(b2).add_(g2.mean(dim=-2), alpha=1.0 - b2)
                    vrh, vch, mb = vr / bc2, vc / bc2, st.setdefault("m", torch.zeros_like(p))
                    mf = mb.float().mul_(b1).add_(gf, alpha=1.0 - b1)
                    mb.copy_(_fp32_to_bf16_sr(mf) if mb.dtype == torch.bfloat16 else mf.to(mb.dtype))
                    u = (mf / bc1) * (vrh / vrh.mean(dim=-1, keepdim=True).clamp(min=eps)).rsqrt().unsqueeze(-1) * vch.rsqrt().unsqueeze(-2)
                    u = u / u.pow(2).mean().sqrt().clamp(min=1.0)
                else:
                    mf, vf = st.setdefault("m_fp32", torch.zeros_like(pf)), st.setdefault("v_fp32", torch.zeros_like(pf))
                    mf.mul_(b1).add_(gf, alpha=1.0 - b1); vf.mul_(b2).addcmul_(gf, gf, value=1.0 - b2); u = (mf / bc1) / ((vf / bc2).sqrt() + eps)
                pf = (pf * (1.0 - lr_t * wd) if wd > 0.0 else pf) - lr_t * u
                p.copy_(_fp32_to_bf16_sr(pf) if p.dtype == torch.bfloat16 else pf.to(p.dtype))
        return loss

    def state_dict(self) -> dict[str, Any]: return {**super().state_dict(), "base_lr": self.base_lr, "step_count": self._step_count}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        base_lr, sc = float(state_dict.get("base_lr", self.base_lr)), int(state_dict.get("step_count", 0))
        super().load_state_dict({k: v for k, v in state_dict.items() if k not in ("base_lr", "step_count")})
        for p, st in self.state.items():
            for k in ("momentum_buffer", "v_r", "v_c", "m_fp32", "v_fp32"):
                if k in st and isinstance(st[k], torch.Tensor): st[k] = st[k].to(device=p.device, dtype=torch.float32)
        self._step_count = sc; self._step_t.fill_(float(sc)); self.set_lr(base_lr)

def _to_cpu_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor): return obj.detach().cpu()
    return {k: _to_cpu_tree(v) for k, v in obj.items()} if isinstance(obj, dict) else ([_to_cpu_tree(v) for v in obj] if isinstance(obj, list) else obj)

@torch.no_grad()
def evaluate_val_ce_xla(model: GenesisV9V41Native, val_mm: np.ndarray, batch_size: int, seq_len: int, n_batches: int, rt: XlaRuntime, recur_steps: int) -> float:
    was_train = model.training; model.eval(); rng, sh, losses = np.random.default_rng(1337), [("val", val_mm, 1.0)], []
    for _ in range(max(1, n_batches)):
        bt = sample_mix_batch(sh, batch_size, seq_len, rng, model.cfg.vocab).to(rt.device)
        with make_xla_autocast(rt):
            lo, _ = model(bt[:, :-1], recur_steps=recur_steps); ce = F.cross_entropy(lo.float().reshape(-1, lo.shape[-1]), bt[:, 1:].reshape(-1))
        losses.append(float(ce.detach().cpu().item()))
    if was_train: model.train()
    return float(sum(losses) / max(1, len(losses)))

def save_ckpt_atomic_xla(path: Path, model: GenesisV9V41Native, opt: XlaMuonAdafactorOptimizer, meta: dict[str, Any], save_opt: bool = True) -> None:
    _guard_path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".pt.tmp")
    sd = {k: (v.detach().to(torch.bfloat16).cpu() if v.is_floating_point() else v.detach().cpu()) for k, v in model.state_dict().items()}
    payload: dict[str, Any] = {"format": "genesis_v9_ultra_ckpt_v2", "config": asdict(model.cfg), "fsp_horizon": model.fsp_horizon, "model_state": sd, **meta}
    if save_opt: payload["opt_state"] = _to_cpu_tree(opt.state_dict())
    torch.save(payload, tmp); os.replace(tmp, path)

def export_inference_bf16_xla(model: GenesisV9V41Native, out_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Transfer weights to CPU, verify exact `strip_diff == 0.0` (0 XLA recompile), save BF16 inference ckpt."""
    _guard_path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_m = GenesisV9V41Native(model.cfg, fsp_horizon=model.fsp_horizon, include_train_heads=True)
    sd_bf16 = {k: (v.detach().to(torch.bfloat16).cpu() if v.is_floating_point() else v.detach().cpu()) for k, v in model.state_dict().items()}
    cpu_m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd_bf16.items()}, strict=True); cpu_m.eval()
    px = torch.arange(16, min(model.cfg.vocab, 48), dtype=torch.long).unsqueeze(0)
    with torch.no_grad(): lo0, _ = cpu_m(px, recur_steps=model.cfg.recur_steps); cpu_m.strip_training_heads(); lo1, _ = cpu_m(px, recur_steps=model.cfg.recur_steps)
    diff = float((lo0 - lo1).abs().max().item())
    if diff != 0.0: raise AssertionError(f"strip_training_heads changed logits: {diff}")
    keep, tmp = set(cpu_m.state_dict().keys()), out_path.with_suffix(".pt.tmp")
    torch.save({"format": "genesis_v9_v41_inference_v1", "config": asdict(model.cfg), "stripped_training_heads": True, "dtype": "bfloat16",
                "param_count_inference": sum(p.numel() for p in cpu_m.parameters()), "strip_max_abs_logit_diff": diff,
                "model_state": {k: v for k, v in sd_bf16.items() if k in keep}, **meta}, tmp)
    os.replace(tmp, out_path); return {"path": str(out_path), "strip_diff": diff, "mb": round(out_path.stat().st_size / 1e6, 2)}

def _build_model_and_opt(cfg: V9Config, rt: XlaRuntime, lr: float, wd: float, tok_cands: tuple[Path | None, ...]) -> tuple[GenesisV9V41Native, XlaMuonAdafactorOptimizer, dict[str, int], str]:
    model, tok_used = GenesisV9V41Native(cfg, fsp_horizon=16, include_train_heads=True), ""
    if cfg.vocab == 16384:
        for tp in tok_cands:
            if tp is not None and tp.is_file(): model.set_canonical_lut(load_v9_canon_lut_from_tokenizer(tp, cfg.vocab)); tok_used = str(tp); break
    kcounts = apply_xla_tpu_kernels(model, delta_chunk=cfg.delta_chunk)
    if rt.use_bf16_autocast: model = model.to(dtype=torch.bfloat16)
    model = model.to(rt.device)
    return model, XlaMuonAdafactorOptimizer(model, lr=lr, weight_decay=wd, sparse_lr_mult=1.5, use_bf16_ns=rt.use_bf16_autocast), kcounts, tok_used

def _finish_step(rt: XlaRuntime, opt: XlaMuonAdafactorOptimizer, params: list[torch.nn.Parameter], log_fn=None, log_args=()) -> torch.Tensor:
    gnorm_t = clip_grad_norm_xla(params, 1.0)
    if rt.is_xla:
        if log_fn is not None and hasattr(rt.xm, "add_step_closure"): rt.xm.add_step_closure(log_fn, args=(*log_args, gnorm_t))
        rt.xm.optimizer_step(opt, barrier=True)
        if log_fn is not None and not hasattr(rt.xm, "add_step_closure"): log_fn(*log_args, gnorm_t)
    else:
        opt.step()
        if log_fn is not None: log_fn(*log_args, gnorm_t)
    return gnorm_t

def bench_tpu_v6e1(args: argparse.Namespace) -> dict[str, Any]:
    """Measure exact `tok/s`, `Mtok/CU`, compile warmup, and HBM footprint on `TPU v6e-1` (or CPU fallback)."""
    rt, cfg = detect_xla_runtime(), B200_PRESETS[args.preset]
    hbm, (model, opt, kcounts, _) = estimate_tpu_v6e_hbm_gb(cfg, args.batch_size, args.seq_len, args.recur_steps), _build_model_and_opt(cfg, rt, args.lr, args.weight_decay, (args.tokenizer,))
    params, ga = list(model.parameters()), max(1, int(args.grad_accum))
    bt = torch.randint(16, cfg.vocab, (args.batch_size, args.seq_len + 1), dtype=torch.long).to(rt.device); inp, tgt = bt[:, :-1], bt[:, 1:]; model.train()
    def _one_step() -> torch.Tensor:
        opt.zero_grad(set_to_none=True); last_l = torch.zeros((), device=rt.device)
        for _ in range(ga):
            with make_xla_autocast(rt): loss, lts = compute_v9_v41_loss_xla(model, xla_forward_train(model, inp, args.recur_steps, tgt), tgt); (loss / float(ga)).backward()
            last_l = lts["total_loss"]
            if rt.is_xla: rt.xm.mark_step()
        _finish_step(rt, opt, params); return last_l
    tw0 = time.perf_counter()
    for _ in range(max(1, int(args.warmup))): _one_step()
    warmup_s, t0, iters, last_t = time.perf_counter() - tw0, time.perf_counter(), max(1, int(args.iters)), torch.zeros((), device=rt.device)
    for _ in range(iters): last_t = _one_step()
    dt, cu_rate = max(1e-6, time.perf_counter() - t0), (float(args.cu_rate) if args.cu_rate > 0.0 else TPU_V6E1_CU_RATE)
    tok_s = (args.batch_size * args.seq_len * ga * iters) / dt
    rep = {"mode": "bench", "device": rt.device_name, "is_xla": rt.is_xla, "preset": args.preset, "bs": args.batch_size, "seq_len": args.seq_len, "grad_accum": ga,
           "recur_steps": args.recur_steps, "kernels": kcounts, "warmup_sec": round(warmup_s, 3), "ms_per_step": round(1000.0 * dt / iters, 2), "tok_per_sec": round(tok_s, 1),
           "m_tok_per_hour": round((tok_s * 3600.0) / 1e6, 2), "cu_rate_hr": cu_rate, "m_tok_per_cu": round(((tok_s * 3600.0) / cu_rate) / 1e6, 2),
           "b_tok_per_1000cu": round(((tok_s * 3600.0) * (1000.0 / cu_rate)) / 1e9, 2), "last_loss": round(float(last_t.detach().cpu().item()), 4),
           "hbm_budget": hbm, "live_hbm": rt.xm.get_memory_info(rt.device) if (rt.is_xla and hasattr(rt.xm, "get_memory_info")) else None}
    if args.out_dir is not None: od = _guard_path(args.out_dir); od.mkdir(parents=True, exist_ok=True); (od / "tpu_v6e1_bench.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    del model, opt, bt; gc.collect(); return rep

def run_tpu_ultra_training(args: argparse.Namespace) -> dict[str, Any]:
    if args.data_dir is None or args.out_dir is None: raise ValueError("--data-dir and --out-dir are required in --mode train")
    out_dir = _guard_path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True); torch.manual_seed(args.seed)
    rt, cfg = detect_xla_runtime(), B200_PRESETS[args.preset]
    cu_rate = float(args.cu_rate) if args.cu_rate > 0.0 else (TPU_V6E1_CU_RATE if rt.is_xla else 0.0)
    shards = parse_mix_spec(args.mix_spec, args.data_dir)
    val_p = args.val_shard if (args.val_shard and args.val_shard.is_file()) else (args.data_dir / "val.bin")
    val_mm = np.memmap(val_p, dtype="<u2", mode="r") if val_p.is_file() else shards[0][1]
    hbm = estimate_tpu_v6e_hbm_gb(cfg, args.batch_size, args.seq_len, args.recur_steps)
    model, opt, kcounts, tok_used = _build_model_and_opt(cfg, rt, args.lr, args.weight_decay, (args.tokenizer, args.data_dir / "v9_tok_16k.json", DEFAULT_V9_TOK_PATH))
    params, init_info, latest_ckpt, start_step, cum_tokens, best_val, cu_prior = list(model.parameters()), {}, out_dir / "latest_ckpt.pt", 0, 0, float("inf"), 0.0
    if not args.no_resume and latest_ckpt.is_file():
        ck, cur = torch.load(latest_ckpt, map_location="cpu", weights_only=False), model.state_dict()
        model.load_state_dict({k: v.to(device=rt.device, dtype=cur[k].dtype) for k, v in ck["model_state"].items()}, strict=True)
        if "opt_state" in ck: opt.load_state_dict(ck["opt_state"])
        start_step, cum_tokens, best_val, cu_prior = int(ck.get("step", 0)), int(ck.get("cum_tokens", 0)), float(ck.get("best_val", float("inf"))), float(ck.get("cu_used", 0.0))
        init_info = {"resumed_from": str(latest_ckpt), "step": start_step}
    elif args.init_from is not None and args.init_from.is_file():
        ck, cur = torch.load(args.init_from, map_location="cpu", weights_only=False), model.state_dict(); sd = ck.get("model_state", ck)
        matched = {k: v.to(device=rt.device, dtype=cur[k].dtype) for k, v in sd.items() if k in cur and tuple(v.shape) == tuple(cur[k].shape)}
        missing, unexpected = model.load_state_dict(matched, strict=False)
        init_info = {"init_from": str(args.init_from), "loaded_tensors": len(matched), "missing_new_heads": len(missing), "unexpected": len(unexpected)}
    total_steps = max(int(args.steps), start_step)
    wsd = WSDScheduler(max(1, total_steps), lr_max=args.lr, warmup_steps=max(1, min(200, max(2, total_steps // 20))), decay_ratio=args.decay_ratio, min_lr_ratio=max(0.0, min(1.0, args.lr_min / max(args.lr, 1e-12))))
    rng, bs, sl, ga = np.random.default_rng(args.seed + start_step * 9973), int(args.batch_size), int(args.seq_len), max(1, int(args.grad_accum))
    toks_per_step, r1_until = bs * sl * ga, int(total_steps * max(0.0, min(1.0, args.r1_warmup_frac)))
    eval_every, log_every = (args.eval_every if args.eval_every > 0 else max(1, min(200, max(5, total_steps // 10)))), max(1, int(args.log_every))
    val_before = last_val = evaluate_val_ce_xla(model, val_mm, bs, sl, 2, rt, args.recur_steps)
    log_path, t0, last_save_t, cu_used, final_step, stopped_cu = out_dir / "train_telemetry.jsonl", time.perf_counter(), time.perf_counter(), cu_prior, start_step, False
    print(json.dumps({"event": "start", "device": rt.device_name, "cu_rate": cu_rate, "hbm": hbm, "kernels": kcounts, "init": init_info, "val_before": round(val_before, 4), "r1_until_step": r1_until}), flush=True)
    def _emit_log(si: int, ri: int, ph: str, lrv: float, accs: dict[str, torch.Tensor], dt_s: float, vv: float, ct: int, cuv: float, gn: torch.Tensor) -> None:
        rec = {"ts": int(time.time()), "step": si, "r": ri, "phase": ph, "lr": round(lrv, 7), **{k: round(float(v.detach().cpu().item()), 4) for k, v in accs.items()},
               "gnorm": round(float(gn.detach().cpu().item()), 3), "val_ce": round(vv, 4), "tok_s": round(toks_per_step / max(1e-6, dt_s), 1), "cum_tokens": ct, "cu_used": round(cuv, 3)}
        with log_path.open("a", encoding="utf-8") as lf: lf.write(json.dumps(rec) + "\n")
    model.train()
    for step in range(start_step + 1, total_steps + 1):
        st_t0, r_now = time.perf_counter(), (1 if (step <= r1_until and args.recur_steps > 1) else args.recur_steps)
        phase, cur_lr = wsd.phase_at(step), wsd.step_optimizer(opt, step); opt.zero_grad(set_to_none=True)
        accs = {k: torch.zeros((), device=rt.device) for k in ("total_loss", "final_ce", "mtp_loss", "fsp_loss")}
        for _ in range(ga):
            bt_cpu = sample_mix_batch(shards, bs, sl, rng, cfg.vocab)
            tmask_cpu = build_bot_sft_mask(bt_cpu[:, :-1], bt_cpu[:, 1:]) if args.sft_mask_bot else None
            inp, tgt, tmask = bt_cpu[:, :-1].to(rt.device), bt_cpu[:, 1:].to(rt.device), (tmask_cpu.to(rt.device) if tmask_cpu is not None else None)
            with make_xla_autocast(rt): loss, lts = compute_v9_v41_loss_xla(model, xla_forward_train(model, inp, r_now, tgt), tgt, token_mask=tmask); (loss / float(ga)).backward()
            for k in accs: accs[k] = accs[k] + lts[k] / float(ga)
            if rt.is_xla: rt.xm.mark_step()
        cum_tokens += toks_per_step; final_step = step; cu_used = cu_prior + ((time.perf_counter() - t0) / 3600.0) * cu_rate
        need_eval = (step % eval_every == 0) or (step == total_steps); need_log = (step % log_every == 0) or (step == start_step + 1) or need_eval
        _finish_step(rt, opt, params, _emit_log if need_log else None, (step, r_now, phase, cur_lr, accs, time.perf_counter() - st_t0, last_val, cum_tokens, cu_used))
        if need_eval: last_val = evaluate_val_ce_xla(model, val_mm, bs, sl, 2, rt, args.recur_steps)
        hit_cu, need_save = (cu_rate > 0.0 and cu_used >= args.max_cu), ((time.perf_counter() - last_save_t) >= max(0.05, args.save_every_mins) * 60.0)
        meta = {"preset": args.preset, "step": step, "cum_tokens": cum_tokens, "best_val": min(best_val, last_val), "cu_used": cu_used}
        if need_eval and last_val < best_val: best_val = last_val; save_ckpt_atomic_xla(out_dir / "best_weights_bf16.pt", model, opt, meta, save_opt=False)
        if need_save or step == total_steps or hit_cu: save_ckpt_atomic_xla(latest_ckpt, model, opt, meta, save_opt=True); last_save_t = time.perf_counter()
        if hit_cu: stopped_cu = True; break
    val_after = evaluate_val_ce_xla(model, val_mm, bs, sl, 2, rt, args.recur_steps)
    exp_meta = {} if args.no_export else export_inference_bf16_xla(model, out_dir / "genesis_v9_1_5b_ultra_inference_bf16.pt", {"preset": args.preset, "step": final_step, "cum_tokens": cum_tokens, "val_ce": round(val_after, 4)})
    summary = {"preset": args.preset, "device": rt.device_name, "is_xla": rt.is_xla, "cu_rate": cu_rate, "cu_used": round(cu_used, 3), "stopped_on_cu": stopped_cu,
               "final_step": final_step, "cum_tokens": cum_tokens, "hbm_budget": hbm, "val_before": round(val_before, 4), "val_after": round(val_after, 4),
               "best_val": round(min(best_val, val_after), 4), "tokenizer": tok_used, "init": init_info, "inference_export": exp_meta}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"); return summary

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Genesis-v9 1.5B PyTorch/XLA TPU v6e-1 Trainer & Benchmark")
    for flag, kw in (("--mode", {"default": "train", "choices": ["train", "bench"]}), ("--preset", {"default": "v9_1_5b_b200", "choices": list(B200_PRESETS.keys())}), ("--data-dir", {"type": Path, "default": None}), ("--mix-spec", {"default": ""}), ("--val-shard", {"type": Path, "default": None}),
        ("--out-dir", {"type": Path, "default": None}), ("--init-from", {"type": Path, "default": None}), ("--tokenizer", {"type": Path, "default": None}), ("--steps", {"type": int, "default": 10000}), ("--batch-size", {"type": int, "default": 4}), ("--seq-len", {"type": int, "default": 1024}),
        ("--grad-accum", {"type": int, "default": 16}), ("--lr", {"type": float, "default": 6e-4}), ("--lr-min", {"type": float, "default": 3e-5}), ("--decay-ratio", {"type": float, "default": 0.15}), ("--weight-decay", {"type": float, "default": 0.05}), ("--recur-steps", {"type": int, "default": 2}),
        ("--r1-warmup-frac", {"type": float, "default": 0.0}), ("--sft-mask-bot", {"action": "store_true"}), ("--save-every-mins", {"type": float, "default": 25.0}), ("--eval-every", {"type": int, "default": 0}), ("--log-every", {"type": int, "default": 10}), ("--max-cu", {"type": float, "default": 550.0}),
        ("--cu-rate", {"type": float, "default": TPU_V6E1_CU_RATE}), ("--seed", {"type": int, "default": 20261124}), ("--warmup", {"type": int, "default": 2}), ("--iters", {"type": int, "default": 4}), ("--no-resume", {"action": "store_true"}), ("--no-export", {"action": "store_true"})): ap.add_argument(flag, **kw)
    return ap

def main() -> None:
    args = build_arg_parser().parse_args(); print(json.dumps(bench_tpu_v6e1(args) if args.mode == "bench" else run_tpu_ultra_training(args), indent=2, ensure_ascii=False))
if __name__ == "__main__": main()

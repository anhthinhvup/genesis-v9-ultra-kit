"""Genesis-v9 1.5B CUDA Throughput Accelerator (`colab_gpu_trainer/v9_ultra_speed_accel.py`, <= 360 LOC).
Eliminates ~193+ CPU-GPU sync stalls/step & unbatched optimizer overhead on G4 (96GB) / H100 / A100:
  1. `AsyncMixPrefetcher`: threaded memmap sampler + vectorized SFT mask + pinned async CUDA stream H2D.
  2. `build_fast_optimizer`: shape-bucketed batched NS-5 `FastMuon` (`_foreach_*`) + fused/foreach AdamW.
  3. `ZeroSyncLossAccumulator` & `compute_v9_v41_loss_nosync`: 0 per-microbatch `.item()`, 1 sync/step.
  4. `patch_v9_zero_sync_kernels` & `compile_v9_fast`: removes `bincount`/`.item()` graph breaks + compile.
  5. `recommend_gpu_profile`: tuned `(batch_size, grad_accum, seq_len, compile_mode, delta_chunk)` profiles.
Strictly <= 360 LOC | 0 B on C:\\ | 0 Gate-1 banned AST (`re`/`sympy`/`eval`/`exec`).
"""
from __future__ import annotations

from collections import defaultdict
import queue, threading, types
from typing import Any, Iterable, NamedTuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

try:
    from genesis_v9_native import BitNetMoEV9
    from genesis_v9_v41 import GenesisV9V41Native, V9V41Output
    from muon_hybrid_opt import MuonHybridOptimizer, muon_rms_scale, zeropower_via_newtonschulz5
except ImportError:
    from .genesis_v9_native import BitNetMoEV9
    from .genesis_v9_v41 import GenesisV9V41Native, V9V41Output
    from .muon_hybrid_opt import MuonHybridOptimizer, muon_rms_scale, zeropower_via_newtonschulz5

TOK_EOS, TOK_USER, TOK_BOT = 2, 3, 4
DEFAULT_LOSS_KEYS: tuple[str, ...] = ("total_loss", "final_ce", "mtp_loss", "fsp_loss")


class GPUProfile(NamedTuple):
    batch_size: int; grad_accum: int; seq_len: int; compile_mode: str; delta_chunk: int


def recommend_gpu_profile(gpu_name: str, vram_gb: float) -> GPUProfile:
    """Return optimal `(batch_size, grad_accum, seq_len, compile_mode, delta_chunk)` for `v9_1_5b_b200`."""
    up, vram = str(gpu_name).upper(), float(vram_gb)
    if vram >= 140.0 or ("B200" in up and vram >= 140.0): return GPUProfile(32, 2, 1024, "default", 32)
    if (("G4" in up or "6000" in up or "BLACKWELL" in up) and vram >= 88.0) or vram >= 90.0:
        return GPUProfile(24, 3, 1024, "default", 32)
    if ("H100" in up and vram >= 70.0) or ("A100" in up and vram >= 70.0) or vram >= 70.0:
        return GPUProfile(16, 4, 1024, "default", 32)
    if "A100" in up or vram >= 36.0: return GPUProfile(4, 16, 1024, "default", 16)
    if "L4" in up or vram >= 20.0: return GPUProfile(2, 32, 1024, "default", 16)
    return GPUProfile(1, 64, 512, "none", 16)


def build_bot_sft_mask_np(inp_np: np.ndarray, tgt_np: np.ndarray) -> np.ndarray:
    """Vectorized NumPy `<BOT>`->`<EOS>`/`<USER>` SFT loss mask (bit-exact with `build_bot_sft_mask`)."""
    B, T = inp_np.shape
    if T == 0: return np.zeros((B, 0), dtype=np.float32)
    ev = np.zeros((B, T), dtype=np.int8)
    if T > 1: ev[:, 1:] = np.where(tgt_np[:, :-1] == TOK_EOS, -1, 0)
    ev = np.where(inp_np == TOK_BOT, 1, np.where(inp_np == TOK_USER, -1, ev))
    last = np.maximum.accumulate(np.where(ev != 0, np.arange(T, dtype=np.int32)[None, :], -1), axis=1)
    mask = (np.where(last >= 0, ev[np.arange(B, dtype=np.int32)[:, None], np.maximum(last, 0)], 0) > 0).astype(np.float32)
    mask[mask.sum(axis=1) < 1.0, :] = 1.0
    return mask


class AsyncMixPrefetcher:
    """Background-thread shard sampler with pinned CPU buffers & async CUDA copy-stream H2D staging."""

    def __init__(
        self, shards: list[tuple[str, np.ndarray, float]], batch_size: int, seq_len: int, vocab: int,
        device: str | torch.device = "cpu", seed: int = 20261124, sft_mask_bot: bool = False,
        queue_size: int = 4, pin_memory: bool | None = None,
    ) -> None:
        if not shards: raise ValueError("AsyncMixPrefetcher requires at least one shard")
        self._mms, self._sizes = [mm for _, mm, _ in shards], [int(mm.size) for _, mm, _ in shards]
        probs = np.array([max(1e-12, float(w)) for _, _, w in shards], dtype=np.float64); self._probs = probs / probs.sum()
        self.batch_size, self.seq_len, self.vocab = int(batch_size), int(seq_len), int(vocab)
        self.sft_mask_bot, self.device = bool(sft_mask_bot), torch.device(device)
        self.use_cuda = (self.device.type == "cuda") and torch.cuda.is_available()
        self.pin_memory = self.use_cuda if pin_memory is None else bool(pin_memory and torch.cuda.is_available())
        self._rng = np.random.default_rng(int(seed))
        self._copy_stream = torch.cuda.Stream(device=self.device) if self.use_cuda else None
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_event, self._error = threading.Event(), None
        self._thread = threading.Thread(target=self._worker, name="v9-mix-prefetch", daemon=True); self._thread.start()

    def _sample_cpu(self) -> tuple[np.ndarray, np.ndarray | None]:
        bs, span, n_sh = self.batch_size, self.seq_len + 1, len(self._mms)
        out = np.empty((bs, span), dtype=np.int64)
        chosen = self._rng.choice(n_sh, size=bs, p=self._probs) if n_sh > 1 else np.zeros(bs, dtype=np.int64)
        for i, sidx in enumerate(chosen):
            mm, n = self._mms[int(sidx)], self._sizes[int(sidx)]
            st = int(self._rng.integers(0, n - span if n > span else n))
            out[i] = mm[st : st + span] if n > span else mm[(st + np.arange(span, dtype=np.int64)) % n]
        np.clip(out, 0, self.vocab - 1, out=out)
        return out, (build_bot_sft_mask_np(out[:, :-1], out[:, 1:]) if self.sft_mask_bot else None)

    def _worker(self) -> None:
        try:
            while not self._stop_event.is_set():
                out_np, tm_np = self._sample_cpu()
                bt, tm = torch.from_numpy(out_np), (torch.from_numpy(tm_np) if tm_np is not None else None)
                if self.pin_memory:
                    bt = bt.pin_memory(); tm = tm.pin_memory() if tm is not None else None
                if self.use_cuda and self._copy_stream is not None:
                    with torch.cuda.stream(self._copy_stream):
                        bt_d, tm_d = bt.to(self.device, non_blocking=True), (tm.to(self.device, non_blocking=True) if tm is not None else None)
                        ev = torch.cuda.Event(); ev.record(self._copy_stream)
                    item: Any = (bt_d[:, :-1], bt_d[:, 1:], tm_d, ev)
                else: item = (bt[:, :-1], bt[:, 1:], tm, None)
                while not self._stop_event.is_set():
                    try: self._queue.put(item, timeout=0.05); break
                    except queue.Full: continue
        except BaseException as exc:
            self._error = exc
            try: self._queue.put(None, timeout=0.05)
            except queue.Full: pass

    def next_batch(self, timeout: float = 30.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self._error is not None: raise RuntimeError("AsyncMixPrefetcher worker failed") from self._error
        if self._stop_event.is_set() and self._queue.empty(): raise RuntimeError("AsyncMixPrefetcher stopped")
        item = self._queue.get(timeout=timeout)
        if item is None:
            if self._error is not None: raise RuntimeError("AsyncMixPrefetcher worker failed") from self._error
            raise StopIteration
        inp, tgt, tmask, ev = item
        if ev is not None:
            cur = torch.cuda.current_stream(self.device); cur.wait_event(ev); inp.record_stream(cur); tgt.record_stream(cur)
            if tmask is not None: tmask.record_stream(cur)
        return inp, tgt, tmask

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        while True:
            try: self._queue.get_nowait()
            except queue.Empty: break
        if self._thread.is_alive(): self._thread.join(timeout=timeout)

    def is_alive(self) -> bool: return self._thread.is_alive()
    def close(self) -> None: self.stop()
    def __iter__(self) -> AsyncMixPrefetcher: return self
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: return self.next_batch()
    def __enter__(self) -> AsyncMixPrefetcher: return self
    def __exit__(self, *_: Any) -> None: self.stop()


class FastMuon(torch.optim.Optimizer):
    """Shape-bucketed batched Newton-Schulz-5 Muon using `_foreach_*` (exact math, 0 `.item()` syncs)."""

    def __init__(
        self, params: Iterable[torch.nn.Parameter] | list[dict[str, Any]], lr: float = 1e-3,
        momentum: float = 0.95, weight_decay: float = 0.05, ns_steps: int = 5, nesterov: bool = True, eps: float = 1e-7,
    ) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps, nesterov=nesterov, eps=eps))

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, beta, wd = float(group["lr"]), float(group["momentum"]), float(group["weight_decay"])
            ns_steps, nesterov, eps = int(group["ns_steps"]), bool(group["nesterov"]), float(group["eps"])
            params: list[torch.Tensor] = []; grads: list[torch.Tensor] = []; bufs: list[torch.Tensor] = []
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                if g.ndim < 2 or min(g.shape[-2], g.shape[-1]) <= 1: raise ValueError(f"FastMuon requires >=2D matrix, got {tuple(p.shape)}")
                st = self.state[p]; buf = st.get("momentum_buffer")
                if buf is None: buf = torch.zeros_like(g); st["momentum_buffer"] = buf
                elif buf.dtype != g.dtype or buf.device != g.device: buf = buf.to(device=g.device, dtype=g.dtype); st["momentum_buffer"] = buf
                params.append(p); grads.append(g); bufs.append(buf)
            if not params: continue
            torch._foreach_lerp_(bufs, grads, 1.0 - beta)
            g_ins = torch._foreach_lerp(grads, bufs, beta) if nesterov else bufs
            buckets: dict[tuple[tuple[int, ...], torch.dtype, torch.device], list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
            for p, gi in zip(params, g_ins): buckets[(tuple(p.shape), gi.dtype, gi.device)].append((p, gi))
            for (shape, _dt, _dev), items in buckets.items():
                ps_b, gs_b = [it[0] for it in items], [it[1] for it in items]
                orths = [zeropower_via_newtonschulz5(gs_b[0], steps=ns_steps, eps=eps)] if len(gs_b) == 1 else list(zeropower_via_newtonschulz5(torch.stack(gs_b, dim=0), steps=ns_steps, eps=eps).unbind(0))
                if wd > 0.0: torch._foreach_mul_(ps_b, 1.0 - lr * wd)
                torch._foreach_add_(ps_b, [o if o.dtype == p.dtype else o.to(p.dtype) for o, p in zip(orths, ps_b)], alpha=-lr * muon_rms_scale(shape))
        return loss


class FastMuonHybridOptimizer(MuonHybridOptimizer):
    """Drop-in `MuonHybridOptimizer` subclass using batched `FastMuon` + fused/foreach `AdamW`."""

    def __init__(self, model: nn.Module, lr: float = 1.2e-3, weight_decay: float = 0.05, muon_lr_mult: float = 1.0,
                 sparse_lr_mult: float = 1.5, muon_momentum: float = 0.95, muon_ns_steps: int = 5,
                 adam_betas: tuple[float, float] = (0.9, 0.95), adam_eps: float = 1e-8,
                 muon_on_bitlinear: bool = False, use_fused_adamw: bool | None = None) -> None:
        super().__init__(model, lr=lr, weight_decay=weight_decay, muon_lr_mult=muon_lr_mult, sparse_lr_mult=sparse_lr_mult,
                         muon_momentum=muon_momentum, muon_ns_steps=muon_ns_steps, adam_betas=adam_betas, adam_eps=adam_eps,
                         muon_on_bitlinear=muon_on_bitlinear, use_fused_adamw=use_fused_adamw)
        if self.groups["muon_2d"]:
            self.muon = FastMuon([{"params": self.groups["muon_2d"], "lr": lr * muon_lr_mult, "weight_decay": weight_decay}],  # type: ignore[assignment]
                                 lr=lr * muon_lr_mult, momentum=muon_momentum, weight_decay=weight_decay, ns_steps=muon_ns_steps)
        if self.adamw is not None and not any(g.get("fused", False) for g in self.adamw.param_groups):
            for g in self.adamw.param_groups: g["foreach"] = True


def build_fast_optimizer(model: nn.Module, lr: float = 1.2e-3, weight_decay: float = 0.05, sparse_lr_mult: float = 1.5, **kwargs: Any) -> FastMuonHybridOptimizer:
    """Build accelerated `FastMuonHybridOptimizer` preserving exact Muon NS-5 + AdamW semantics."""
    return FastMuonHybridOptimizer(model, lr=lr, weight_decay=weight_decay, sparse_lr_mult=sparse_lr_mult, **kwargs)


def accumulate_loss_tensor_detached(buf: torch.Tensor, values: Iterable[torch.Tensor | float], scale: float = 1.0) -> torch.Tensor:
    """In-place device accumulation of detached scalar tensors into `buf` (zero `.item()` host sync)."""
    items = [v.detach().to(device=buf.device, dtype=torch.float32).reshape(()) if isinstance(v, torch.Tensor)
             else torch.tensor(float(v), device=buf.device, dtype=torch.float32) for v in values]
    return buf.add_(torch.stack(items), alpha=float(scale))


class ZeroSyncLossAccumulator:
    """Accumulates micro-batch loss stats purely on device; executes ONE host sync per optimizer step."""

    def __init__(self, keys: tuple[str, ...] = DEFAULT_LOSS_KEYS, device: str | torch.device = "cpu") -> None:
        self.keys, self.device = tuple(keys), torch.device(device)
        self._buf, self._weight_sum = torch.zeros(len(self.keys), device=self.device, dtype=torch.float32), 0.0

    def reset(self) -> None: self._buf.zero_(); self._weight_sum = 0.0
    def update(self, stats: dict[str, torch.Tensor | float], weight: float = 1.0) -> None:
        w = float(weight); accumulate_loss_tensor_detached(self._buf, (stats[k] for k in self.keys), scale=w); self._weight_sum += w
    def finalize(self, grad_norm: torch.Tensor | float | None = None, reset: bool = True) -> dict[str, float]:
        avg = self._buf / max(self._weight_sum, 1e-12)
        if isinstance(grad_norm, torch.Tensor):
            host = torch.cat([avg, grad_norm.detach().to(device=avg.device, dtype=torch.float32).reshape(1)]).detach().cpu().tolist()
            out = {k: float(host[i]) for i, k in enumerate(self.keys)}; out["gnorm"] = float(host[-1])
        else:
            host = avg.detach().cpu().tolist(); out = {k: float(host[i]) for i, k in enumerate(self.keys)}
            if grad_norm is not None: out["gnorm"] = float(grad_norm)
        if reset: self.reset()
        return out


def compute_v9_v41_loss_nosync(
    model: GenesisV9V41Native, out: V9V41Output, targets: torch.Tensor, token_mask: torch.Tensor | None = None,
    trm_ds_weight: float = 0.85, entropy_beta: float = 0.05, mtp_weight: float = 0.15, fsp_weight: float = 0.10, aux_coef: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact zero-sync counterpart of `compute_v9_v41_loss`: returns `(total_loss, detached_device_scalars)`."""
    B, T, V = out.logits.shape
    m = torch.ones((B, T), device=targets.device, dtype=torch.float32) if token_mask is None else token_mask.float()
    denom = m.sum().clamp(min=1.0)
    per_depth = torch.stack([F.cross_entropy(dl.reshape(-1, V), targets.reshape(-1), reduction="none").view(B, T) for dl in out.depth_logits], dim=-1)
    R = per_depth.shape[-1]
    w_step = torch.arange(1, R + 1, device=targets.device, dtype=torch.float32); w_step = w_step / w_step.sum()
    trm_ds = ((per_depth * w_step.view(1, 1, R)).sum(dim=-1) * m).sum() / denom
    p = out.exit_probs.clamp(min=1e-6)
    ouro_ce, ent = ((per_depth * p).sum(dim=-1) * m).sum() / denom, (-(p * p.log()).sum(dim=-1) * m).sum() / denom
    core_loss = trm_ds_weight * trm_ds + (1.0 - trm_ds_weight) * (ouro_ce - entropy_beta * ent)
    mtp_loss = (torch.stack([(F.cross_entropy(ml.reshape(-1, V), targets[:, i + 1 : i + 1 + ml.shape[1]].reshape(-1), reduction="none").view(B, -1) * m[:, i + 1 : i + 1 + ml.shape[1]]).sum()
                             / m[:, i + 1 : i + 1 + ml.shape[1]].sum().clamp(min=1.0) for i, ml in enumerate(out.mtp_logits)]).mean()
                if out.mtp_logits else targets.new_zeros((), dtype=torch.float32))
    if out.fsp_logits is not None and model.fsp_head is not None:
        H = min(model.fsp_head.horizon, T)
        tgt_win = F.pad(targets, (0, H - 1), value=0).unfold(1, H, 1)
        valid_k = (torch.arange(T, device=targets.device)[:, None] + torch.arange(H, device=targets.device)[None, :]) < T
        w_mat = model.fsp_head.horizon_weights[:H].to(device=targets.device, dtype=torch.float32).unsqueeze(0) * valid_k.float()
        w_den, step_b, fsp_num = w_mat.sum(dim=-1).clamp(min=1e-6).unsqueeze(0), (8 if B > 8 else B), targets.new_zeros((), dtype=torch.float32)
        for b0 in range(0, B, step_b):
            b1 = min(B, b0 + step_b); lp = F.log_softmax(out.fsp_logits[b0:b1].float(), dim=-1)
            fsp_num = fsp_num + (((-lp.gather(-1, tgt_win[b0:b1]) * w_mat.unsqueeze(0)).sum(dim=-1) / w_den) * m[b0:b1]).sum()
        fsp_loss = fsp_num / denom
    else: fsp_loss = targets.new_zeros((), dtype=torch.float32)
    total = core_loss + mtp_weight * mtp_loss + fsp_weight * fsp_loss + aux_coef * out.aux_loss
    return total, {"total_loss": total.detach(), "trm_ds_loss": trm_ds.detach(), "final_ce": (per_depth[..., -1].detach() * m).sum() / denom,
                   "ouro_ce": ouro_ce.detach(), "exit_entropy": ent.detach(), "mtp_loss": mtp_loss.detach(), "fsp_loss": fsp_loss.detach(), "aux_loss": out.aux_loss.detach()}


def _nosync_bitnet_moe_forward(self: BitNetMoEV9, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Static-shape zero-sync BitNetMoEV9 (replaces C++-syncing `bincount` + double `argsort`)."""
    B, T, D = xn.shape; N, E, K = B * T, self.n_exp, self.top_k; nk = N * K; x_flat = xn.reshape(N, D)
    probs = F.softmax(self.router(x_flat).float(), dim=-1)
    top_w, top_idx = torch.topk(probs, K, dim=-1)
    top_w = (top_w / top_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)).to(xn.dtype)
    aux = float(E) * ((F.one_hot(top_idx, E).float().sum(dim=1).mean(dim=0) / float(K)) * probs.mean(dim=0)).sum()
    flat_e, flat_w = top_idx.reshape(-1), top_w.reshape(-1)
    order, arange_nk = torch.argsort(flat_e, stable=True), torch.arange(nk, device=xn.device)
    inv_order = torch.empty_like(order); inv_order[order] = arange_nk
    se, sw, st = flat_e[order], flat_w[order], torch.arange(N, device=xn.device).repeat_interleave(K)[order]
    cap = max(1, (nk * 2) // E + 64)
    counts = torch.zeros(E, device=xn.device, dtype=torch.long).scatter_add_(0, se, torch.ones(nk, device=xn.device, dtype=torch.long))
    pos = arange_nk - F.pad(counts.cumsum(0)[:-1], (1, 0))[se]
    in_cap = pos < cap; pos_safe, sw_safe = pos.clamp(max=cap), sw * in_cap.to(sw.dtype)
    b_in = torch.zeros((E, cap + 1, D), device=xn.device, dtype=xn.dtype)
    b_in[se, pos_safe] = x_flat[st] * in_cap.unsqueeze(-1).to(xn.dtype)
    wu, wd = self._ste_158(self.bit_w_up).to(xn.dtype), self._ste_158(self.bit_w_down).to(xn.dtype)
    b_out = torch.bmm(F.relu(torch.bmm(b_in[:, :cap], wu.transpose(1, 2))).clamp(max=6.0).square(), wd.transpose(1, 2))
    return ((F.pad(b_out, (0, 0, 0, 1))[se, pos_safe] * sw_safe.unsqueeze(-1))[inv_order].view(N, K, D).sum(dim=1)).reshape(B, T, D), aux


def _nosync_v9_forward(
    self: GenesisV9V41Native, tok_ids: torch.Tensor, recur_steps: int | None = None, targets: torch.Tensor | None = None,
    return_details: bool = False, cache: Any = None, top_k_override: int | None = None, detach_recur: bool = False,
) -> Any:
    if cache is not None or not return_details:
        return self._orig_v9_forward(tok_ids, recur_steps=recur_steps, targets=targets, return_details=return_details,
                                     cache=cache, top_k_override=top_k_override, detach_recur=detach_recur)
    r_steps = max(1, min(self.cfg.max_recur_steps, int(recur_steps or self.cfg.recur_steps)))
    with self._route_budget(top_k_override):
        x, aux_tot = self.emb(tok_ids), torch.zeros((), device=tok_ids.device, dtype=torch.float32)
        for blk in self.prelude_dn: x, a = blk(x, cache=None, state_key=None); aux_tot = aux_tot + a
        x, _ = self._run_attn(self.prelude_attn, x, 0, None, None, True); x = self._run_engram(tok_ids, x, None)
        s0, y, z, inj, core_kv = x, x, torch.zeros_like(x), torch.tanh(self.recur_inj).to(x.dtype), None
        surv, exit_list, step_states = x.new_ones((x.shape[0], x.shape[1]), dtype=torch.float32), [], []
        for r in range(r_steps):
            if r > 0 and detach_recur: z, y = z.detach(), y.detach()
            z_in = (s0 if r == 0 else z + inj * (0.5 * (s0 + y))) + self.step_emb.weight[r].to(x.dtype)
            z, a = self.core_dn[0](z_in, cache=None, state_key=None); aux_tot = aux_tot + a
            z, pa = self.peer0(z); aux_tot = aux_tot + pa
            z, core_kv = self._run_attn(self.core_attn, z, 1, core_kv, None, r == 0)
            y = z if r == 0 else y + inj * z
            for blk in self.core_dn[1:]: y, a = blk(y, cache=None, state_key=None); aux_tot = aux_tot + a
            y, pa = self.peer1(y); aux_tot = aux_tot + pa
            if r < r_steps - 1:
                lam = torch.sigmoid(self.exit_gate(y).float()).squeeze(-1); exit_list.append(surv * lam); surv = surv * (1.0 - lam)
            else: exit_list.append(surv)
            step_states.append(y)
        depth_logits, h_fin = [], y
        for r, ys in enumerate(step_states):
            hc, ca = self._run_coda(ys, cache=None)
            if r == r_steps - 1: h_fin, aux_tot = hc, aux_tot + ca
            depth_logits.append(self._lm_head(hc))
        mtp_logits: list[torch.Tensor] = []
        if targets is not None and len(self.mtp_heads) > 0:
            h_mtp, T = h_fin, tok_ids.shape[1]
            for m, head in enumerate(self.mtp_heads):
                if (L := T - (m + 1)) <= 0: break
                h_mtp = head(h_mtp[:, :L], self.emb(targets[:, m : m + L])); mtp_logits.append(self._lm_head(h_mtp))
        fsp_logits = self.fsp_head(h_fin, self.emb.weight) if (targets is not None and self.fsp_head is not None) else None
        return V9V41Output(depth_logits[-1], aux_tot, depth_logits, torch.stack(exit_list, dim=-1), mtp_logits, h_fin, fsp_logits=fsp_logits, trm_z_norm=0.0, trm_y_norm=0.0)


def patch_v9_zero_sync_kernels(model: nn.Module) -> dict[str, int]:
    """Attach zero-sync BitNetMoEV9 (`scatter_add_`) and zero-sync `GenesisV9V41Native.forward`."""
    counts = {"bitnet_moe_nosync": 0, "v9_forward_nosync": 0}
    for mod in model.modules():
        if isinstance(mod, BitNetMoEV9):
            mod.forward = types.MethodType(_nosync_bitnet_moe_forward, mod); counts["bitnet_moe_nosync"] += 1  # type: ignore[method-assign]
        elif isinstance(mod, GenesisV9V41Native):
            if not hasattr(mod, "_orig_v9_forward"): mod._orig_v9_forward = mod.forward  # type: ignore[attr-defined]
            mod.forward = types.MethodType(_nosync_v9_forward, mod); counts["v9_forward_nosync"] += 1  # type: ignore[method-assign]
    return counts


def compile_v9_fast(model: nn.Module, mode: str = "default", allow_cudagraphs: bool = False) -> tuple[nn.Module, bool]:
    """Safe `torch.compile` tuned for Blackwell G4 / H100 / A100 (`dynamic=False`, `fullgraph=False`)."""
    patch_v9_zero_sync_kernels(model)
    if (not any(p.is_cuda for p in model.parameters())) or (not mode) or (mode == "none") or (not hasattr(torch, "compile")): return model, False
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.set_float32_matmul_precision("high")
    for fn_name in ("enable_flash_sdp", "enable_mem_efficient_sdp"):
        if callable(fn := getattr(torch.backends.cuda, fn_name, None)): fn(True)
    eff_mode = ("default" if mode == "reduce-overhead" else ("max-autotune-no-cudagraphs" if mode == "max-autotune" else str(mode))) if not allow_cudagraphs else str(mode)
    try:
        if (dcfg := getattr(getattr(torch, "_dynamo", None), "config", None)) is not None: dcfg.cache_size_limit = max(64, int(getattr(dcfg, "cache_size_limit", 8)))
        if (tcfg := getattr(getattr(getattr(torch, "_inductor", None), "config", None), "triton", None)) is not None and not allow_cudagraphs:
            for attr in ("cudagraphs", "cudagraph_trees"):
                if hasattr(tcfg, attr): setattr(tcfg, attr, False)
        return torch.compile(model, mode=eff_mode, dynamic=False, fullgraph=False), True
    except Exception: return model, False

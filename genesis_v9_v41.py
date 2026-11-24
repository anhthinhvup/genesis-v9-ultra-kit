"""
Genesis-v9 + DeepSeek-V4.1 Integrated Architecture (`colab_gpu_trainer/genesis_v9_v41.py`, <= 400 LOC).
Integrates Phase-8 DeepSeek-V4.1 upgrades with `GenesisV9Native`, `MuonHybridOptimizer`, & `v9_tok_16k`:
  - ChannelGatedDeltaNetV9 (per-channel diagonal decay) + BitNet b1.58 MoE + stateful incremental decode.
  - EngramV41Module (2/3/4-gram NFKC + grouped/low-rank context gates) + exact streaming history.
  - Huginn Shared-KV Attention with incremental FP4 E2M1 KV-Cache (`FP4KVCacheV9`) + Asymmetric CED.
  - TRM-style Dual-State Recursive Core `(z scratchpad, y answer)` + Deep Supervision (`0` extra inference params).
  - FSP (Future Summary Prediction) training-only head over next `H=16` tokens (`0` inference FLOPs/params).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import torch, torch.nn as nn, torch.nn.functional as F

try:
    from genesis_v8_native import rms_norm
    from genesis_v9_native import BitNetMoEV9, HuginnSharedKVAttnBlockV9, MTPHeadV9, PeerLayerV9, V9_PRESETS, V9Config, _HASH_PRIMES, build_nfkc_canon_lut
    from v9_v41_upgrades import AsymmetricCEDWrapper, ChannelGatedDeltaNetV9, EngramV41Module, FP4KVCacheV9
except ImportError:
    from .genesis_v8_native import rms_norm
    from .genesis_v9_native import BitNetMoEV9, HuginnSharedKVAttnBlockV9, MTPHeadV9, PeerLayerV9, V9_PRESETS, V9Config, _HASH_PRIMES, build_nfkc_canon_lut
    from .v9_v41_upgrades import AsymmetricCEDWrapper, ChannelGatedDeltaNetV9, EngramV41Module, FP4KVCacheV9

DEFAULT_V9_TOK_PATH = Path(r"E:\learn\project_genesis\.tmp_e\oss_harvest\v9_tok_16k.json")


def load_v9_canon_lut_from_tokenizer(tok_path: Path = DEFAULT_V9_TOK_PATH, vocab_size: int = 16384) -> torch.Tensor:
    """Build DeepSeek Engram NFKC+lowercase canonical LUT directly from `v9_tok_16k.json`."""
    obj = json.loads(tok_path.read_text(encoding="utf-8"))
    texts = list(obj.get("special_tokens", []))[:16] + [bytes([b]).decode("utf-8", errors="replace") for b in range(256)]
    texts.extend(bytes.fromhex(h).decode("utf-8", errors="replace") for h, _, _ in obj.get("pieces", []))
    return build_nfkc_canon_lut(vocab_size, texts[:vocab_size])


def _deltanet_step_loop(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, la: torch.Tensor, beta: torch.Tensor, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    outs = []
    for t in range(q.shape[2]):
        qt, kt, vt = q[:, :, t].float(), k[:, :, t].float(), v[:, :, t].float()
        S = S * torch.exp(la[:, :, t].float()).unsqueeze(-1)
        err = vt - torch.einsum("bhkd,bhk->bhd", S, kt)
        S = S + torch.einsum("bhk,bhd->bhkd", kt, beta[:, :, t].float().unsqueeze(-1) * err)
        outs.append(torch.einsum("bhkd,bhk->bhd", S, qt))
    return torch.stack(outs, dim=2).to(q.dtype), S


def _channel_deltanet_stateful(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, la: torch.Tensor, beta: torch.Tensor,
    state: torch.Tensor | None = None, chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact channel-wise diagonal-decay DeltaNet returning `(out, final_state)`."""
    B, H, T, D = q.shape
    S = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32) if state is None else state.float()
    if T <= 8: return _deltanet_step_loop(q, k, v, la, beta, S)
    pad = (chunk_size - (T % chunk_size)) % chunk_size
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        la, beta = F.pad(la, (0, 0, 0, pad)), F.pad(beta, (0, pad))
    nc, C = q.shape[2] // chunk_size, chunk_size
    qf, kf, vf = (t.float().view(B, H, nc, C, D) for t in (q, k, v))
    gf, bf = la.float().view(B, H, nc, C, D), beta.float().view(B, H, nc, C)
    cg_all = gf.cumsum(dim=-2)
    ed = torch.exp((cg_all.unsqueeze(-2) - cg_all.unsqueeze(-3)).clamp(max=0.0))
    sm = torch.tril(torch.ones((C, C), device=q.device, dtype=torch.bool), diagonal=-1)
    cm = torch.tril(torch.ones((C, C), device=q.device, dtype=torch.bool), diagonal=0)
    A = (bf.unsqueeze(-1) * torch.einsum("...id,...jd,...ijd->...ij", kf, kf, ed)).masked_fill(~sm, 0.0)
    eye = torch.eye(C, device=q.device, dtype=torch.float32)
    Tm = torch.linalg.solve_triangular(eye + A, eye.expand_as(A), upper=False, unitriangular=True)
    W, U = torch.matmul(Tm, kf * bf.unsqueeze(-1) * torch.exp(cg_all.clamp(max=0.0))), torch.matmul(Tm, vf * bf.unsqueeze(-1))
    qkt = torch.einsum("...id,...jd,...ijd->...ij", qf, kf, ed).masked_fill(~cm, 0.0)
    outs = []
    for c in range(nc):
        qc, kc, wc, uc, cg = qf[:, :, c], kf[:, :, c], W[:, :, c], U[:, :, c], cg_all[:, :, c]
        vn = uc - torch.matmul(wc, S)
        outs.append(torch.matmul(qc * torch.exp(cg.clamp(max=0.0)), S) + torch.matmul(qkt[:, :, c], vn))
        if c == nc - 1 and pad:
            rem = C - pad
            _, S = _deltanet_step_loop(qc[:, :, :rem], kc[:, :, :rem], vf[:, :, c, :rem], gf[:, :, c, :rem], bf[:, :, c, :rem], S)
        else:
            ge = cg[:, :, -1:, :]
            S = torch.exp(ge.clamp(max=0.0)).transpose(-1, -2) * S + torch.matmul((kc * torch.exp((ge - cg).clamp(max=0.0))).transpose(-1, -2), vn)
    return torch.stack(outs, dim=2).reshape(B, H, nc * C, D)[:, :, :T, :].to(q.dtype), S


@dataclass
class V9V41InferenceCache:
    """Unified incremental inference cache: FP4 KV-cache + DeltaNet states + Engram history."""
    kv_cache: FP4KVCacheV9; pos_offset: int = 0
    dn_states: dict[str, torch.Tensor] = field(default_factory=dict)
    engram_prev_cids: torch.Tensor | None = None; engram_conv_tail: torch.Tensor | None = None

    def reset(self) -> None:
        self.kv_cache.reset(); self.pos_offset = 0; self.dn_states.clear(); self.engram_prev_cids = self.engram_conv_tail = None


class ChannelDeltaNetMoEBlockV9(nn.Module):
    """Per-channel diagonal-decay Gated DeltaNet (`ChannelGatedDeltaNetV9`) + BitNet b1.58 MoE."""

    def __init__(self, cfg: V9Config) -> None:
        super().__init__()
        self.dn = ChannelGatedDeltaNetV9(cfg.d_model, cfg.delta_heads, chunk_size=cfg.delta_chunk, scalar_mode=False)
        self.norm2 = nn.Parameter(torch.ones(cfg.d_model))
        self.moe = BitNetMoEV9(cfg.d_model, cfg.n_experts, cfg.top_k, cfg.d_ff)

    def forward(self, x: torch.Tensor, cache: V9V41InferenceCache | None = None, state_key: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if cache is None or state_key is None:
            x = self.dn(x)
        else:
            B, T, D = x.shape; dn = self.dn; xn = rms_norm(x, dn.norm)
            q = F.normalize(dn.wq(xn).view(B, T, dn.nh, dn.hd).transpose(1, 2).float(), dim=-1, eps=1e-5).to(x.dtype)
            k = F.normalize(dn.wk(xn).view(B, T, dn.nh, dn.hd).transpose(1, 2).float(), dim=-1, eps=1e-5).to(x.dtype)
            v = dn.wv(xn).view(B, T, dn.nh, dn.hd).transpose(1, 2)
            la = F.logsigmoid(dn.w_alpha(xn).float()).view(B, T, dn.nh, dn.hd).transpose(1, 2).clamp(min=-6.0)
            beta = torch.sigmoid(dn.w_beta(xn).float()).transpose(1, 2)
            prev_s = cache.dn_states.get(state_key)
            if prev_s is None and cache.pos_offset > 0 and state_key.startswith("core_"):
                pfx, _, ridx = state_key.rpartition("_r")
                if ridx and ridx != "0" and (base_s := cache.dn_states.get(f"{pfx}_r0")) is not None:
                    prev_s = base_s.clone()
            o, new_s = _channel_deltanet_stateful(q, k, v, la, beta, state=prev_s, chunk_size=dn.chunk)
            cache.dn_states[state_key] = new_s.detach()
            x = x + dn.wo(o.transpose(1, 2).reshape(B, T, D))
        m, aux = self.moe(rms_norm(x, self.norm2))
        return x + m, aux


class FSPHeadV9(nn.Module):
    """Future Summary Prediction training-only head (compressed BoW over next `H=16` tokens, 0 inference FLOPs)."""

    def __init__(self, d_model: int, horizon: int = 16, decay: float = 0.92) -> None:
        super().__init__()
        self.d_model, self.horizon, self.decay, self.rank = d_model, int(horizon), float(decay), max(32, d_model // 2)
        self.norm_in, self.norm_out = nn.Parameter(torch.ones(d_model)), nn.Parameter(torch.ones(d_model))
        self.proj_down, self.proj_up = nn.Linear(d_model, self.rank, bias=False), nn.Linear(self.rank, d_model, bias=False)
        w = torch.tensor([self.decay ** k for k in range(self.horizon)], dtype=torch.float32)
        self.register_buffer("horizon_weights", w / w.sum(), persistent=False)

    def forward(self, h: torch.Tensor, emb_weight: torch.Tensor) -> torch.Tensor:
        s = h + self.proj_up(F.silu(self.proj_down(rms_norm(h, self.norm_in))))
        return 30.0 * torch.tanh(F.linear(rms_norm(s, self.norm_out), emb_weight.detach().to(h.dtype)) / 30.0)

    def compute_loss(self, fsp_logits: torch.Tensor, targets: torch.Tensor, token_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, float]:
        B, T, _ = fsp_logits.shape; H = min(self.horizon, T)
        log_p = F.log_softmax(fsp_logits.float(), dim=-1)
        w = self.horizon_weights[:H].to(device=fsp_logits.device, dtype=torch.float32)
        nll = torch.zeros((B, T), device=fsp_logits.device, dtype=torch.float32)
        ws = torch.zeros((B, T), device=fsp_logits.device, dtype=torch.float32)
        with torch.no_grad():
            pred_top, hit_fut = fsp_logits.argmax(dim=-1), torch.zeros((B, T), device=fsp_logits.device, dtype=torch.bool)
        for k in range(H):
            L = T - k; tgt_k = targets[:, k : k + L]
            nll[:, :L] = nll[:, :L] + (-log_p[:, :L].gather(-1, tgt_k.unsqueeze(-1)).squeeze(-1)) * w[k]
            ws[:, :L] = ws[:, :L] + w[k]
            if k >= 1:
                with torch.no_grad(): hit_fut[:, :L] |= (pred_top[:, :L] == tgt_k)
        m = torch.ones((B, T), device=targets.device, dtype=torch.float32) if token_mask is None else token_mask.float()
        return ((nll / ws.clamp(min=1e-6)) * m).sum() / m.sum().clamp(min=1.0), (float(hit_fut[:, :-1].float().mean().item()) if T > 1 else 0.0)


@dataclass
class V9V41Output:
    logits: torch.Tensor; aux_loss: torch.Tensor
    depth_logits: list[torch.Tensor]; exit_probs: torch.Tensor
    mtp_logits: list[torch.Tensor]; hidden: torch.Tensor
    fsp_logits: torch.Tensor | None = None; trm_z_norm: float = 0.0; trm_y_norm: float = 0.0


def v9_v41_param_breakdown(
    cfg: V9Config, prefill_top_k: int = 1, decode_top_k: int | None = None,
    prefill_recur: int = 1, decode_recur: int | None = None,
    fsp_horizon: int = 16, prefill_len: int = 2048, decode_len: int = 256,
) -> dict[str, int | float]:
    """Exact analytical parameter & Asymmetric CED FLOPs accounting for `GenesisV9V41Native`."""
    d, n_dn = cfg.d_model, cfg.n_dn_layers
    emb = cfg.vocab * d
    dn_per = 5 * d * d + cfg.delta_heads * d + cfg.delta_heads + 3 * d
    moe_per = d * cfg.n_experts + 2 * cfg.n_experts * d * cfg.d_ff
    attn_per, attn_sh = 4 * d * d + 2 * d * cfg.d_attn_ff + 2 * d, 2 * d * d + 2 * d * cfg.d_attn_ff + 2 * d
    channel_dn, moe, attn = n_dn * dn_per, n_dn * moe_per, 2 * attn_per
    g = max(1, min(cfg.engram_heads, d // 8))
    while d % g != 0: g -= 1
    eng_d, eng_rank = cfg.engram_heads * cfg.engram_dim, max(16, d // 4)
    eng_proj = 2 * eng_d * d + (4 + cfg.engram_conv_k) * d + 2 * d * eng_rank + g
    engram = cfg.engram_heads * cfg.engram_buckets * cfg.engram_dim + eng_proj
    sub_d = (d // cfg.peer_heads) // 2
    peer = 2 * (2 * d + d * d + 2 * cfg.peer_heads * cfg.peer_sub_k * sub_d + 2 * (cfg.peer_sub_k ** 2) * d)
    trm_ctrl = (cfg.max_recur_steps + 3) * d + 1
    mtp_p, fsp_p = max(0, cfg.mtp_k - 1) * (4 * d * d + 3 * d), (2 * d * max(32, d // 2) + 2 * d) if fsp_horizon > 0 else 0
    train_only, dense_inf, sparse_tot = mtp_p + fsp_p, emb + channel_dn + attn + trm_ctrl, moe + engram + peer
    dense_tot, tot_train, tot_inf = dense_inf + train_only, dense_inf + train_only + sparse_tot, dense_inf + sparse_tot

    def _active(tk: int, r: int) -> int:
        moe_a = d * cfg.n_experts + 2 * tk * d * cfg.d_ff
        peer_a = 2 * d + d * d + 2 * cfg.peer_heads * cfg.peer_sub_k * sub_d + 2 * cfg.peer_heads * cfg.peer_k * d
        base = emb + n_dn * (dn_per + moe_a) + 2 * attn_per + eng_proj + eng_d + 2 * peer_a + trm_ctrl
        return int(base + (max(1, r) - 1) * (cfg.n_core_dn * (dn_per + moe_a) + attn_sh + 2 * peer_a))

    p_act, d_act = _active(prefill_top_k, prefill_recur), _active(int(decode_top_k or cfg.top_k), int(decode_recur or cfg.recur_steps))
    sym_f, ced_f, sp = 2 * d_act * (prefill_len + decode_len), 2 * (p_act * prefill_len + d_act * decode_len), max(1, sparse_tot)
    return {
        "emb": emb, "channel_deltanet": channel_dn, "attn": attn, "trm_ouro_ctrl": trm_ctrl, "mtp_heads": mtp_p, "fsp_head": fsp_p,
        "train_only_heads": train_only, "dense_inference": dense_inf, "dense_total": dense_tot, "moe": moe, "engram_v41": engram,
        "peer": peer, "sparse_total": sparse_tot, "total_train": tot_train, "total_inference": tot_inf,
        "trm_extra_inference_params": 0, "fsp_inference_params": 0, "fsp_inference_flops": 0,
        "active_prefill": p_act, "active_decode": d_act, "decode_to_prefill_ratio": round(d_act / max(1, p_act), 3),
        "ced_before_symmetric_gflops": round(sym_f / 1e9, 3), "ced_after_asymmetric_gflops": round(ced_f / 1e9, 3),
        "ced_flops_saved_pct": round(100.0 * (1.0 - ced_f / max(1, sym_f)), 2),
        "moe_pct": round(100.0 * moe / tot_train, 2), "engram_pct": round(100.0 * engram / tot_train, 2),
        "peer_pct": round(100.0 * peer / tot_train, 2), "dense_pct": round(100.0 * dense_tot / tot_train, 2),
        "sparse_moe_pct": round(100.0 * moe / sp, 2), "sparse_engram_pct": round(100.0 * engram / sp, 2), "sparse_peer_pct": round(100.0 * peer / sp, 2),
    }


class GenesisV9V41Native(nn.Module):
    """Genesis-v9 + DeepSeek-V4.1 + TRM Dual-State `(z, y)` Deep Supervision + FSP (H=16)."""

    def __init__(self, cfg: V9Config, fsp_horizon: int = 16, include_train_heads: bool = True) -> None:
        super().__init__()
        self.cfg, self.fsp_horizon = cfg, int(fsp_horizon)
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model); nn.init.normal_(self.emb.weight, 0.0, 0.02)
        self.prelude_dn = nn.ModuleList([ChannelDeltaNetMoEBlockV9(cfg) for _ in range(cfg.n_prelude_dn)])
        self.prelude_attn, self.engram = HuginnSharedKVAttnBlockV9(cfg), EngramV41Module(cfg)
        self.core_dn = nn.ModuleList([ChannelDeltaNetMoEBlockV9(cfg) for _ in range(cfg.n_core_dn)])
        self.peer0, self.core_attn, self.peer1 = PeerLayerV9(cfg), HuginnSharedKVAttnBlockV9(cfg), PeerLayerV9(cfg)
        self.step_emb = nn.Embedding(cfg.max_recur_steps, cfg.d_model); nn.init.normal_(self.step_emb.weight, 0.0, 0.01)
        self.recur_inj = nn.Parameter(torch.full((cfg.d_model,), 0.25))
        self.exit_gate = nn.Linear(cfg.d_model, 1, bias=True); nn.init.zeros_(self.exit_gate.weight); nn.init.constant_(self.exit_gate.bias, -1.0)
        self.coda_dn = nn.ModuleList([ChannelDeltaNetMoEBlockV9(cfg) for _ in range(cfg.n_coda_dn)])
        self.norm_f = nn.Parameter(torch.ones(cfg.d_model))
        self.mtp_heads = nn.ModuleList([MTPHeadV9(cfg.d_model) for _ in range(max(0, cfg.mtp_k - 1))]) if include_train_heads else nn.ModuleList()
        self.fsp_head = FSPHeadV9(cfg.d_model, horizon=fsp_horizon) if (include_train_heads and fsp_horizon > 0) else None

    def set_canonical_lut(self, lut: torch.Tensor) -> None: self.engram.canon_lut.copy_(lut.to(device=self.engram.canon_lut.device, dtype=torch.long))
    def strip_training_heads(self) -> None: self.mtp_heads = nn.ModuleList(); self.fsp_head = None
    def make_inference_cache(self, block_size: int | None = None) -> V9V41InferenceCache:
        hd = self.cfg.d_model // self.cfg.attn_heads
        return V9V41InferenceCache(FP4KVCacheV9(self.cfg.attn_heads, hd, block_size=min(32, hd) if block_size is None else int(block_size), n_layers=2))
    def make_ced_wrapper(self, prefill_top_k: int = 1, decode_top_k: int | None = None, prefill_recur: int = 1, decode_recur: int | None = None) -> AsymmetricCEDWrapper:
        return AsymmetricCEDWrapper(self, prefill_top_k, decode_top_k, prefill_recur, decode_recur)
    def param_breakdown(self, prefill_top_k: int = 1, prefill_recur: int = 1) -> dict[str, int | float]:
        return v9_v41_param_breakdown(self.cfg, prefill_top_k=prefill_top_k, prefill_recur=prefill_recur, fsp_horizon=self.fsp_horizon)
    def _lm_head(self, h: torch.Tensor) -> torch.Tensor:
        return 30.0 * torch.tanh(F.linear(rms_norm(h, self.norm_f), self.emb.weight.to(h.dtype)) / 30.0)
    def _run_coda(self, h: torch.Tensor, cache: V9V41InferenceCache | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        aux = h.new_zeros((), dtype=torch.float32)
        for i, blk in enumerate(self.coda_dn):
            h, a = blk(h, cache=cache, state_key=f"coda_{i}" if cache is not None else None); aux = aux + a
        return h, aux
    def forward_prefill(self, tok_ids: torch.Tensor, cache: V9V41InferenceCache | None = None, prefill_top_k: int = 1, prefill_recur: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(tok_ids, recur_steps=prefill_recur, return_details=False, cache=cache, top_k_override=prefill_top_k)
    def forward_decode(self, tok_ids: torch.Tensor, cache: V9V41InferenceCache | None = None, decode_top_k: int | None = None, decode_recur: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(tok_ids, recur_steps=decode_recur or self.cfg.recur_steps, return_details=False, cache=cache, top_k_override=decode_top_k or self.cfg.top_k)

    @contextmanager
    def _route_budget(self, top_k: int | None):
        if top_k is None: yield; return
        blocks = [b.moe for b in (*self.prelude_dn, *self.core_dn, *self.coda_dn)]
        saved = [b.top_k for b in blocks]
        try:
            for b in blocks: b.top_k = max(1, min(self.cfg.n_experts, int(top_k)))
            yield
        finally:
            for b, k in zip(blocks, saved): b.top_k = k

    @staticmethod
    def _rope_offset(blk: HuginnSharedKVAttnBlockV9, x: torch.Tensor, offset: int) -> torch.Tensor:
        ang = torch.arange(offset, offset + x.shape[2], device=x.device, dtype=torch.float32)[:, None] * blk.inv_freq[None, :]
        c, s = ang.cos().to(x.dtype)[None, None], ang.sin().to(x.dtype)[None, None]
        return torch.stack([x[..., 0::2] * c - x[..., 1::2] * s, x[..., 0::2] * s + x[..., 1::2] * c], dim=-1).flatten(-2)

    def _run_attn(
        self, blk: HuginnSharedKVAttnBlockV9, x: torch.Tensor, layer_idx: int,
        kv_shared: tuple[torch.Tensor, torch.Tensor] | None, cache: V9V41InferenceCache | None, write_cache: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        if cache is None: return blk(x, kv_shared=kv_shared)
        B, T, D = x.shape; xn = rms_norm(x, blk.norm1)
        q = self._rope_offset(blk, blk.wq(xn).view(B, T, blk.nh, blk.hd).transpose(1, 2), cache.pos_offset)
        if write_cache:
            k = self._rope_offset(blk, blk.wk(xn).view(B, T, blk.nh, blk.hd).transpose(1, 2), cache.pos_offset).contiguous()
            v = blk.wv(xn).view(B, T, blk.nh, blk.hd).transpose(1, 2).contiguous()
            cache.kv_cache.append(layer_idx, k, v)
        k_all, v_all = cache.kv_cache.get_kv(layer_idx, dtype=q.dtype); S_len = k_all.shape[2]
        if S_len > blk.win: k_all, v_all, S_len = k_all[:, :, -blk.win :], v_all[:, :, -blk.win :], blk.win
        if T == 1: o = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=False)
        elif cache.pos_offset == 0 and T == S_len and T <= blk.win: o = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=True)
        else:
            dist = torch.arange(S_len - T, S_len, device=x.device)[:, None] - torch.arange(S_len, device=x.device)[None, :]
            o = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=(dist >= 0) & (dist < blk.win))
        x = x + blk.wo(o.transpose(1, 2).reshape(B, T, D))
        return x + blk.ff_dn(F.relu(blk.ff_up(rms_norm(x, blk.norm2))).clamp(max=6.0).square()), None

    def _run_engram(self, tok_ids: torch.Tensor, h: torch.Tensor, cache: V9V41InferenceCache | None) -> torch.Tensor:
        if cache is None: return self.engram(tok_ids, h)
        eng, (B, T, D) = self.engram, h.shape
        cid = eng.canon_lut[tok_ids.clamp(0, eng.vocab - 1)]
        prev = torch.zeros((B, 3), device=cid.device, dtype=torch.long) if cache.engram_prev_cids is None else cache.engram_prev_cids
        full = torch.cat([prev, cid], dim=1); p1, p2, p3 = full[:, 2 : 2 + T], full[:, 1 : 1 + T], full[:, 0:T]
        cache.engram_prev_cids = full[:, -3:].detach()
        embs = [tbl(((((cid * m0) ^ (p1 * m1 + m2)) if i % 3 == 0 else (((cid * m0) ^ (p1 * m1) ^ (p2 * m2)) if i % 3 == 1 else ((cid * m0) ^ (p1 * m1) ^ (p2 * m2) ^ (p3 * (m0 ^ m2))))) & 0x7FFFFFFF) % eng.buckets)
                for i, tbl in enumerate(eng.tables) for (m0, m1, m2) in (_HASH_PRIMES[i % len(_HASH_PRIMES)],)]
        e_raw = torch.cat(embs, dim=-1).to(h.dtype); k, v = eng.wk(e_raw), eng.wv(e_raw)
        hn, kn = rms_norm(h, eng.norm_h).float(), rms_norm(k, eng.norm_k).float()
        hg, kg = hn.view(B, T, eng.gate_groups, eng.g_dim), kn.view(B, T, eng.gate_groups, eng.g_dim)
        ag = torch.sigmoid((hg * kg).sum(dim=-1, keepdim=True) * (eng.g_dim ** -0.5) + eng.head_bias.view(1, 1, -1, 1))
        vg = (ag.expand(-1, -1, -1, eng.g_dim).reshape(B, T, D).to(h.dtype) * torch.sigmoid(eng.ctx_up(F.silu(eng.ctx_down(hn.to(h.dtype)))))) * v
        vn, ck = rms_norm(vg, eng.norm_v).transpose(1, 2), eng.conv_k - 1
        tail = torch.zeros((B, D, ck), device=h.device, dtype=vn.dtype) if cache.engram_conv_tail is None else cache.engram_conv_tail
        c_in = torch.cat([tail, vn], dim=2); cache.engram_conv_tail = c_in[:, :, -ck:].detach()
        return h + vg + F.silu(eng.conv(c_in).transpose(1, 2))

    def forward(
        self, tok_ids: torch.Tensor, recur_steps: int | None = None, targets: torch.Tensor | None = None,
        return_details: bool = False, cache: V9V41InferenceCache | None = None, top_k_override: int | None = None, detach_recur: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | V9V41Output:
        r_steps = max(1, min(self.cfg.max_recur_steps, int(recur_steps or self.cfg.recur_steps)))
        with self._route_budget(top_k_override):
            x, aux_tot = self.emb(tok_ids), torch.zeros((), device=tok_ids.device, dtype=torch.float32)
            for i, blk in enumerate(self.prelude_dn):
                x, a = blk(x, cache=cache, state_key=f"pre_{i}" if cache is not None else None); aux_tot = aux_tot + a
            x, _ = self._run_attn(self.prelude_attn, x, 0, None, cache, True)
            x = self._run_engram(tok_ids, x, cache)
            s0, y, z, inj, core_kv = x, x, torch.zeros_like(x), torch.tanh(self.recur_inj).to(x.dtype), None
            surv, exit_list, step_states = x.new_ones((x.shape[0], x.shape[1]), dtype=torch.float32), [], []
            for r in range(r_steps):
                if r > 0 and detach_recur: z, y = z.detach(), y.detach()
                z_in = (s0 if r == 0 else z + inj * (0.5 * (s0 + y))) + self.step_emb.weight[r].to(x.dtype)
                z, a = self.core_dn[0](z_in, cache=cache, state_key=f"core_0_r{r}" if cache is not None else None); aux_tot = aux_tot + a
                z, pa = self.peer0(z); aux_tot = aux_tot + pa
                z, core_kv = self._run_attn(self.core_attn, z, 1, core_kv, cache, r == 0)
                y = z if r == 0 else y + inj * z
                for j, blk in enumerate(self.core_dn[1:], start=1):
                    y, a = blk(y, cache=cache, state_key=f"core_{j}_r{r}" if cache is not None else None); aux_tot = aux_tot + a
                y, pa = self.peer1(y); aux_tot = aux_tot + pa
                if r < r_steps - 1:
                    lam = torch.sigmoid(self.exit_gate(y).float()).squeeze(-1)
                    exit_list.append(surv * lam); surv = surv * (1.0 - lam)
                else: exit_list.append(surv)
                if return_details: step_states.append(y)
            if cache is not None: cache.pos_offset += int(tok_ids.shape[1])
            if not return_details:
                h_fin, ca = self._run_coda(y, cache=cache)
                return self._lm_head(h_fin), aux_tot + ca
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
                    h_mtp = head(h_mtp[:, :L], self.emb(targets[:, m : m + L]))
                    mtp_logits.append(self._lm_head(h_mtp))
            fsp_logits = self.fsp_head(h_fin, self.emb.weight) if (targets is not None and self.fsp_head is not None) else None
            return V9V41Output(
                depth_logits[-1], aux_tot, depth_logits, torch.stack(exit_list, dim=-1), mtp_logits, h_fin,
                fsp_logits=fsp_logits, trm_z_norm=float(z.detach().float().norm(dim=-1).mean().item()),
                trm_y_norm=float(y.detach().float().norm(dim=-1).mean().item()),
            )


def compute_v9_v41_loss(
    model: GenesisV9V41Native, out: V9V41Output, targets: torch.Tensor, token_mask: torch.Tensor | None = None,
    trm_ds_weight: float = 0.85, entropy_beta: float = 0.05, mtp_weight: float = 0.15, fsp_weight: float = 0.10, aux_coef: float = 0.01,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Combined TRM Deep Supervision (monotone ramp) + Ouro exit gate + MTP + FSP (H=16) + Aux loss."""
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
    mtp_loss = (
        torch.stack([
            (F.cross_entropy(ml.reshape(-1, V), targets[:, i + 1 : i + 1 + ml.shape[1]].reshape(-1), reduction="none").view(B, -1) * m[:, i + 1 : i + 1 + ml.shape[1]]).sum()
            / m[:, i + 1 : i + 1 + ml.shape[1]].sum().clamp(min=1.0)
            for i, ml in enumerate(out.mtp_logits)
        ]).mean() if out.mtp_logits else targets.new_zeros((), dtype=torch.float32)
    )
    fsp_loss, fsp_hit = (model.fsp_head.compute_loss(out.fsp_logits, targets, token_mask=m) if (out.fsp_logits is not None and model.fsp_head is not None) else (targets.new_zeros((), dtype=torch.float32), 0.0))
    total = core_loss + mtp_weight * mtp_loss + fsp_weight * fsp_loss + aux_coef * out.aux_loss
    depth_ces = [round(float(((per_depth[..., r].detach() * m).sum() / denom).item()), 4) for r in range(R)]
    return total, {
        "total_loss": float(total.detach().item()), "trm_ds_loss": float(trm_ds.detach().item()), "final_ce": depth_ces[-1],
        "depth_ces": depth_ces, "ouro_ce": float(ouro_ce.detach().item()), "exit_entropy": float(ent.detach().item()),
        "mtp_loss": float(mtp_loss.detach().item()), "fsp_loss": float(fsp_loss.detach().item()),
        "fsp_plan_hit_h16": round(fsp_hit, 4), "aux_loss": float(out.aux_loss.detach().item()),
    }

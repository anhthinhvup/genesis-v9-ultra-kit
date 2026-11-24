"""
Genesis-v9 Phase 8 DeepSeek-V4.1-Flash Architecture Upgrades (`colab_gpu_trainer/v9_v41_upgrades.py`).
Plug-in PyTorch/CPU-verifiable modules (strictly <= 400 LOC, 0 B written to C:\\):
  1. `FP4KVCacheV9`: Block-wise FP4 E2M1 KV-cache quantization (~2x smaller than Q8, ~7x vs F32).
  2. `AsymmetricCEDWrapper`: Light-prefill / deep-decode MoE + recurrence routing & FLOPs accounting.
  3. `EngramV41Module`: Multi-scale 2/3/4-gram NFKC DeepSeek-V4.1 context-gated Engram.
  4. `ChannelGatedDeltaNetV9`: Per-channel `diag(alpha_t) in R^{d_h}` Gated DeltaNet with exact WY chunking.
"""
from __future__ import annotations

from contextlib import contextmanager
import json, os, time
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from genesis_v8_native import rms_norm
    from genesis_v9_native import DeepSeekEngramV9, GenesisV9Native, V9_PRESETS, V9Config, _HASH_PRIMES, build_nfkc_canon_lut
except ImportError:
    from .genesis_v8_native import rms_norm
    from .genesis_v9_native import DeepSeekEngramV9, GenesisV9Native, V9_PRESETS, V9Config, _HASH_PRIMES, build_nfkc_canon_lut

_FP4_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_FP4_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
DEFAULT_AB_JSON = Path(r"E:\learn\project_genesis\.tmp_e\oss_harvest\v9_upgrades_ab.json")


class FP4KVCacheV9(nn.Module):
    """Block-wise FP4 E2M1 KV-Cache quantizer with uint8 nibble packing + fp16 block scales."""

    def __init__(self, n_heads: int, head_dim: int, block_size: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        if head_dim % block_size != 0 or block_size % 2 != 0:
            raise ValueError(f"head_dim={head_dim} must be divisible by even block_size={block_size}")
        self.n_heads, self.head_dim, self.block_size, self.n_layers = n_heads, head_dim, block_size, n_layers
        self.register_buffer("mids", torch.tensor(_FP4_MIDS, dtype=torch.float32), persistent=False)
        self.register_buffer("lut", torch.tensor(_FP4_LUT, dtype=torch.float32), persistent=False)
        self._store: list[tuple[torch.Tensor, ...] | None] = [None] * n_layers

    def reset(self) -> None:
        self._store = [None] * self.n_layers

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig, D, Bq = x.shape, x.shape[-1], self.block_size
        xb = x.float().reshape(*orig[:-1], D // Bq, Bq)
        scale = (xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 6.0).to(torch.float16)
        xn = xb / scale.float()
        code = torch.bucketize(xn.abs(), self.mids).to(torch.uint8) | ((xn < 0).to(torch.uint8) << 3)
        return (code[..., 0::2] | (code[..., 1::2] << 4)).reshape(*orig[:-1], D // 2).contiguous(), scale.squeeze(-1).contiguous()

    def dequantize(self, packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        orig, Bq, D = packed.shape, self.block_size, packed.shape[-1] * 2
        codes = torch.stack([(packed & 0x0F).long(), ((packed >> 4) & 0x0F).long()], dim=-1).reshape(*orig[:-1], D // Bq, Bq)
        return (self.lut[codes] * scale.float().unsqueeze(-1)).reshape(*orig[:-1], D).to(dtype)

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        entry, prev = (*self.quantize(k), *self.quantize(v)), self._store[layer_idx]
        self._store[layer_idx] = entry if prev is None else tuple(torch.cat([a, b], dim=2) for a, b in zip(prev, entry))

    def get_kv(self, layer_idx: int, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        e = self._store[layer_idx]
        if e is None: raise RuntimeError(f"Layer {layer_idx} KV cache is empty")
        return self.dequantize(e[0], e[1], dtype), self.dequantize(e[2], e[3], dtype)

    def attend(self, layer_idx: int, q: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, *self.get_kv(layer_idx, dtype=q.dtype), is_causal=is_causal)

    def memory_bytes(self) -> int:
        return sum(t.nbytes for e in self._store if e is not None for t in e)

    def bytes_per_token(self) -> float:
        toks = next((int(e[0].shape[0] * e[0].shape[2]) for e in self._store if e is not None), 0)
        return float(self.memory_bytes()) / max(1, toks)


def _q8_roundtrip(x: torch.Tensor, block_size: int = 32) -> tuple[torch.Tensor, int]:
    orig, D = x.shape, x.shape[-1]
    xb = x.float().reshape(*orig[:-1], D // block_size, block_size)
    scale = (xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0).to(torch.float16)
    q = torch.clamp(torch.round(xb / scale.float()), -127.0, 127.0).to(torch.int8)
    return (q.float() * scale.float()).reshape(orig).to(x.dtype), int(q.nbytes + scale.nbytes)


class AsymmetricCEDWrapper(nn.Module):
    """DeepSeek-V4.1 Asymmetric Causal Encoder-Decoder wrapper: light prefill vs deep decode routing."""

    def __init__(
        self, model: GenesisV9Native, prefill_top_k: int = 1, decode_top_k: int | None = None,
        prefill_recur: int = 1, decode_recur: int | None = None,
    ) -> None:
        super().__init__()
        self.model, cfg = model, model.cfg
        self.prefill_top_k = max(1, min(cfg.n_experts, int(prefill_top_k)))
        self.decode_top_k = max(1, min(cfg.n_experts, int(decode_top_k or cfg.top_k)))
        self.prefill_recur = max(1, min(cfg.max_recur_steps, int(prefill_recur)))
        self.decode_recur = max(1, min(cfg.max_recur_steps, int(decode_recur or cfg.recur_steps)))

    @contextmanager
    def _route_budget(self, top_k: int):
        blocks = [b.moe for b in (*self.model.prelude_dn, *self.model.core_dn, *self.model.coda_dn)]
        saved = [b.top_k for b in blocks]
        try:
            for b in blocks:
                b.top_k = top_k
            yield
        finally:
            for b, k in zip(blocks, saved):
                b.top_k = k

    def forward_prefill(self, tok_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with self._route_budget(self.prefill_top_k):
            return self.model(tok_ids, recur_steps=self.prefill_recur)

    def forward_decode(self, tok_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with self._route_budget(self.decode_top_k):
            return self.model(tok_ids, recur_steps=self.decode_recur)

    @staticmethod
    def active_params_for(cfg: V9Config, top_k: int, recur_steps: int) -> int:
        d, r = cfg.d_model, max(1, int(recur_steps))
        dn_per = 4 * d * d + 2 * cfg.delta_heads * d + 2 * cfg.delta_heads + 2 * d
        moe_act = d * cfg.n_experts + 2 * top_k * d * cfg.d_ff
        attn_per = 4 * d * d + 2 * d * cfg.d_attn_ff + 2 * d
        attn_sh = 2 * d * d + 2 * d * cfg.d_attn_ff + 2 * d
        eng_d = cfg.engram_heads * cfg.engram_dim
        engram_act = 2 * eng_d * d + (3 + cfg.engram_conv_k) * d + eng_d
        sub_d = (d // cfg.peer_heads) // 2
        peer_act = 2 * d + d * d + 2 * cfg.peer_heads * cfg.peer_sub_k * sub_d + 2 * cfg.peer_heads * cfg.peer_k * d
        base = cfg.vocab * d + cfg.n_dn_layers * (dn_per + moe_act) + 2 * attn_per + engram_act + 2 * peer_act + (cfg.max_recur_steps + 3) * d + 1
        return int(base + (r - 1) * (cfg.n_core_dn * (dn_per + moe_act) + attn_sh + 2 * peer_act))

    def account_sequence(self, cfg: V9Config, prefill_len: int, decode_len: int) -> dict[str, Any]:
        p_act = self.active_params_for(cfg, self.prefill_top_k, self.prefill_recur)
        d_act = self.active_params_for(cfg, self.decode_top_k, self.decode_recur)
        sym, ced = 2 * d_act * (prefill_len + decode_len), 2 * (p_act * prefill_len + d_act * decode_len)
        return {
            "prefill_len": prefill_len, "decode_len": decode_len,
            "prefill_active_params": p_act, "decode_active_params": d_act,
            "decode_to_prefill_ratio": round(d_act / max(1, p_act), 3),
            "before_symmetric_gflops": round(sym / 1e9, 3), "after_asymmetric_ced_gflops": round(ced / 1e9, 3),
            "total_flops_saved_pct": round(100.0 * (1.0 - ced / max(1, sym)), 2),
            "prefill_flops_speedup_x": round(d_act / max(1, p_act), 3),
        }


class EngramV41Module(nn.Module):
    """DeepSeek-V4.1 Context-Gated Engram: 2/3/4-gram NFKC hash + multi-head & low-rank channel gates."""

    def __init__(self, cfg: V9Config) -> None:
        super().__init__()
        self.vocab, self.d, self.n_heads = cfg.vocab, cfg.d_model, cfg.engram_heads
        self.buckets, self.conv_k = cfg.engram_buckets, cfg.engram_conv_k
        g = max(1, min(cfg.engram_heads, cfg.d_model // 8))
        while self.d % g != 0:
            g -= 1
        self.gate_groups, self.g_dim = g, self.d // g
        self.register_buffer("canon_lut", build_nfkc_canon_lut(cfg.vocab), persistent=True)
        self.tables = nn.ModuleList([nn.Embedding(cfg.engram_buckets, cfg.engram_dim) for _ in range(cfg.engram_heads)])
        for tbl in self.tables:
            nn.init.normal_(tbl.weight, 0.0, 0.02)
        tot_d, rank = cfg.engram_heads * cfg.engram_dim, max(16, self.d // 4)
        self.wk, self.wv = nn.Linear(tot_d, self.d, bias=False), nn.Linear(tot_d, self.d, bias=False)
        self.norm_h, self.norm_k, self.norm_v = (nn.Parameter(torch.ones(self.d)) for _ in range(3))
        self.head_bias = nn.Parameter(torch.zeros(self.gate_groups))
        self.ctx_down, self.ctx_up = nn.Linear(self.d, rank, bias=False), nn.Linear(rank, self.d, bias=True)
        nn.init.zeros_(self.ctx_up.bias)
        self.conv = nn.Conv1d(self.d, self.d, kernel_size=cfg.engram_conv_k, groups=self.d, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, tok_ids: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        cid = self.canon_lut[tok_ids.clamp(0, self.vocab - 1)]
        p1, p2, p3 = F.pad(cid[:, :-1], (1, 0)), F.pad(cid[:, :-2], (2, 0)), F.pad(cid[:, :-3], (3, 0))
        embs = []
        for i, tbl in enumerate(self.tables):
            m0, m1, m2 = _HASH_PRIMES[i % len(_HASH_PRIMES)]
            hv = ((cid * m0) ^ (p1 * m1 + m2)) if i % 3 == 0 else (
                ((cid * m0) ^ (p1 * m1) ^ (p2 * m2)) if i % 3 == 1 else ((cid * m0) ^ (p1 * m1) ^ (p2 * m2) ^ (p3 * (m0 ^ m2)))
            )
            embs.append(tbl((hv & 0x7FFFFFFF) % self.buckets))
        e_raw = torch.cat(embs, dim=-1).to(h.dtype)
        k, v = self.wk(e_raw), self.wv(e_raw)
        hn, kn = rms_norm(h, self.norm_h).float(), rms_norm(k, self.norm_k).float()
        B, T, D = h.shape
        hg, kg = hn.view(B, T, self.gate_groups, self.g_dim), kn.view(B, T, self.gate_groups, self.g_dim)
        alpha_g = torch.sigmoid((hg * kg).sum(dim=-1, keepdim=True) * (self.g_dim ** -0.5) + self.head_bias.view(1, 1, -1, 1))
        gamma = torch.sigmoid(self.ctx_up(F.silu(self.ctx_down(hn.to(h.dtype)))))
        vg = (alpha_g.expand(-1, -1, -1, self.g_dim).reshape(B, T, D).to(h.dtype) * gamma) * v
        return h + vg + F.silu(self.conv(F.pad(rms_norm(vg, self.norm_v).transpose(1, 2), (self.conv_k - 1, 0))).transpose(1, 2))


def chunkwise_channel_deltanet(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    log_alpha: torch.Tensor, beta: torch.Tensor, chunk_size: int = 32,
) -> torch.Tensor:
    """Exact Chunkwise WY Parallel Channel-Wise Diagonal-Decay Gated DeltaNet (`diag(alpha_t) in R^{d_h}`)."""
    B, H, T, D = q.shape
    pad = (chunk_size - (T % chunk_size)) % chunk_size
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        log_alpha, beta = F.pad(log_alpha, (0, 0, 0, pad)), F.pad(beta, (0, pad))
    T_pad, C, in_dtype = q.shape[2], chunk_size, q.dtype
    nc = T_pad // C
    qf, kf, vf = (t.float().view(B, H, nc, C, D) for t in (q, k, v))
    gf, bf = log_alpha.float().view(B, H, nc, C, D), beta.float().view(B, H, nc, C)
    cum_g = gf.cumsum(dim=-2)
    exp_diff = torch.exp((cum_g.unsqueeze(-2) - cum_g.unsqueeze(-3)).clamp(max=0.0))
    strict_m = torch.tril(torch.ones((C, C), device=q.device, dtype=torch.bool), diagonal=-1)
    causal_m = torch.tril(torch.ones((C, C), device=q.device, dtype=torch.bool), diagonal=0)
    A = (bf.unsqueeze(-1) * torch.einsum("...id,...jd,...ijd->...ij", kf, kf, exp_diff)).masked_fill(~strict_m, 0.0)
    eye = torch.eye(C, device=q.device, dtype=torch.float32)
    T_mat = torch.linalg.solve_triangular(eye + A, eye.expand_as(A), upper=False, unitriangular=True)
    W = torch.matmul(T_mat, kf * bf.unsqueeze(-1) * torch.exp(cum_g.clamp(max=0.0)))
    U = torch.matmul(T_mat, vf * bf.unsqueeze(-1))
    qkt_d = torch.einsum("...id,...jd,...ijd->...ij", qf, kf, exp_diff).masked_fill(~causal_m, 0.0)
    S, outs = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32), []
    for c in range(nc):
        qc, kc, wc, uc, cg_c = qf[:, :, c], kf[:, :, c], W[:, :, c], U[:, :, c], cum_g[:, :, c]
        g_end, v_new = cg_c[:, :, -1:, :], uc - torch.matmul(wc, S)
        outs.append(torch.matmul(qc * torch.exp(cg_c.clamp(max=0.0)), S) + torch.matmul(qkt_d[:, :, c], v_new))
        k_dec = kc * torch.exp((g_end - cg_c).clamp(max=0.0))
        S = torch.exp(g_end.clamp(max=0.0)).transpose(-1, -2) * S + torch.matmul(k_dec.transpose(-1, -2), v_new)
    return torch.stack(outs, dim=2).reshape(B, H, T_pad, D)[:, :, :T, :].to(in_dtype)


def recurrent_channel_deltanet_ref(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, log_alpha: torch.Tensor, beta: torch.Tensor,
) -> torch.Tensor:
    """Exact token-by-token reference for per-channel diagonal-decay Gated DeltaNet."""
    B, H, T, D = q.shape
    S, outs = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32), []
    for t in range(T):
        qt, kt, vt = q[:, :, t].float(), k[:, :, t].float(), v[:, :, t].float()
        S = S * torch.exp(log_alpha[:, :, t].float()).unsqueeze(-1)
        err = vt - torch.einsum("bhkd,bhk->bhd", S, kt)
        S = S + torch.einsum("bhk,bhd->bhkd", kt, beta[:, :, t].float().unsqueeze(-1) * err)
        outs.append(torch.einsum("bhkd,bhk->bhd", S, qt))
    return torch.stack(outs, dim=2).to(q.dtype)


class ChannelGatedDeltaNetV9(nn.Module):
    """Genesis-v9 Gated DeltaNet upgraded with per-channel diagonal decay `diag(alpha_t) in R^{d_h}`."""

    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 32, scalar_mode: bool = False) -> None:
        super().__init__()
        self.d_model, self.nh, self.hd, self.chunk, self.scalar_mode = d_model, n_heads, d_model // n_heads, chunk_size, scalar_mode
        self.norm = nn.Parameter(torch.ones(d_model))
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d_model, d_model, bias=False) for _ in range(4))
        self.w_alpha = nn.Linear(d_model, n_heads if scalar_mode else d_model, bias=True)
        self.w_beta = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.w_alpha.bias, 2.0); nn.init.constant_(self.w_beta.bias, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        xn = rms_norm(x, self.norm)
        q = F.normalize(self.wq(xn).view(B, T, self.nh, self.hd).transpose(1, 2).float(), dim=-1, eps=1e-5).to(x.dtype)
        k = F.normalize(self.wk(xn).view(B, T, self.nh, self.hd).transpose(1, 2).float(), dim=-1, eps=1e-5).to(x.dtype)
        v = self.wv(xn).view(B, T, self.nh, self.hd).transpose(1, 2)
        la = (F.logsigmoid(self.w_alpha(xn).float()).transpose(1, 2).clamp(min=-6.0).unsqueeze(-1).expand(-1, -1, -1, self.hd)
              if self.scalar_mode else F.logsigmoid(self.w_alpha(xn).float()).view(B, T, self.nh, self.hd).transpose(1, 2).clamp(min=-6.0))
        beta = torch.sigmoid(self.w_beta(xn).float()).transpose(1, 2)
        return x + self.wo(chunkwise_channel_deltanet(q, k, v, la, beta, chunk_size=self.chunk).transpose(1, 2).reshape(B, T, D))


def _bench_fp4_kv() -> dict[str, Any]:
    torch.manual_seed(42)
    T, H, D = 512, 10, 64
    q, ks, vs = torch.randn(1, H, T, D) * 0.5, [torch.randn(1, H, T, D) * 0.5 for _ in range(2)], [torch.randn(1, H, T, D) * 0.5 for _ in range(2)]
    cache_v9 = FP4KVCacheV9(H, D, 32, 2)
    t0 = time.perf_counter()
    for lyr in range(2):
        cache_v9.append(lyr, ks[lyr], vs[lyr])
    o_fp4 = cache_v9.attend(0, q, is_causal=True)
    dt_fp4 = max(1e-6, time.perf_counter() - t0)
    o_f32 = F.scaled_dot_product_attention(q, ks[0], vs[0], is_causal=True)
    k_q8, kb = _q8_roundtrip(ks[0], 32); v_q8, vb = _q8_roundtrip(vs[0], 32)
    o_q8 = F.scaled_dot_product_attention(q, k_q8, v_q8, is_causal=True)
    f32_bpt, q8_bpt, fp4_bpt = float(sum(t.nbytes for t in (*ks, *vs)) / T), float(2 * (kb + vb) / T), cache_v9.bytes_per_token()
    cache_v8 = FP4KVCacheV9(6, 64, 32, 2)
    for lyr in range(2):
        cache_v8.append(lyr, torch.randn(1, 6, 128, 64), torch.randn(1, 6, 128, 64))
    return {
        "v9_f32_bytes_per_tok": round(f32_bpt, 2), "v9_q8_bytes_per_tok": round(q8_bpt, 2),
        "v9_fp4_bytes_per_tok": round(fp4_bpt, 2), "v8_fp4_bytes_per_tok": round(cache_v8.bytes_per_token(), 2),
        "compression_vs_q8_x": round(q8_bpt / fp4_bpt, 3), "compression_vs_f32_x": round(f32_bpt / fp4_bpt, 3),
        "attn_cosine_fp4_vs_f32": round(float(F.cosine_similarity(o_f32.reshape(-1), o_fp4.reshape(-1), dim=0).item()), 5),
        "attn_rmse_fp4_vs_f32": round(float((o_f32 - o_fp4).pow(2).mean().sqrt().item()), 5),
        "attn_cosine_q8_vs_f32": round(float(F.cosine_similarity(o_f32.reshape(-1), o_q8.reshape(-1), dim=0).item()), 5),
        "attn_rmse_q8_vs_f32": round(float((o_f32 - o_q8).pow(2).mean().sqrt().item()), 5),
        "cpu_fp4_quant_attend_tok_s": round(T / dt_fp4, 1),
    }


def _bench_ced_engram_deltanet() -> dict[str, Any]:
    torch.manual_seed(42)
    tiny = V9_PRESETS["v9_tiny"]
    model = GenesisV9Native(tiny).eval()
    ced = AsymmetricCEDWrapper(model, prefill_top_k=1, decode_top_k=2, prefill_recur=1, decode_recur=2)
    p_ids, d_ids = torch.randint(16, tiny.vocab, (1, 128)), torch.randint(16, tiny.vocab, (1, 16))
    with torch.no_grad():
        _ = model(p_ids, recur_steps=2); _ = ced.forward_prefill(p_ids)  # Warmup
        t0 = time.perf_counter(); lo_sym, _ = model(p_ids, recur_steps=2); dt_sym = max(1e-6, time.perf_counter() - t0)
        t1 = time.perf_counter(); lo_ced, _ = ced.forward_prefill(p_ids); dt_ced = max(1e-6, time.perf_counter() - t1)
        lo_dec, _ = ced.forward_decode(d_ids)
    assert torch.isfinite(lo_ced).all() and torch.isfinite(lo_dec).all() and lo_ced.shape == lo_sym.shape
    ced_res = {
        **AsymmetricCEDWrapper(model, 1, 3, 1, 2).account_sequence(V9_PRESETS["v9_1_5b"], 2048, 256),
        "tiny_measured_prefill_speedup_x": round(dt_sym / dt_ced, 3),
        "tiny_prefill_ms_before": round(dt_sym * 1000.0, 2), "tiny_prefill_ms_after": round(dt_ced * 1000.0, 2),
    }
    # Engram V4.1 vs V9 context-gated polysemy disambiguation A/B (disjoint noisy held-out context eval)
    torch.manual_seed(7)
    tok_s = torch.tensor([[20, 45, 77, 102, 130, 155, 200, 220]] * 2)
    base_h = torch.randn(1, 8, tiny.d_model)
    h_tr = torch.cat([base_h, -base_h], dim=0)
    h_te = torch.cat([base_h + 0.15 * torch.randn_like(base_h), -base_h + 0.15 * torch.randn_like(base_h)], dim=0)
    tgt = torch.zeros_like(h_tr); tgt[0, :, :32], tgt[1, :, 32:] = 1.0, 1.0
    eng_mse: dict[str, float] = {}
    for tag, mod in (("before_engram_v9_heldout_mse", DeepSeekEngramV9(tiny)), ("after_engram_v41_heldout_mse", EngramV41Module(tiny))):
        opt = torch.optim.AdamW(mod.parameters(), lr=8e-3)
        for _ in range(50):
            opt.zero_grad(); F.mse_loss(mod(tok_s, h_tr) - h_tr, tgt).backward(); opt.step()
        with torch.no_grad():
            eng_mse[tag] = round(float(F.mse_loss(mod(tok_s, h_te) - h_te, tgt).item()), 5)
    # Channel-wise DeltaNet exact parity + module gradcheck + selective-subspace associative recall A/B
    torch.manual_seed(42)
    q, k, v = F.normalize(torch.randn(2, 4, 77, 32), dim=-1), F.normalize(torch.randn(2, 4, 77, 32), dim=-1), torch.randn(2, 4, 77, 32)
    la, beta = -torch.rand(2, 4, 77, 32) * 4.5 - 0.01, torch.sigmoid(torch.randn(2, 4, 77))
    o_ref = recurrent_channel_deltanet_ref(q, k, v, la, beta)
    max_16 = float((o_ref - chunkwise_channel_deltanet(q, k, v, la, beta, 16)).abs().max().item())
    max_32 = float((o_ref - chunkwise_channel_deltanet(q, k, v, la, beta, 32)).abs().max().item())
    assert max_16 < 1e-4 and max_32 < 1e-4, f"Parity failed: {max_16}, {max_32}"
    x_smk = torch.randn(2, 24, 64, requires_grad=True)
    ChannelGatedDeltaNetV9(64, 4, 16)(x_smk).sum().backward()
    assert x_smk.grad is not None and torch.isfinite(x_smk.grad).all()

    def _recall_batch(n: int, seed: int):
        g, d, L = torch.Generator().manual_seed(seed), 32, 16
        ks_b, vs_b, qs_b = (torch.zeros(n, 1, L, d) for _ in range(3))
        kl, vl = F.normalize(torch.randn(n, d // 2, generator=g), dim=-1), F.normalize(torch.randn(n, d, generator=g), dim=-1)
        ko = F.normalize(torch.randn(n, 6, d // 2, generator=g), dim=-1)
        ks_b[:, 0, 0, : d // 2], vs_b[:, 0, 0] = kl, vl
        ks_b[:, 0, 1:7, d // 2 :], vs_b[:, 0, 1:7] = ko, F.normalize(torch.randn(n, 6, d, generator=g), dim=-1)
        sig = torch.zeros(n, L, 2); sig[:, 0:7, 0] = 1.0; sig[:, 7:14, 1] = 1.0
        qs_b[:, 0, 14, : d // 2], qs_b[:, 0, 15, d // 2 :] = kl, ko[:, 0]
        return qs_b, ks_b, vs_b, sig, torch.stack([vl, torch.zeros_like(vl)], dim=1)

    q_tr, k_tr, v_tr, s_tr, y_tr = _recall_batch(128, 101)
    q_te, k_te, v_te, s_te, y_te = _recall_batch(64, 202)
    recall_res: dict[str, Any] = {}
    for tag, is_sc in (("before_scalar_alpha", True), ("after_channel_diag_alpha", False)):
        torch.manual_seed(11)
        ga, gb = nn.Linear(2, 1 if is_sc else 32), nn.Linear(2, 1)
        nn.init.constant_(ga.bias, 1.5); nn.init.constant_(gb.bias, 1.5)
        opt = torch.optim.Adam((*ga.parameters(), *gb.parameters()), lr=0.15)
        for _ in range(120):
            opt.zero_grad()
            lp = F.logsigmoid(ga(s_tr)).unsqueeze(1).clamp(min=-6.0)
            lp = lp.expand(-1, -1, -1, 32) if is_sc else lp
            F.mse_loss(chunkwise_channel_deltanet(q_tr, k_tr, v_tr, lp, torch.sigmoid(gb(s_tr)).squeeze(-1).unsqueeze(1), 16)[:, 0, -2:, :], y_tr).backward()
            opt.step()
        with torch.no_grad():
            le = F.logsigmoid(ga(s_te)).unsqueeze(1).clamp(min=-6.0)
            oe = chunkwise_channel_deltanet(q_te, k_te, v_te, le.expand(-1, -1, -1, 32) if is_sc else le, torch.sigmoid(gb(s_te)).squeeze(-1).unsqueeze(1), 16)[:, 0, -2:, :]
            recall_res[tag] = {
                "heldout_total_mse": round(float(F.mse_loss(oe, y_te).item()), 6),
                "persistent_retain_cosine": round(float(F.cosine_similarity(oe[:, 0], y_te[:, 0], dim=-1).mean().item()), 5),
                "stale_scratchpad_residual_norm": round(float(oe[:, 1].norm(dim=-1).mean().item()), 5),
            }
    return {
        "asymmetric_ced": ced_res, "engram_v41_ab": eng_mse,
        "channel_deltanet": {
            "parity_max_abs_c16": round(max_16, 8), "parity_max_abs_c32": round(max_32, 8),
            "parity_pass_lt_1e4": bool(max_16 < 1e-4 and max_32 < 1e-4), "associative_recall_ab": recall_res,
        },
    }


def run_all_verifications(out_json: Path = DEFAULT_AB_JSON) -> dict[str, Any]:
    if str(out_json).lower().startswith("c:"):
        raise RuntimeError(f"Refusing write to C:\\: {out_json}")
    res = {"fp4_kv_cache": _bench_fp4_kv(), **_bench_ced_engram_deltanet()}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(out_json.read_text(encoding="utf-8")) if out_json.is_file() else {}
    merged["v9_v41_upgrades"] = res
    out_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


if __name__ == "__main__":
    os.environ.setdefault("TEMP", r"E:\learn\project_genesis\.tmp_e")
    os.environ.setdefault("TMP", r"E:\learn\project_genesis\.tmp_e")
    print(json.dumps(run_all_verifications(), indent=2))

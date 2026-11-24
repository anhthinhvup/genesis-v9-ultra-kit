"""
Genesis-v9 Native 10-Lever Data-Efficiency Architecture (`colab_gpu_trainer/genesis_v9_native.py`).
~1.53B total / ~77M-101M active (v9_1_5b) ready for 100B-300B tokens on TPU TRC / Modal B200:
  - U-shaped Sparsity: ~65.8% BitNet 1.58-bit MoE : ~19.8% Full DeepSeek Engram : ~12.4% Split-Key PEER.
  - Selective Precision (Kumar ICLR 2025): BF16/INT8 nn.Linear for DeltaNet QKVO & Shared-KV Attn (W=2048);
    BitNet b1.58 ternary STE strictly ONLY in large MoE FFN experts.
  - Full DeepSeek Engram (arXiv:2601.07372) at Layer 2 + Query-Normed Split-Key PEER + Ouro + MTP (k=3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import unicodedata
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from genesis_v8_native import chunkwise_gated_deltanet, rms_norm
except ImportError:
    from .genesis_v8_native import chunkwise_gated_deltanet, rms_norm

TOK_PAD, TOK_BOS, TOK_EOS, TOK_USER, TOK_BOT, TOK_CTX = 0, 1, 2, 3, 4, 5
TOK_THINK, TOK_END_THINK, TOK_VERIFY = 11, 12, 13
V9_SPECIAL_TOKENS: tuple[tuple[str, int], ...] = (
    ("<PAD>", 0), ("<BOS>", 1), ("<EOS>", 2), ("<USER>", 3), ("<BOT>", 4), ("<CTX>", 5),
    ("<CALL_CAS>", 6), ("<CALL_SAT>", 7), ("<CALL_CODE>", 8), ("<CALL_RAG>", 9), ("<TOOL_OUT>", 10),
    ("<THINK>", TOK_THINK), ("</THINK>", TOK_END_THINK), ("<VERIFY>", TOK_VERIFY),
)
_HASH_PRIMES: tuple[tuple[int, int, int], ...] = (
    (1000003, 9176, 17), (15485863, 32452843, 53), (49979687, 67867967, 97), (86028121, 104395301, 131),
    (2654435761, 1315423911, 2246822519), (2166136261, 16777619, 3266489917),
    (374761393, 668265263, 1274126177), (2860486313, 3267000013, 1500450271),
)


@dataclass(frozen=True)
class V9Config:
    vocab: int = 16384; d_model: int = 640
    n_prelude_dn: int = 1; n_core_dn: int = 2; n_coda_dn: int = 3
    n_experts: int = 64; top_k: int = 3; d_ff: int = 2048
    attn_heads: int = 10; attn_window: int = 2048; d_attn_ff: int = 1280
    delta_heads: int = 10; delta_chunk: int = 64
    engram_heads: int = 8; engram_buckets: int = 262144; engram_dim: int = 144; engram_conv_k: int = 4
    peer_heads: int = 8; peer_sub_k: int = 272; peer_k: int = 8
    recur_steps: int = 2; max_recur_steps: int = 8; mtp_k: int = 3
    tok_think: int = TOK_THINK; tok_end_think: int = TOK_END_THINK; tok_verify: int = TOK_VERIFY

    @property
    def n_dn_layers(self) -> int:
        return self.n_prelude_dn + self.n_core_dn + self.n_coda_dn

    def param_breakdown(self) -> dict[str, int | float]:
        d, n_dn = self.d_model, self.n_dn_layers
        emb = self.vocab * d
        dn_per = 4 * d * d + 2 * self.delta_heads * d + 2 * self.delta_heads + 2 * d
        moe_per = d * self.n_experts + 2 * self.n_experts * d * self.d_ff
        attn_per = 4 * d * d + 2 * d * self.d_attn_ff + 2 * d
        deltanet, moe, attn = n_dn * dn_per, n_dn * moe_per, 2 * attn_per
        eng_d = self.engram_heads * self.engram_dim
        engram_proj = 2 * eng_d * d + (3 + self.engram_conv_k) * d
        engram = self.engram_heads * self.engram_buckets * self.engram_dim + engram_proj
        sub_d = (d // self.peer_heads) // 2
        peer_per = 2 * d + d * d + 2 * self.peer_heads * self.peer_sub_k * sub_d + 2 * (self.peer_sub_k ** 2) * d
        peer = 2 * peer_per
        ouro_ctrl = (self.max_recur_steps + 3) * d + 1
        ouro_mtp = ouro_ctrl + max(0, self.mtp_k - 1) * (4 * d * d + 3 * d)
        dense_total, sparse_total = emb + deltanet + attn + ouro_mtp, moe + engram + peer
        total = dense_total + sparse_total
        moe_act = d * self.n_experts + 2 * self.top_k * d * self.d_ff
        peer_act = 2 * d + d * d + 2 * self.peer_heads * self.peer_sub_k * sub_d + 2 * self.peer_heads * self.peer_k * d
        active_r1 = emb + deltanet + n_dn * moe_act + attn + engram_proj + eng_d + 2 * peer_act + ouro_ctrl
        attn_shared_step = 2 * d * d + 2 * d * self.d_attn_ff + 2 * d  # Shared-KV skips wk/wv on r>=1
        active_r2 = active_r1 + self.n_core_dn * (dn_per + moe_act) + attn_shared_step + 2 * peer_act
        sp = max(1, sparse_total)
        return {
            "emb": emb, "deltanet": deltanet, "attn": attn, "ouro_mtp": ouro_mtp,
            "dense_total": dense_total, "moe": moe, "engram": engram, "peer": peer,
            "sparse_total": sparse_total, "total": total, "active_r1": active_r1, "active_r2": active_r2,
            "moe_pct": round(100.0 * moe / total, 2), "engram_pct": round(100.0 * engram / total, 2),
            "peer_pct": round(100.0 * peer / total, 2), "dense_pct": round(100.0 * dense_total / total, 2),
            "sparse_moe_pct": round(100.0 * moe / sp, 2), "sparse_engram_pct": round(100.0 * engram / sp, 2),
            "sparse_peer_pct": round(100.0 * peer / sp, 2),
        }

    def num_params(self) -> int:
        return int(self.param_breakdown()["total"])


V9_PRESETS: dict[str, V9Config] = {
    "v9_tiny": V9Config(
        vocab=512, d_model=64, n_prelude_dn=1, n_core_dn=2, n_coda_dn=1,
        n_experts=8, top_k=2, d_ff=96, attn_heads=4, attn_window=64, d_attn_ff=128,
        delta_heads=4, delta_chunk=16, engram_heads=4, engram_buckets=512, engram_dim=16,
        engram_conv_k=4, peer_heads=4, peer_sub_k=16, peer_k=2,
        recur_steps=2, max_recur_steps=4, mtp_k=3,
    ),
    "v9_1_5b": V9Config(),
}


@dataclass
class V9Output:
    logits: torch.Tensor; aux_loss: torch.Tensor
    depth_logits: list[torch.Tensor]; exit_probs: torch.Tensor
    mtp_logits: list[torch.Tensor]; hidden: torch.Tensor


def build_nfkc_canon_lut(vocab_size: int, id_to_text: Sequence[str] | None = None) -> torch.Tensor:
    """DeepSeek Engram NFKC + lowercase canonical projection P: V -> V'."""
    if id_to_text is None:
        lut = list(range(vocab_size))
        for b in range(65, 91):
            if 16 + b + 32 < vocab_size:
                lut[16 + b] = 16 + b + 32
        return torch.tensor(lut, dtype=torch.long)
    seen: dict[str, int] = {}
    lut = [tid if (tid < 16 or tid >= len(id_to_text)) else seen.setdefault(
        unicodedata.normalize("NFKC", id_to_text[tid]).lower().strip() or f"\x00{tid}", tid
    ) for tid in range(vocab_size)]
    return torch.tensor(lut, dtype=torch.long)


class DeepSeekEngramV9(nn.Module):
    """Full DeepSeek Engram (arXiv:2601.07372): NFKC canon + N-gram hash + context gate + zero-init Conv1D."""

    def __init__(self, cfg: V9Config) -> None:
        super().__init__()
        self.vocab, self.d = cfg.vocab, cfg.d_model
        self.n_heads, self.buckets, self.conv_k = cfg.engram_heads, cfg.engram_buckets, cfg.engram_conv_k
        self.register_buffer("canon_lut", build_nfkc_canon_lut(cfg.vocab), persistent=True)
        self.tables = nn.ModuleList([nn.Embedding(cfg.engram_buckets, cfg.engram_dim) for _ in range(cfg.engram_heads)])
        for tbl in self.tables: nn.init.normal_(tbl.weight, 0.0, 0.01)
        tot_d = cfg.engram_heads * cfg.engram_dim
        self.wk, self.wv = nn.Linear(tot_d, cfg.d_model, bias=False), nn.Linear(tot_d, cfg.d_model, bias=False)
        self.norm_h, self.norm_k, self.norm_v = (nn.Parameter(torch.ones(cfg.d_model)) for _ in range(3))
        self.conv = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=cfg.engram_conv_k, groups=cfg.d_model, bias=False)
        nn.init.zeros_(self.conv.weight)

    def set_canonical_lut(self, lut: torch.Tensor) -> None:
        self.canon_lut.copy_(lut.to(device=self.canon_lut.device, dtype=torch.long))

    def forward(self, tok_ids: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        cid = self.canon_lut[tok_ids.clamp(0, self.vocab - 1)]
        p1, p2, half = F.pad(cid[:, :-1], (1, 0)), F.pad(cid[:, :-2], (2, 0)), max(1, self.n_heads // 2)
        embs = [tbl(((((cid * m0) ^ (p1 * m1 + m2)) if i < half else ((cid * m0) ^ (p1 * m1) ^ (p2 * m2))) & 0x7FFFFFFF) % self.buckets)
                for i, tbl in enumerate(self.tables) for (m0, m1, m2) in (_HASH_PRIMES[i % len(_HASH_PRIMES)],)]
        e_raw = torch.cat(embs, dim=-1).to(h.dtype)
        k, v = self.wk(e_raw), self.wv(e_raw)
        alpha = torch.sigmoid(
            (rms_norm(h, self.norm_h).float() * rms_norm(k, self.norm_k).float()).sum(dim=-1, keepdim=True) * (self.d ** -0.5)
        ).to(h.dtype)
        vg = alpha * v
        return h + vg + F.silu(self.conv(F.pad(rms_norm(vg, self.norm_v).transpose(1, 2), (self.conv_k - 1, 0))).transpose(1, 2))


class BitNetMoEV9(nn.Module):
    """Selective-Precision BitNet b1.58 Ternary STE MoE (ONLY BitNet module in Genesis-v9)."""

    def __init__(self, d: int, n_exp: int, top_k: int, d_ff: int) -> None:
        super().__init__()
        self.d, self.n_exp, self.top_k, self.d_ff = d, n_exp, top_k, d_ff
        self.router = nn.Linear(d, n_exp, bias=False)
        self.bit_w_up = nn.Parameter(torch.empty(n_exp, d_ff, d).normal_(0.0, 0.02))
        self.bit_w_down = nn.Parameter(torch.empty(n_exp, d, d_ff).normal_(0.0, 0.02))

    @staticmethod
    def _ste_158(w: torch.Tensor) -> torch.Tensor:
        s = w.abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        return w + (torch.clamp(torch.round(w / s), -1.0, 1.0) * s - w).detach()

    def forward(self, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = xn.shape
        N, E, K = B * T, self.n_exp, self.top_k
        x_flat = xn.reshape(N, D)
        probs = F.softmax(self.router(x_flat).float(), dim=-1)
        top_w, top_idx = torch.topk(probs, K, dim=-1)
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)).to(xn.dtype)
        aux = float(E) * ((F.one_hot(top_idx, E).float().sum(dim=1).mean(dim=0) / float(K)) * probs.mean(dim=0)).sum()
        flat_e, flat_w = top_idx.reshape(-1), top_w.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        se, sw = flat_e[order], flat_w[order]
        st = torch.arange(N, device=xn.device).repeat_interleave(K)[order]
        cap = max(1, (N * K * 3) // (2 * E) + 32)
        starts = F.pad(torch.bincount(se, minlength=E).cumsum(0)[:-1], (1, 0))
        pos = torch.arange(N * K, device=xn.device) - starts[se]
        valid = pos < cap
        ve, vp, vt, vw = se[valid], pos[valid], st[valid], sw[valid]
        b_in = torch.zeros((E, cap, D), device=xn.device, dtype=xn.dtype)
        b_in[ve, vp] = x_flat[vt]
        wu, wd = self._ste_158(self.bit_w_up).to(xn.dtype), self._ste_158(self.bit_w_down).to(xn.dtype)
        b_out = torch.bmm(F.relu(torch.bmm(b_in, wu.transpose(1, 2))).clamp(max=6.0).square(), wd.transpose(1, 2))
        out = torch.zeros_like(x_flat)
        out.index_add_(0, vt, b_out[ve, vp] * vw.unsqueeze(-1))
        return out.reshape(B, T, D), aux


class GatedDeltaNetMoEBlockV9(nn.Module):
    """BF16/INT8-ready Gated DeltaNet QKVO + BitNet b1.58 MoE block."""

    def __init__(self, cfg: V9Config) -> None:
        super().__init__()
        d, nh = cfg.d_model, cfg.delta_heads
        self.nh, self.hd, self.chunk = nh, d // nh, cfg.delta_chunk
        self.norm1, self.norm2 = nn.Parameter(torch.ones(d)), nn.Parameter(torch.ones(d))
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.w_ab = nn.Linear(d, 2 * nh, bias=True)
        nn.init.constant_(self.w_ab.bias[:nh], 2.0); nn.init.constant_(self.w_ab.bias[nh:], -1.0)
        self.moe = BitNetMoEV9(d, cfg.n_experts, cfg.top_k, cfg.d_ff)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        xn = rms_norm(x, self.norm1)
        q = F.normalize(self.wq(xn).view(B, T, self.nh, self.hd).transpose(1, 2).float(), dim=-1, eps=1e-5).to(x.dtype)
        k = F.normalize(self.wk(xn).view(B, T, self.nh, self.hd).transpose(1, 2).float(), dim=-1, eps=1e-5).to(x.dtype)
        v = self.wv(xn).view(B, T, self.nh, self.hd).transpose(1, 2)
        ab = self.w_ab(xn).float()
        log_a = F.logsigmoid(ab[:, :, : self.nh]).transpose(1, 2).clamp(min=-6.0)
        beta = torch.sigmoid(ab[:, :, self.nh :]).transpose(1, 2)
        o = chunkwise_gated_deltanet(q, k, v, log_a, beta, chunk_size=self.chunk)
        x = x + self.wo(o.transpose(1, 2).reshape(B, T, D))
        m, aux = self.moe(rms_norm(x, self.norm2))
        return x + m, aux


class HuginnSharedKVAttnBlockV9(nn.Module):
    """BF16/INT8-ready Sliding-Window RoPE Shared-KV Attention (W=2048) + standard FFN."""

    def __init__(self, cfg: V9Config) -> None:
        super().__init__()
        d, nh = cfg.d_model, cfg.attn_heads
        self.nh, self.hd, self.win = nh, d // nh, cfg.attn_window
        self.norm1, self.norm2 = nn.Parameter(torch.ones(d)), nn.Parameter(torch.ones(d))
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.ff_up, self.ff_dn = nn.Linear(d, cfg.d_attn_ff, bias=False), nn.Linear(cfg.d_attn_ff, d, bias=False)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        self.register_buffer("inv_freq", inv, persistent=False)

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        ang = torch.arange(x.shape[2], device=x.device, dtype=torch.float32)[:, None] * self.inv_freq[None, :]
        c, s = ang.cos().to(x.dtype)[None, None], ang.sin().to(x.dtype)[None, None]
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(-2)

    def forward(
        self, x: torch.Tensor, kv_shared: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, D = x.shape
        xn = rms_norm(x, self.norm1)
        q = self._rope(self.wq(xn).view(B, T, self.nh, self.hd).transpose(1, 2))
        if kv_shared is None:
            k = self._rope(self.wk(xn).view(B, T, self.nh, self.hd).transpose(1, 2))
            kv_shared = (k, self.wv(xn).view(B, T, self.nh, self.hd).transpose(1, 2))
        k, v = kv_shared
        if T <= self.win:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            idx = torch.arange(T, device=x.device)
            dist = idx[:, None] - idx[None, :]
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=(dist >= 0) & (dist < self.win))
        x = x + self.wo(o.transpose(1, 2).reshape(B, T, D))
        return x + self.ff_dn(F.relu(self.ff_up(rms_norm(x, self.norm2))).clamp(max=6.0).square()), kv_shared


class PeerLayerV9(nn.Module):
    """Query-Normed Split-Key PEER: sub_d=(d//nh)//2 splitting [q1; q2] + load-balance aux loss."""

    def __init__(self, cfg: V9Config) -> None:
        super().__init__()
        assert cfg.d_model % (2 * cfg.peer_heads) == 0, "d_model must be divisible by 2*peer_heads"
        self.d, self.nh, self.sub_k, self.top_k = cfg.d_model, cfg.peer_heads, cfg.peer_sub_k, cfg.peer_k
        self.sub_d = (self.d // self.nh) // 2
        self.norm, self.wq = nn.Parameter(torch.ones(self.d)), nn.Linear(self.d, self.d, bias=False)
        self.q_norm_a, self.q_norm_b = (nn.Parameter(torch.ones(self.nh, self.sub_d)) for _ in range(2))
        self.keys_a = nn.Parameter(torch.empty(self.nh, self.sub_k, self.sub_d).normal_(0.0, 0.05))
        self.keys_b = nn.Parameter(torch.empty(self.nh, self.sub_k, self.sub_d).normal_(0.0, 0.05))
        self.u_exp = nn.Parameter(torch.empty(self.sub_k * self.sub_k, self.d).normal_(0.0, 0.015))
        self.v_exp = nn.Parameter(torch.empty(self.sub_k * self.sub_k, self.d).normal_(0.0, 0.015))

    @staticmethod
    def _q14(w: torch.Tensor) -> torch.Tensor:
        return w + (torch.clamp(torch.round(w.float() * 16384.0), -32767.0, 32767.0) / 16384.0 - w.float()).to(w.dtype).detach()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        N, scale = B * T, self.sub_d ** -0.5
        xn = rms_norm(x, self.norm)
        q = self.wq(xn).view(N, self.nh, 2, self.sub_d)
        q1, q2 = rms_norm(q[:, :, 0], self.q_norm_a), rms_norm(q[:, :, 1], self.q_norm_b)
        sa, sb = torch.einsum("nhd,hkd->nhk", q1, self.keys_a) * scale, torch.einsum("nhd,hkd->nhk", q2, self.keys_b) * scale
        va, ia = torch.topk(sa, self.top_k, dim=-1)
        vb, ib = torch.topk(sb, self.top_k, dim=-1)
        pa, pb = F.softmax(sa.float(), dim=-1).mean(dim=0), F.softmax(sb.float(), dim=-1).mean(dim=0)
        fa = F.one_hot(ia, self.sub_k).float().sum(dim=2).mean(dim=0) / float(self.top_k)
        fb = F.one_hot(ib, self.sub_k).float().sum(dim=2).mean(dim=0) / float(self.top_k)
        peer_aux = 0.5 * float(self.sub_k) * ((fa * pa).sum(dim=-1).mean() + (fb * pb).sum(dim=-1).mean())
        comb_v = (va.unsqueeze(-1) + vb.unsqueeze(-2)).reshape(N, self.nh, self.top_k * self.top_k)
        comb_i = (ia.unsqueeze(-1) * self.sub_k + ib.unsqueeze(-2)).reshape(N, self.nh, self.top_k * self.top_k)
        tv, ti = torch.topk(comb_v, self.top_k, dim=-1)
        w, eids = F.softmax(tv.float(), dim=-1).to(x.dtype), torch.gather(comb_i, -1, ti)
        ue, ve = self._q14(self.u_exp[eids]), self._q14(self.v_exp[eids])
        act = F.relu((xn.reshape(N, 1, 1, D) * ue).sum(dim=-1)).clamp(max=6.0).square()
        return x + ((w * act).unsqueeze(-1) * ve).sum(dim=(1, 2)).view(B, T, D), peer_aux


class MTPHeadV9(nn.Module):
    """Lightweight DeepSeek-V3 Multi-Token Prediction block sharing emb & lm_head."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.norm_h, self.norm_e, self.norm_m = (nn.Parameter(torch.ones(d)) for _ in range(3))
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.ff_up, self.ff_dn = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)

    def forward(self, h_prev: torch.Tensor, next_emb: torch.Tensor) -> torch.Tensor:
        z = self.proj(torch.cat([rms_norm(h_prev, self.norm_h), rms_norm(next_emb, self.norm_e)], dim=-1))
        return z + self.ff_dn(F.relu(self.ff_up(rms_norm(z, self.norm_m))).clamp(max=6.0).square())


class GenesisV9Native(nn.Module):
    """Genesis-v9: Selective-Precision DeltaNet+Huginn + Layer-2 Engram + Split-Key PEER + Ouro + MTP."""

    def __init__(self, cfg: V9Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        nn.init.normal_(self.emb.weight, 0.0, 0.02)
        self.prelude_dn = nn.ModuleList([GatedDeltaNetMoEBlockV9(cfg) for _ in range(cfg.n_prelude_dn)])
        self.prelude_attn = HuginnSharedKVAttnBlockV9(cfg)
        self.engram = DeepSeekEngramV9(cfg)  # Layer-2 contextualized insertion
        self.core_dn = nn.ModuleList([GatedDeltaNetMoEBlockV9(cfg) for _ in range(cfg.n_core_dn)])
        self.peer0, self.core_attn, self.peer1 = PeerLayerV9(cfg), HuginnSharedKVAttnBlockV9(cfg), PeerLayerV9(cfg)
        self.step_emb = nn.Embedding(cfg.max_recur_steps, cfg.d_model)
        nn.init.normal_(self.step_emb.weight, 0.0, 0.01)
        self.recur_inj = nn.Parameter(torch.full((cfg.d_model,), 0.25))
        self.exit_gate = nn.Linear(cfg.d_model, 1, bias=True)
        nn.init.zeros_(self.exit_gate.weight); nn.init.constant_(self.exit_gate.bias, -1.0)
        self.coda_dn = nn.ModuleList([GatedDeltaNetMoEBlockV9(cfg) for _ in range(cfg.n_coda_dn)])
        self.norm_f = nn.Parameter(torch.ones(cfg.d_model))
        self.mtp_heads = nn.ModuleList([MTPHeadV9(cfg.d_model) for _ in range(max(0, cfg.mtp_k - 1))])

    def _lm_head(self, h: torch.Tensor) -> torch.Tensor:
        logits = F.linear(rms_norm(h, self.norm_f), self.emb.weight.to(h.dtype))
        return 30.0 * torch.tanh(logits / 30.0)

    def _run_coda(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aux = h.new_zeros((), dtype=torch.float32)
        for blk in self.coda_dn:
            h, a = blk(h); aux = aux + a
        return h, aux

    def forward(
        self, tok_ids: torch.Tensor, recur_steps: int | None = None,
        targets: torch.Tensor | None = None, return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | V9Output:
        r_steps = max(1, min(self.cfg.max_recur_steps, int(recur_steps or self.cfg.recur_steps)))
        x, aux_tot = self.emb(tok_ids), torch.zeros((), device=tok_ids.device, dtype=torch.float32)
        for blk in self.prelude_dn:
            x, a = blk(x); aux_tot = aux_tot + a
        x, _ = self.prelude_attn(x, kv_shared=None)
        x = self.engram(tok_ids, x)
        s0, core_kv = x, None
        surv = x.new_ones((x.shape[0], x.shape[1]), dtype=torch.float32)
        exit_list, step_states = [], []
        for r in range(r_steps):
            x = x + (torch.tanh(self.recur_inj).to(x.dtype) * s0 if r > 0 else 0) + self.step_emb.weight[r].to(x.dtype)
            x, a = self.core_dn[0](x); aux_tot = aux_tot + a
            x, pa = self.peer0(x); aux_tot = aux_tot + pa
            x, core_kv = self.core_attn(x, kv_shared=core_kv)
            for blk in self.core_dn[1:]:
                x, a = blk(x); aux_tot = aux_tot + a
            x, pa = self.peer1(x); aux_tot = aux_tot + pa
            if r < r_steps - 1:
                lam = torch.sigmoid(self.exit_gate(x).float()).squeeze(-1)
                exit_list.append(surv * lam); surv = surv * (1.0 - lam)
            else:
                exit_list.append(surv)
            if return_details:
                step_states.append(x)
        if not return_details:
            h_fin, ca = self._run_coda(x)
            return self._lm_head(h_fin), aux_tot + ca
        depth_logits, h_fin = [], x
        for r, xs in enumerate(step_states):
            hc, ca = self._run_coda(xs)
            if r == r_steps - 1:
                h_fin, aux_tot = hc, aux_tot + ca
            depth_logits.append(self._lm_head(hc))
        mtp_logits: list[torch.Tensor] = []
        if targets is not None and self.mtp_heads:
            h_mtp, T = h_fin, tok_ids.shape[1]
            for m, head in enumerate(self.mtp_heads):
                if (L := T - (m + 1)) <= 0:
                    break
                h_mtp = head(h_mtp[:, :L], self.emb(targets[:, m : m + L]))
                mtp_logits.append(self._lm_head(h_mtp))
        return V9Output(depth_logits[-1], aux_tot, depth_logits, torch.stack(exit_list, dim=-1), mtp_logits, h_fin)

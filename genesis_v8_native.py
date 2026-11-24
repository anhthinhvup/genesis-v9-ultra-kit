"""
Genesis-v8 Native BICA (`colab_gpu_trainer/genesis_v8_native.py`, <= 400 LOC):
Chunkwise WY Multi-Head Gated DeltaNet (ICLR 2025) + Huginn Shared-KV W=1024
+ DeepSeek Engram O(1) + BitNet b1.58 bmm MoE + DeepMind 2x262K PEER.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class V8Config:
    vocab: int = 4096; d_model: int = 384; n_layers: int = 8
    n_experts: int = 16; top_k: int = 2; d_ff: int = 512
    peer_sub_k: int = 512; peer_heads: int = 6; peer_k: int = 4
    attn_heads: int = 6; attn_window: int = 1024
    delta_heads: int = 6; delta_chunk: int = 64
    engram_buckets: int = 65536; engram_dim: int = 64; engram_heads: int = 2
    recur_steps: int = 2

    def num_params(self) -> int:
        d = self.d_model
        emb = self.vocab * d
        engram = self.engram_heads * self.engram_buckets * self.engram_dim + self.engram_heads * self.engram_dim * d + d
        # Exact GenesisV8Native topology: 5 DeltaNet-MoE + 2 HuginnSharedKVAttn + 2 PEER + recur_inj/norm_f
        dn_attn = 4 * d * d + 2 * self.delta_heads * (d + 1) + 2 * d
        moe = self.n_experts * (d + 2 * d * self.d_ff)
        sw_attn = 4 * d * d + 2 * d + 2 * d * self.d_ff
        peer = 2 * (d + d * d + self.peer_heads * 2 * self.peer_sub_k * (d // self.peer_heads) + 2 * (self.peer_sub_k ** 2) * d)
        return emb + engram + 5 * (dn_attn + moe) + 2 * sw_attn + peer + 2 * d

    def actual_active_params(self, recur_steps: int | None = None, include_lm_head: bool = True) -> int:
        """Exact per-token active parameter count for GenesisV8Native at recurrent depth R."""
        r = max(1, int(self.recur_steps if recur_steps is None else recur_steps))
        d = self.d_model
        engram_act = self.engram_heads * self.engram_dim + self.engram_heads * self.engram_dim * d + d
        dn_act = 4 * d * d + 2 * self.delta_heads * (d + 1) + 2 * d + self.n_experts * d + self.top_k * (2 * d * self.d_ff)
        sw_act = 4 * d * d + 2 * d + 2 * d * self.d_ff
        peer_act = d + d * d + self.peer_heads * 2 * self.peer_sub_k * (d // self.peer_heads) + self.peer_heads * self.peer_k * (2 * d)
        body = engram_act + 3 * dn_act + sw_act + r * (2 * dn_act + sw_act + 2 * peer_act) + d
        return body + (self.vocab * d if include_lm_head else 0)


V8_PRESETS: dict[str, V8Config] = {
    "v8_465m": V8Config(d_model=384, n_layers=8, attn_window=1024, peer_sub_k=512),
    "v8_580m": V8Config(d_model=512, n_layers=8, attn_heads=8, delta_heads=8, d_ff=640, attn_window=1024, peer_sub_k=512, peer_heads=8),
}


def rms_norm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype) * g


class BitLinearV8(nn.Module):
    def __init__(self, in_f: int, out_f: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f).normal_(0.0, 0.02))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        s = w.abs().mean().clamp(min=1e-6)
        q = torch.clamp(torch.round(w / s), -1.0, 1.0)
        w_eff = w + (q * s - w).detach()
        return F.linear(x, w_eff.to(x.dtype))


class EngramMemoryV8(nn.Module):
    """DeepSeek Engram (2026) O(1) deterministic multi-head N-gram hash memory."""

    def __init__(self, cfg: V8Config) -> None:
        super().__init__()
        self.buckets = cfg.engram_buckets
        self.n_heads = cfg.engram_heads
        self.tables = nn.ModuleList([
            nn.Embedding(cfg.engram_buckets, cfg.engram_dim) for _ in range(cfg.engram_heads)
        ])
        for t in self.tables:
            nn.init.normal_(t.weight, 0.0, 0.01)
        self.proj = nn.Linear(cfg.engram_heads * cfg.engram_dim, cfg.d_model, bias=False)
        self.gate = nn.Parameter(torch.full((cfg.d_model,), 0.1))

    def forward(self, tok_ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        prev1 = F.pad(tok_ids[:, :-1], (1, 0), value=0)
        prev2 = F.pad(tok_ids[:, :-2], (2, 0), value=0)
        h0 = ((tok_ids * 1000003) ^ (prev1 * 9176 + 17)) & (self.buckets - 1)
        h1 = ((tok_ids * 2654435761) ^ (prev1 * 1315423911) ^ (prev2 * 2246822519)) & (self.buckets - 1)
        hashes = (h0, h1)
        feats = torch.cat([tbl(hashes[i]) for i, tbl in enumerate(self.tables)], dim=-1)
        return x + torch.tanh(self.gate).to(x.dtype) * self.proj(feats.to(x.dtype))


def chunkwise_gated_deltanet(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Exact Chunkwise WY Parallel Gated DeltaNet (Yang et al., ICLR 2025).
    q, k, v: (B, H, T, D); log_alpha, beta: (B, H, T).
    Recurrence: S_t = alpha_t * S_{t-1} * (I - beta_t k_t k_t^T) + beta_t v_t k_t^T, o_t = S_t q_t.
    """
    B, H, T, D = q.shape
    pad = (chunk_size - (T % chunk_size)) % chunk_size
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        log_alpha = F.pad(log_alpha, (0, pad))
        beta = F.pad(beta, (0, pad))
    T_pad = q.shape[2]
    nc = T_pad // chunk_size
    C = chunk_size

    # Work in fp32 for WY stability, then cast output back to input dtype
    in_dtype = q.dtype
    qf = q.float().view(B, H, nc, C, D)
    kf = k.float().view(B, H, nc, C, D)
    vf = v.float().view(B, H, nc, C, D)
    gf = log_alpha.float().view(B, H, nc, C)
    bf = beta.float().view(B, H, nc, C)

    cum_g = gf.cumsum(dim=-1)  # (B, H, nc, C)
    decay_diff = (cum_g.unsqueeze(-1) - cum_g.unsqueeze(-2)).clamp(max=0.0)  # i >= j
    strict_mask = torch.tril(torch.ones((C, C), device=q.device, dtype=torch.bool), diagonal=-1)
    causal_mask = torch.tril(torch.ones((C, C), device=q.device, dtype=torch.bool), diagonal=0)

    kkt = torch.matmul(kf, kf.transpose(-1, -2))  # (B, H, nc, C, C)
    A = (bf.unsqueeze(-1) * torch.exp(decay_diff) * kkt).masked_fill(~strict_mask, 0.0)
    eye = torch.eye(C, device=q.device, dtype=torch.float32)
    # Exact unit-lower-triangular solve (I + A) * T = I
    T_mat = torch.linalg.solve_triangular(eye + A, eye.expand_as(A), upper=False, unitriangular=True)

    k_scaled = kf * (bf * torch.exp(cum_g)).unsqueeze(-1)
    v_scaled = vf * bf.unsqueeze(-1)
    W = torch.matmul(T_mat, k_scaled)  # (B, H, nc, C, D)
    U = torch.matmul(T_mat, v_scaled)  # (B, H, nc, C, D)

    qkt = torch.matmul(qf, kf.transpose(-1, -2)) * torch.exp(decay_diff)
    qkt = qkt.masked_fill(~causal_mask, 0.0)

    S = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
    outs: list[torch.Tensor] = []
    for c in range(nc):
        qc, kc = qf[:, :, c], kf[:, :, c]
        wc, uc = W[:, :, c], U[:, :, c]
        cg_c = cum_g[:, :, c]
        g_end = cg_c[:, :, -1:]  # (B, H, 1)
        v_new = uc - torch.matmul(wc, S)  # (B, H, C, D)
        o_inter = torch.matmul(qc * torch.exp(cg_c).unsqueeze(-1), S)
        o_intra = torch.matmul(qkt[:, :, c], v_new)
        outs.append(o_inter + o_intra)
        k_decay = kc * torch.exp((g_end - cg_c).clamp(max=0.0)).unsqueeze(-1)
        S = S * torch.exp(g_end).unsqueeze(-1) + torch.matmul(k_decay.transpose(-1, -2), v_new)

    out = torch.stack(outs, dim=2).reshape(B, H, T_pad, D)
    return out[:, :, :T, :].to(in_dtype)


def recurrent_gated_deltanet_ref(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, log_alpha: torch.Tensor, beta: torch.Tensor,
) -> torch.Tensor:
    """Reference exact token-by-token Gated DeltaNet for parity verification."""
    B, H, T, D = q.shape
    S = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
    outs: list[torch.Tensor] = []
    for t in range(T):
        qt, kt, vt = q[:, :, t].float(), k[:, :, t].float(), v[:, :, t].float()
        at = torch.exp(log_alpha[:, :, t].float()).unsqueeze(-1).unsqueeze(-1)
        bt = beta[:, :, t].float().unsqueeze(-1)
        S = S * at
        pred = torch.einsum("bhkd,bhk->bhd", S, kt)
        err = vt - pred
        S = S + torch.einsum("bhk,bhd->bhkd", kt, bt * err)
        outs.append(torch.einsum("bhkd,bhk->bhd", S, qt))
    return torch.stack(outs, dim=2).to(q.dtype)


class FastTokenBucketMoEV8(nn.Module):
    def __init__(self, d: int, n_exp: int, top_k: int, d_ff: int) -> None:
        super().__init__()
        self.d, self.n_exp, self.top_k, self.d_ff = d, n_exp, top_k, d_ff
        self.router = nn.Linear(d, n_exp, bias=False)
        self.w_up = nn.Parameter(torch.empty(n_exp, d_ff, d).normal_(0.0, 0.02))
        self.w_down = nn.Parameter(torch.empty(n_exp, d, d_ff).normal_(0.0, 0.02))

    def _quant(self, w: torch.Tensor) -> torch.Tensor:
        s = w.abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        q = torch.clamp(torch.round(w / s), -1.0, 1.0)
        return w + (q * s - w).detach()

    def forward(self, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = xn.shape
        N, E, K = B * T, self.n_exp, self.top_k
        x_flat = xn.reshape(N, D)
        probs = F.softmax(self.router(x_flat).float(), dim=-1)
        top_w, top_idx = torch.topk(probs, K, dim=-1)
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)).to(xn.dtype)
        f_e = F.one_hot(top_idx, E).float().sum(dim=1).mean(dim=0) / float(K)
        aux = float(E) * (f_e * probs.mean(dim=0)).sum()

        flat_e = top_idx.reshape(-1)
        flat_t = torch.arange(N, device=xn.device).repeat_interleave(K)
        flat_w = top_w.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        se, st, sw = flat_e[order], flat_t[order], flat_w[order]

        cap = max(1, (N * K * 3) // (2 * E) + 32)
        counts = torch.bincount(se, minlength=E)
        starts = F.pad(counts.cumsum(0)[:-1], (1, 0))
        pos = torch.arange(N * K, device=xn.device) - starts[se]
        valid = pos < cap
        ve, vp, vt, vw = se[valid], pos[valid], st[valid], sw[valid]

        b_in = torch.zeros((E, cap, D), device=xn.device, dtype=xn.dtype)
        b_in[ve, vp] = x_flat[vt]
        wu = self._quant(self.w_up).to(xn.dtype)
        wd = self._quant(self.w_down).to(xn.dtype)
        h = F.relu(torch.bmm(b_in, wu.transpose(1, 2))).clamp(max=6.0).square()
        b_out = torch.bmm(h, wd.transpose(1, 2))

        out = torch.zeros_like(x_flat)
        out.index_add_(0, vt, b_out[ve, vp] * vw.unsqueeze(-1))
        return out.reshape(B, T, D), aux


class GatedDeltaNetMoEBlock(nn.Module):
    """Multi-Head Gated DeltaNet + BitNet Top-2 MoE block."""

    def __init__(self, cfg: V8Config) -> None:
        super().__init__()
        d, nh = cfg.d_model, cfg.delta_heads
        self.nh, self.hd, self.chunk = nh, d // nh, cfg.delta_chunk
        self.norm1 = nn.Parameter(torch.ones(d))
        self.wq = BitLinearV8(d, d)
        self.wk = BitLinearV8(d, d)
        self.wv = BitLinearV8(d, d)
        self.wo = BitLinearV8(d, d)
        self.w_ab = nn.Linear(d, 2 * nh, bias=True)
        nn.init.constant_(self.w_ab.bias[:nh], 2.0)   # start with slow decay alpha ~ 0.88
        nn.init.constant_(self.w_ab.bias[nh:], -1.0)  # moderate initial write rate beta ~ 0.27
        self.norm2 = nn.Parameter(torch.ones(d))
        self.moe = FastTokenBucketMoEV8(d, cfg.n_experts, cfg.top_k, cfg.d_ff)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        xn = rms_norm(x, self.norm1)
        q = F.normalize(self.wq(xn).view(B, T, self.nh, self.hd).transpose(1, 2).float(), dim=-1, eps=1e-5).to(x.dtype)
        k = F.normalize(self.wk(xn).view(B, T, self.nh, self.hd).transpose(1, 2).float(), dim=-1, eps=1e-5).to(x.dtype)
        v = self.wv(xn).view(B, T, self.nh, self.hd).transpose(1, 2)
        ab = self.w_ab(xn).float()
        log_alpha = F.logsigmoid(ab[:, :, : self.nh]).transpose(1, 2).clamp(min=-6.0)
        beta = torch.sigmoid(ab[:, :, self.nh :]).transpose(1, 2)
        o = chunkwise_gated_deltanet(q, k, v, log_alpha, beta, chunk_size=self.chunk)
        x = x + self.wo(o.transpose(1, 2).reshape(B, T, D))
        m, aux = self.moe(rms_norm(x, self.norm2))
        return x + m, aux


class HuginnSharedKVAttnBlock(nn.Module):
    """Sliding-Window RoPE Attention (W=1024) designed for Shared-KV recurrent depth."""

    def __init__(self, cfg: V8Config) -> None:
        super().__init__()
        d, nh = cfg.d_model, cfg.attn_heads
        self.nh, self.hd, self.win = nh, d // nh, cfg.attn_window
        self.norm1 = nn.Parameter(torch.ones(d))
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.norm2 = nn.Parameter(torch.ones(d))
        self.ff_up = BitLinearV8(d, cfg.d_ff)
        self.ff_dn = BitLinearV8(cfg.d_ff, d)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        self.register_buffer("inv_freq", inv, persistent=False)

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[2]
        ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * self.inv_freq[None, :]
        c, s = ang.cos().to(x.dtype)[None, None], ang.sin().to(x.dtype)[None, None]
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, kv_shared: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, D = x.shape
        xn = rms_norm(x, self.norm1)
        q = self._rope(self.wq(xn).view(B, T, self.nh, self.hd).transpose(1, 2))
        if kv_shared is None:
            k = self._rope(self.wk(xn).view(B, T, self.nh, self.hd).transpose(1, 2))
            v = self.wv(xn).view(B, T, self.nh, self.hd).transpose(1, 2)
            kv_shared = (k, v)
        else:
            k, v = kv_shared
        if T <= self.win:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            idx = torch.arange(T, device=x.device)
            dist = idx[:, None] - idx[None, :]
            mask = (dist >= 0) & (dist < self.win)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.wo(o.transpose(1, 2).reshape(B, T, D))
        h = F.relu(self.ff_up(rms_norm(x, self.norm2))).clamp(max=6.0).square()
        return x + self.ff_dn(h), kv_shared


class PeerLayerV8(nn.Module):
    def __init__(self, cfg: V8Config) -> None:
        super().__init__()
        self.d, self.nh, self.sub_k, self.top_k = cfg.d_model, cfg.peer_heads, cfg.peer_sub_k, cfg.peer_k
        self.sub_d = self.d // self.nh
        self.norm = nn.Parameter(torch.ones(self.d))
        self.wq = nn.Linear(self.d, self.d, bias=False)
        self.keys_a = nn.Parameter(torch.empty(self.nh, self.sub_k, self.sub_d).normal_(0.0, 0.05))
        self.keys_b = nn.Parameter(torch.empty(self.nh, self.sub_k, self.sub_d).normal_(0.0, 0.05))
        self.u_exp = nn.Parameter(torch.empty(self.sub_k * self.sub_k, self.d).normal_(0.0, 0.015))
        self.v_exp = nn.Parameter(torch.empty(self.sub_k * self.sub_k, self.d).normal_(0.0, 0.015))

    def _q14(self, w: torch.Tensor) -> torch.Tensor:
        q = torch.clamp(torch.round(w.float() * 16384.0), -32767.0, 32767.0) / 16384.0
        return w + (q.to(w.dtype) - w).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        xn = rms_norm(x, self.norm)
        q = self.wq(xn).view(B * T, self.nh, self.sub_d)
        sa = torch.einsum("nhd,hkd->nhk", q, self.keys_a)
        sb = torch.einsum("nhd,hkd->nhk", q, self.keys_b)
        va, ia = torch.topk(sa, self.top_k, dim=-1)
        vb, ib = torch.topk(sb, self.top_k, dim=-1)
        comb_v = (va.unsqueeze(-1) + vb.unsqueeze(-2)).reshape(B * T, self.nh, self.top_k * self.top_k)
        comb_i = (ia.unsqueeze(-1) * self.sub_k + ib.unsqueeze(-2)).reshape(B * T, self.nh, self.top_k * self.top_k)
        tv, ti = torch.topk(comb_v, self.top_k, dim=-1)
        eids = torch.gather(comb_i, -1, ti)
        w = F.softmax(tv.float(), dim=-1).to(x.dtype)
        ue = self._q14(self.u_exp[eids])
        ve = self._q14(self.v_exp[eids])
        dot = (xn.reshape(B * T, 1, 1, D) * ue).sum(dim=-1)
        act = F.relu(dot).clamp(max=6.0).square()
        delta = ((w * act).unsqueeze(-1) * ve).sum(dim=(1, 2))
        return x + delta.view(B, T, D)


class GenesisV8Native(nn.Module):
    """Genesis-v8 Native BICA: Engram + Gated DeltaNet + Huginn Shared-KV W=1024 + PEER."""

    def __init__(self, cfg: V8Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        nn.init.normal_(self.emb.weight, 0.0, 0.02)
        self.engram = EngramMemoryV8(cfg)
        # Prelude: DN0 -> Attn0(Shared-KV)
        self.prelude_dn = GatedDeltaNetMoEBlock(cfg)
        self.prelude_attn = HuginnSharedKVAttnBlock(cfg)
        # Recurrent Core (looped R times with Huginn Shared-KV on core_attn!)
        self.core_dn1 = GatedDeltaNetMoEBlock(cfg)
        self.peer0 = PeerLayerV8(cfg)
        self.core_attn = HuginnSharedKVAttnBlock(cfg)
        self.core_dn2 = GatedDeltaNetMoEBlock(cfg)
        self.peer1 = PeerLayerV8(cfg)
        self.recur_inj = nn.Parameter(torch.full((cfg.d_model,), 0.25))
        # Coda: DN_coda1 -> DN_coda2 -> norm_f
        self.coda_dn1 = GatedDeltaNetMoEBlock(cfg)
        self.coda_dn2 = GatedDeltaNetMoEBlock(cfg)
        self.norm_f = nn.Parameter(torch.ones(cfg.d_model))

    def forward(self, tok_ids: torch.Tensor, recur_steps: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        r_steps = max(1, int(recur_steps or self.cfg.recur_steps))
        x = self.engram(tok_ids, self.emb(tok_ids))
        aux_tot = x.new_zeros((), dtype=torch.float32)

        x, a = self.prelude_dn(x); aux_tot = aux_tot + a
        x, _ = self.prelude_attn(x, kv_shared=None)
        s0 = x
        core_kv = None
        for r in range(r_steps):
            if r > 0:
                x = x + torch.tanh(self.recur_inj).to(x.dtype) * s0
            x, a = self.core_dn1(x); aux_tot = aux_tot + a
            x = self.peer0(x)
            x, core_kv = self.core_attn(x, kv_shared=core_kv)  # Huginn Shared-KV across all R!
            x, a = self.core_dn2(x); aux_tot = aux_tot + a
            x = self.peer1(x)
        x, a = self.coda_dn1(x); aux_tot = aux_tot + a
        x, a = self.coda_dn2(x); aux_tot = aux_tot + a
        xn = rms_norm(x, self.norm_f)
        logits = F.linear(xn, self.emb.weight.to(xn.dtype))
        logits = 30.0 * torch.tanh(logits / 30.0)
        return logits, aux_tot

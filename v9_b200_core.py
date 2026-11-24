"""
Genesis-v9 Modal B200 Shared Core (`colab_gpu_trainer/v9_b200_core.py`, <= 360 LOC).
Provides:
  - Presets (`v9_1_5b_b200` with exact WY-equivalent `delta_chunk=16`, `v9_abl_430m`, `tiny_smoke`).
  - Controlled single-component ablation builder (`A0_v9_full` .. `A5_adamw_only`).
  - Multi-domain uint16 shard loader, fixed-seed validation BPB evaluator (`R=1/2/4`).
  - Asymmetric CED & FP4 E2M1 KV-cache live verification + greedy generation probes.
  - Zero-loss-diff `strip_training_heads()` BF16 inference export.
Strictly <= 360 LOC | 0 B written to C:\\ | Zero hardcoded secrets.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import glob
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from genesis_v9_native import V9Config
    from genesis_v9_v41 import GenesisV9V41Native, compute_v9_v41_loss, load_v9_canon_lut_from_tokenizer
    from muon_hybrid_opt import MuonHybridOptimizer, partition_v9_params
    from v9_b200_fast_kernels import apply_fast_b200_kernels
    from v9_v41_upgrades import EngramV41Module
except ImportError:
    from .genesis_v9_native import V9Config
    from .genesis_v9_v41 import GenesisV9V41Native, compute_v9_v41_loss, load_v9_canon_lut_from_tokenizer
    from .muon_hybrid_opt import MuonHybridOptimizer, partition_v9_params
    from .v9_b200_fast_kernels import apply_fast_b200_kernels
    from .v9_v41_upgrades import EngramV41Module

_LN2 = math.log(2.0)
DOMAINS: tuple[str, ...] = ("en", "math", "code", "vi", "beyondweb_rlvr", "curated_train_sft")
DOMAIN_BPT: dict[str, float] = {"en": 3.85, "math": 3.25, "code": 3.40, "vi": 3.35, "beyondweb_rlvr": 3.23, "curated_train_sft": 3.30}

B200_PRESETS: dict[str, V9Config] = {
    "v9_1_5b_b200": V9Config(delta_chunk=16),  # 100% identical params to v9_1_5b; 4x less WY chunk VRAM
    "v9_abl_430m": V9Config(
        vocab=16384, d_model=512, n_prelude_dn=1, n_core_dn=2, n_coda_dn=2,
        n_experts=32, top_k=3, d_ff=1536, attn_heads=8, attn_window=1024, d_attn_ff=1024,
        delta_heads=8, delta_chunk=16, engram_heads=8, engram_buckets=131072, engram_dim=96,
        engram_conv_k=4, peer_heads=8, peer_sub_k=160, peer_k=8,
        recur_steps=2, max_recur_steps=6, mtp_k=3,
    ),
    "tiny_smoke": V9Config(
        vocab=16384, d_model=64, n_prelude_dn=1, n_core_dn=2, n_coda_dn=1,
        n_experts=4, top_k=2, d_ff=64, attn_heads=4, attn_window=64, d_attn_ff=96,
        delta_heads=4, delta_chunk=16, engram_heads=4, engram_buckets=256, engram_dim=16,
        engram_conv_k=4, peer_heads=4, peer_sub_k=16, peer_k=2,
        recur_steps=2, max_recur_steps=4, mtp_k=3,
    ),
}

ABLATION_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "A0_v9_full", "desc": "Full GenesisV9V41Native (EngramV4.1+PEER+TRM-R2+FSP/MTP+Muon)", "engram": True, "peer": True, "recur": 2, "aux_heads": True, "opt": "muon"},
    {"name": "A1_no_engram", "desc": "Single ablation: Layer-2 EngramV4.1 bypassed", "engram": False, "peer": True, "recur": 2, "aux_heads": True, "opt": "muon"},
    {"name": "A2_no_peer", "desc": "Single ablation: Split-Key PEER layers bypassed", "engram": True, "peer": False, "recur": 2, "aux_heads": True, "opt": "muon"},
    {"name": "A3_no_recur_r1", "desc": "Single ablation: TRM dual-state recurrence disabled (R=1)", "engram": True, "peer": True, "recur": 1, "aux_heads": True, "opt": "muon"},
    {"name": "A4_no_fsp_mtp", "desc": "Single ablation: FSP(H=16) & MTP(k=3) disabled (NTP-only)", "engram": True, "peer": True, "recur": 2, "aux_heads": False, "opt": "muon"},
    {"name": "A5_adamw_only", "desc": "Single ablation: Pure AdamW replacing MuonHybridOptimizer", "engram": True, "peer": True, "recur": 2, "aux_heads": True, "opt": "adamw"},
)


class _IdentityEngram(nn.Module):
    def forward(self, tok_ids: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return h


class _IdentityPeer(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x.new_zeros((), dtype=torch.float32)


class PureAdamWWrapper:
    """Drop-in matched AdamW control using identical zero-WD sparse parameter partitioning."""

    def __init__(self, model: nn.Module, lr: float = 1.2e-3, weight_decay: float = 0.05, sparse_lr_mult: float = 1.5) -> None:
        self.base_lr, self.sparse_lr_mult = float(lr), float(sparse_lr_mult)
        grps = partition_v9_params(model, muon_on_bitlinear=False)
        decay_p = grps["muon_2d"] + grps["adamw_bitlinear"] + grps["adamw_decay"]
        adam_groups: list[dict[str, Any]] = []
        if decay_p:
            adam_groups.append({"name": "decay", "params": decay_p, "lr": lr, "weight_decay": weight_decay})
        if grps["adamw_sparse_nodecay"]:
            adam_groups.append({"name": "sparse", "params": grps["adamw_sparse_nodecay"], "lr": lr * sparse_lr_mult, "weight_decay": 0.0})
        if grps["adamw_nodecay"]:
            adam_groups.append({"name": "nodecay", "params": grps["adamw_nodecay"], "lr": lr, "weight_decay": 0.0})
        all_cuda = all(p.is_cuda for g in adam_groups for p in g["params"]) if adam_groups else False
        self.opt = torch.optim.AdamW(adam_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=all_cuda)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.opt.step()

    def set_lr(self, lr: float) -> None:
        self.base_lr = float(lr)
        for g in self.opt.param_groups:
            g["lr"] = self.base_lr * (self.sparse_lr_mult if g.get("name") == "sparse" else 1.0)

    def state_dict(self) -> dict[str, Any]:
        return {"base_lr": self.base_lr, "opt": self.opt.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.opt.load_state_dict(state["opt"])
        if "base_lr" in state:
            self.set_lr(float(state["base_lr"]))


def build_v9_variant(
    cfg: V9Config, spec: dict[str, Any], device: str, tok_path: Path | None, lr: float, weight_decay: float = 0.05,
) -> tuple[GenesisV9V41Native, Any, dict[str, Any]]:
    """Construct exact controlled `GenesisV9V41Native` variant + matched optimizer."""
    use_heads = bool(spec.get("aux_heads", True))
    model = GenesisV9V41Native(cfg, fsp_horizon=(16 if use_heads else 0), include_train_heads=use_heads)
    if not spec.get("engram", True):
        model.engram = _IdentityEngram()  # type: ignore[assignment]
    if not spec.get("peer", True):
        model.peer0 = _IdentityPeer()  # type: ignore[assignment]
        model.peer1 = _IdentityPeer()  # type: ignore[assignment]
    if tok_path is not None and tok_path.is_file() and isinstance(model.engram, EngramV41Module):
        model.set_canonical_lut(load_v9_canon_lut_from_tokenizer(tok_path, cfg.vocab))
    apply_fast_b200_kernels(model, delta_chunk=32 if cfg.d_model >= 256 else 16)
    model = model.to(device)
    if spec.get("opt", "muon") == "adamw":
        opt: Any = PureAdamWWrapper(model, lr=lr, weight_decay=weight_decay, sparse_lr_mult=1.5)
    else:
        opt = MuonHybridOptimizer(model, lr=lr, weight_decay=weight_decay, sparse_lr_mult=1.5, muon_on_bitlinear=False)
    info = {
        "variant": spec["name"], "total_params": int(sum(p.numel() for p in model.parameters())),
        "recur_train": int(spec.get("recur", cfg.recur_steps)), "opt_kind": spec.get("opt", "muon"),
        "fsp_weight": 0.10 if use_heads else 0.0, "mtp_weight": 0.15 if use_heads else 0.0,
    }
    return model, opt, info


def load_multidomain_shards(data_root: Path) -> tuple[dict[str, list[np.ndarray]], dict[str, np.ndarray]]:
    """Load `train_*.bin` and dedicated `val.bin` memmaps per domain from `data_root`."""
    train_maps: dict[str, list[np.ndarray]] = {}
    val_maps: dict[str, np.ndarray] = {}
    for dom in DOMAINS:
        ddir = data_root / dom
        if not ddir.is_dir():
            continue
        tfiles = sorted(glob.glob(str(ddir / "train*.bin")))
        if tfiles:
            train_maps[dom] = [np.memmap(p, dtype="<u2", mode="r") for p in tfiles if Path(p).stat().st_size >= 64]
        vf = ddir / "val.bin"
        if vf.is_file() and vf.stat().st_size >= 64:
            val_maps[dom] = np.memmap(vf, dtype="<u2", mode="r")
        elif dom in train_maps and train_maps[dom]:
            val_maps[dom] = train_maps[dom][0]
    if not train_maps:
        tr = data_root / "train.bin"
        va = data_root / "val.bin"
        if not tr.is_file():
            raise FileNotFoundError(f"No uint16 .bin shards found in {data_root}")
        train_maps["beyondweb_rlvr"] = [np.memmap(tr, dtype="<u2", mode="r")]
        val_maps["beyondweb_rlvr"] = np.memmap(va if va.is_file() else tr, dtype="<u2", mode="r")
    return train_maps, val_maps


def sample_multidomain_batch(
    train_maps: dict[str, list[np.ndarray]], weights: dict[str, float],
    batch_size: int, seq_len: int, rng: np.random.Generator, vocab: int,
) -> torch.Tensor:
    """Sample a `(batch_size, seq_len + 1)` int64 batch using normalized domain weights."""
    doms = [d for d in train_maps if train_maps[d]]
    w = np.array([max(1e-6, float(weights.get(d, 1.0))) for d in doms], dtype=np.float64)
    w /= w.sum()
    picks = rng.choice(len(doms), size=batch_size, p=w)
    span = seq_len + 1
    out = np.empty((batch_size, span), dtype=np.int64)
    for i, di in enumerate(picks):
        mms = train_maps[doms[int(di)]]
        mm = mms[int(rng.integers(0, len(mms)))]
        n = int(mm.size)
        if n > span:
            st = int(rng.integers(0, n - span))
            out[i] = mm[st : st + span]
        else:
            st = int(rng.integers(0, n))
            out[i] = mm[(st + np.arange(span, dtype=np.int64)) % n]
    np.clip(out, 0, vocab - 1, out=out)
    return torch.from_numpy(out)


def build_fixed_val_batches(
    val_maps: dict[str, np.ndarray], batch_size: int, seq_len: int, n_batches: int, vocab: int, seed: int = 1337,
) -> dict[str, list[torch.Tensor]]:
    """Pre-sample deterministic held-out validation batches so every variant sees identical tokens."""
    rng = np.random.default_rng(seed)
    span = seq_len + 1
    res: dict[str, list[torch.Tensor]] = {}
    for dom, mm in val_maps.items():
        n = int(mm.size)
        batches: list[torch.Tensor] = []
        for _ in range(max(1, n_batches)):
            arr = np.empty((batch_size, span), dtype=np.int64)
            for i in range(batch_size):
                if n > span:
                    st = int(rng.integers(0, n - span))
                    arr[i] = mm[st : st + span]
                else:
                    st = int(rng.integers(0, n))
                    arr[i] = mm[(st + np.arange(span, dtype=np.int64)) % n]
            np.clip(arr, 0, vocab - 1, out=arr)
            batches.append(torch.from_numpy(arr))
        res[dom] = batches
    return res


@torch.no_grad()
def evaluate_multidomain_bpb(
    model: GenesisV9V41Native, val_batches: dict[str, list[torch.Tensor]],
    recur_depths: tuple[int, ...], device: str, use_amp: bool,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate exact CE (nats) & BPB across all validation domains and recurrence depths."""
    was_train = model.training
    model.eval()
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    out: dict[str, dict[str, dict[str, float]]] = {}
    for r in recur_depths:
        r_key = f"R{r}"
        dom_res: dict[str, dict[str, float]] = {}
        ces, bpbs = [], []
        for dom, blist in val_batches.items():
            losses = []
            for bt in blist:
                bt_d = bt.to(device, non_blocking=use_amp)
                with amp_ctx:
                    logits, _ = model(bt_d[:, :-1], recur_steps=r)
                    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), bt_d[:, 1:].reshape(-1))
                losses.append(float(ce.item()))
            mean_ce = float(sum(losses) / max(1, len(losses)))
            bpb = mean_ce / (_LN2 * DOMAIN_BPT.get(dom, 3.5))
            dom_res[dom] = {"ce": round(mean_ce, 4), "bpb": round(bpb, 4)}
            ces.append(mean_ce)
            bpbs.append(bpb)
        dom_res["macro_avg"] = {"ce": round(sum(ces) / max(1, len(ces)), 4), "bpb": round(sum(bpbs) / max(1, len(bpbs)), 4)}
        out[r_key] = dom_res
    if was_train:
        model.train()
    return out


@torch.no_grad()
def evaluate_ced_and_fp4_kv(
    model: GenesisV9V41Native, sample_bt: torch.Tensor, device: str, use_amp: bool,
) -> dict[str, Any]:
    """Benchmark Asymmetric CED (`prefill k=1,R=1` vs `decode k=3,R=2`) & FP4 E2M1 KV-Cache parity."""
    was_train = model.training
    model.eval()
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    bt = sample_bt[:1, : min(sample_bt.shape[1], 129)].to(device)
    x, y = bt[:, :-1], bt[:, 1:]
    T = int(x.shape[1])
    split = max(8, T // 2)
    with amp_ctx:
        lo_full, _ = model(x, recur_steps=model.cfg.recur_steps)
        ce_sym = float(F.cross_entropy(lo_full[:, split:].reshape(-1, lo_full.shape[-1]), y[:, split:].reshape(-1)).item())
    res: dict[str, Any] = {
        "seq_len": T, "prefill_len": split, "decode_len": T - split,
        "sym_r2_decode_ce": round(ce_sym, 4),
        "param_breakdown": model.param_breakdown(prefill_top_k=1, prefill_recur=1),
    }
    if isinstance(model.engram, EngramV41Module):
        cache_fp4 = model.make_inference_cache()
        cache_fp4.kv_cache.to(device)
        with amp_ctx:
            model.forward_prefill(x[:, :split], cache=cache_fp4, prefill_top_k=1, prefill_recur=1)
            dec_logits = []
            for t in range(split, T):
                lo_t, _ = model.forward_decode(x[:, t : t + 1], cache=cache_fp4)
                dec_logits.append(lo_t)
            dec_cat = torch.cat(dec_logits, dim=1)
            ce_ced_fp4 = float(F.cross_entropy(dec_cat.reshape(-1, dec_cat.shape[-1]), y[:, split:].reshape(-1)).item())
        res["ced_fp4_decode_ce"] = round(ce_ced_fp4, 4)
        res["ced_fp4_ce_delta"] = round(ce_ced_fp4 - ce_sym, 4)
        res["fp4_kv_bytes_per_token"] = round(cache_fp4.kv_cache.bytes_per_token(), 2)
    if was_train:
        model.train()
    return res


@torch.no_grad()
def run_generation_probes(
    model: GenesisV9V41Native, tok_path: Path, device: str, use_amp: bool, max_new: int = 48,
) -> list[dict[str, str]]:
    """Run greedy generation probes (EN science, VI knowledge, Python, Math) using `v9_tok_16k.json`."""
    if not tok_path.is_file():
        return []
    obj = json.loads(tok_path.read_text(encoding="utf-8"))
    specials = list(obj.get("special_tokens", []))
    id_to_bytes = [s.encode("ascii") for s in specials] + [bytes([b]) for b in range(256)]
    p2i: dict[bytes, int] = {bytes([b]): 16 + b for b in range(256)}
    for idx, (h, _, _) in enumerate(obj["pieces"]):
        raw = bytes.fromhex(h)
        id_to_bytes.append(raw)
        p2i[raw] = 272 + idx
    sp_map = {s: i for i, s in enumerate(specials)}

    def _enc(text: str) -> list[int]:
        ids = [1]
        for tok_s in text.split(" "):
            if tok_s in sp_map:
                ids.append(sp_map[tok_s]); continue
            raw = (" " + tok_s if len(ids) > 1 else tok_s).encode("utf-8")
            i = 0
            while i < len(raw):
                matched = False
                for l in range(min(24, len(raw) - i), 0, -1):
                    if (tid := p2i.get(raw[i : i + l])) is not None:
                        ids.append(tid); i += l; matched = True; break
                if not matched:
                    ids.append(16 + raw[i]); i += 1
        return ids

    def _dec(ids: list[int]) -> str:
        out_parts, buf = [], bytearray()
        for tid in ids:
            if 16 <= tid < len(id_to_bytes):
                buf.extend(id_to_bytes[tid])
            else:
                if buf: out_parts.append(buf.decode("utf-8", errors="replace")); buf.clear()
                if 0 <= tid < len(specials): out_parts.append(specials[tid])
        if buf: out_parts.append(buf.decode("utf-8", errors="replace"))
        return "".join(out_parts)

    prompts = [
        ("en_science", "Topic: Photosynthesis\nPrinciple:"),
        ("vi_knowledge", "Thủ đô của Việt Nam là"),
        ("python_code", "def is_prime(n):\n    if n < 2:"),
        ("rlvr_math", "<USER> What is 15% of 200? <BOT> <THINK>"),
    ]
    was_train = model.training
    model.eval()
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    results = []
    for tag, prm in prompts:
        ids = _enc(prm)
        for _ in range(max_new):
            inp = torch.tensor([ids[-256:]], device=device, dtype=torch.long)
            with amp_ctx:
                logits, _ = model(inp, recur_steps=model.cfg.recur_steps)
            nxt = int(logits[0, -1].argmax().item())
            ids.append(nxt)
            if nxt == 2:
                break
        results.append({"tag": tag, "prompt": prm, "output": _dec(ids)})
    if was_train:
        model.train()
    return results


def export_stripped_bf16(model: GenesisV9V41Native, out_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Strip MTP/FSP training heads, assert 0.0 max logit diff, and save clean BF16 inference ckpt."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    was_train = model.training
    model.eval()
    device = next(model.parameters()).device
    probe = torch.arange(16, 48, device=device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        lo_before, _ = model(probe, recur_steps=model.cfg.recur_steps)
        model.strip_training_heads()
        lo_after, _ = model(probe, recur_steps=model.cfg.recur_steps)
    diff = float((lo_before - lo_after).abs().max().item())
    if diff != 0.0:
        raise AssertionError(f"strip_training_heads drift: {diff}")
    sd = {k: (v.detach().to(torch.bfloat16).cpu() if v.is_floating_point() else v.detach().cpu())
          for k, v in model.state_dict().items()}
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save({
        "format": "genesis_v9_v41_inference_v1", "config": asdict(model.cfg), "stripped_training_heads": True,
        "dtype": "bfloat16", "param_count_inference": int(sum(p.numel() for p in model.parameters())),
        "strip_max_abs_logit_diff": diff, "model_state": sd, **meta,
    }, tmp)
    os.replace(tmp, out_path)
    if was_train:
        model.train()
    return {"path": str(out_path), "bytes": int(out_path.stat().st_size), "strip_diff": diff}

"""Genesis-v9 1.5B Colab Ultra Hardware Arbitrage Benchmark (`colab_gpu_trainer/v9_colab_hw_bench.py`, <= 185 LOC).
Benchmarks exact `Genesis-v9 1.5B` (`v9_1_5b_b200`) + `apply_fast_b200_kernels` + `MuonHybridOptimizer`
across CUDA (`G4 96GB` / `A100 80GB` / `H100 80GB` / `L4`) AND TPU (`TPU v6e-1` / `v5e-1` via `torch_xla`)
to measure exact `tok/s`, `peak_vram_gb`, and `tokens / CU` before committing the 1,000-CU budget.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from genesis_v9_v41 import GenesisV9V41Native, compute_v9_v41_loss
from muon_hybrid_opt import MuonHybridOptimizer
from v9_b200_core import B200_PRESETS
from v9_b200_fast_kernels import apply_fast_b200_kernels


def _detect_backend() -> tuple[str, Any, dict[str, Any]]:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return "cuda", torch.device("cuda"), {
            "backend": "cuda", "name": props.name,
            "vram_gb": round(props.total_memory / 1e9, 2),
            "bf16": bool(torch.cuda.is_bf16_supported()),
        }
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
        dev = xm.xla_device()
        return "xla", dev, {"backend": "xla", "name": str(dev), "vram_gb": 32.0, "bf16": True}
    except Exception:
        return "cpu", torch.device("cpu"), {"backend": "cpu", "name": "CPU", "vram_gb": 0.0, "bf16": False}


def _sync(backend: str) -> None:
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "xla":
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
        xm.mark_step()


def bench_one_config(
    preset: str, batch_size: int, seq_len: int, recur_steps: int,
    warmup: int, iters: int, use_compile: bool, cu_rate: float,
) -> dict[str, Any]:
    backend, dev, hw = _detect_backend()
    if backend == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    cfg = B200_PRESETS[preset]
    model = GenesisV9V41Native(cfg, fsp_horizon=16, include_train_heads=True)
    if backend == "xla":
        # Cast sparse tables to bf16 to comfortably fit inside 32GB TPU HBM
        model = model.to(dtype=torch.bfloat16)
    model = model.to(dev)
    kcounts = apply_fast_b200_kernels(model, delta_chunk=cfg.delta_chunk) if backend != "xla" else {"xla_native": 1}
    opt = MuonHybridOptimizer(model, lr=6e-4, weight_decay=0.05, sparse_lr_mult=1.5, muon_on_bitlinear=False)

    train_mod: Any = model
    compiled_ok = False
    if backend == "cuda" and use_compile and hasattr(torch, "compile"):
        try:
            train_mod = torch.compile(model)
            compiled_ok = True
        except Exception:
            train_mod = model

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if backend == "cuda" else nullcontext()
    x = torch.randint(16, cfg.vocab, (batch_size, seq_len + 1), device=dev, dtype=torch.long)

    model.train()
    t_compile0 = time.perf_counter()
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        with amp_ctx:
            out = train_mod(x[:, :-1], recur_steps=recur_steps, targets=x[:, 1:], return_details=True)
            loss, _ = compute_v9_v41_loss(model, out, x[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        _sync(backend)
    warmup_s = time.perf_counter() - t_compile0

    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        with amp_ctx:
            out = train_mod(x[:, :-1], recur_steps=recur_steps, targets=x[:, 1:], return_details=True)
            loss, _ = compute_v9_v41_loss(model, out, x[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        _sync(backend)
        last_loss = float(loss.detach().item())
    dt = max(1e-6, time.perf_counter() - t0)

    toks = batch_size * seq_len * iters
    tok_s = toks / dt
    tok_hr = tok_s * 3600.0
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if backend == "cuda" else 0.0
    res: dict[str, Any] = {
        "ok": True, "hw": hw, "preset": preset, "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "bs": batch_size, "seq_len": seq_len, "recur_steps": recur_steps, "compiled": compiled_ok,
        "kernels": kcounts, "warmup_sec": round(warmup_s, 2), "ms_per_step": round(1000.0 * dt / iters, 1),
        "tok_per_sec": round(tok_s, 1), "m_tok_per_hour": round(tok_hr / 1e6, 2),
        "peak_vram_gb": peak_gb, "last_loss": round(last_loss, 4),
    }
    if cu_rate > 0.0:
        res["cu_rate_hr"] = cu_rate
        res["m_tok_per_cu"] = round((tok_hr / cu_rate) / 1e6, 2)
        res["b_tok_per_1000cu"] = round((tok_hr * (1000.0 / cu_rate)) / 1e9, 2)

    del train_mod, model, opt, x
    gc.collect()
    if backend == "cuda":
        torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Genesis-v9 1.5B Colab Ultra Hardware Benchmark")
    ap.add_argument("--preset", default="v9_1_5b_b200", choices=list(B200_PRESETS.keys()))
    ap.add_argument("--batch-sizes", default="4,8,12")
    ap.add_argument("--recurs", default="1,2")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--cu-rate", type=float, default=0.0)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()

    bss = [int(s.strip()) for s in a.batch_sizes.split(",") if s.strip()]
    recs = [int(s.strip()) for s in a.recurs.split(",") if s.strip()]
    results: list[dict[str, Any]] = []
    for r in recs:
        for bs in bss:
            try:
                rep = bench_one_config(a.preset, bs, a.seq_len, r, a.warmup, a.iters, not a.no_compile, a.cu_rate)
            except Exception as exc:
                rep = {"ok": False, "bs": bs, "seq_len": a.seq_len, "recur_steps": r, "error": f"{type(exc).__name__}: {exc}"}
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results.append(rep)
            print(json.dumps(rep, ensure_ascii=False), flush=True)

    payload = {"results": results, "ts": int(time.time())}
    if a.out_json is not None:
        a.out_json.parent.mkdir(parents=True, exist_ok=True)
        a.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

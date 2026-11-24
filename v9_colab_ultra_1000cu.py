"""Genesis-v9 1.5B Colab Pro+ Ultra 1,000-CU Production Trainer (`colab_gpu_trainer/v9_colab_ultra_1000cu.py`, <= 395 LOC).
Upgrades over `train_v9_colab_200cu.py`:
  1. Drops in `apply_fast_b200_kernels(model, delta_chunk=16)` (`3.6x-6.0x` speedup, 0 NaN).
  2. `--init-from <bf16_inference_or_train_ckpt.pt>` warm-start (`strict=False` to re-attach MTP/FSP heads).
  3. Multi-shard weighted memmap mixture (`--mix-spec shard1.bin:0.4,shard2.bin:0.2,...`).
  4. Dynamic Ouro/Huginn `R=1 -> R=2` curriculum (`--r1-warmup-frac 0.60` for `+35%` free pretrain tokens).
  5. Exact `<BOT>` (id=4) -> `<EOS>` (id=2) / `<USER>` (id=3) SFT loss masking (`--sft-mask-bot`).
  6. Atomic Drive checkpointing + live `CU` guard + zero-diff BF16 inference export (`strip_diff == 0.0`).
Strictly <= 400 LOC | 0 B on C:\\ | Zero hardcoded secrets.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from genesis_v9_v41 import DEFAULT_V9_TOK_PATH, GenesisV9V41Native, compute_v9_v41_loss, load_v9_canon_lut_from_tokenizer
from v9_b200_core import B200_PRESETS
from v9_b200_fast_kernels import apply_fast_b200_kernels
from v9_data_efficiency import WSDScheduler
from v9_ultra_speed_accel import (
    AsyncMixPrefetcher, ZeroSyncLossAccumulator, build_fast_optimizer,
    compile_v9_fast, compute_v9_v41_loss_nosync, recommend_gpu_profile,
)

TOK_EOS, TOK_USER, TOK_BOT = 2, 3, 4


def _guard_path(path: Path) -> Path:
    res = path.resolve()
    if os.name == "nt" and str(res).lower().startswith("c:"):
        raise RuntimeError(f"Drive C:\\ write forbidden: {res}")
    return res


def estimate_ultra_cu_rate(device: str) -> tuple[str, float]:
    if device != "cuda" or not torch.cuda.is_available():
        return "CPU", 0.0
    name = torch.cuda.get_device_name(0)
    up, vram_gb = name.upper(), torch.cuda.get_device_properties(0).total_memory / 1e9
    if "H100" in up: return name, 18.05
    if "6000" in up or "BLACKWELL" in up or "B200" in up or vram_gb >= 90.0: return name, 8.80
    if "A100" in up: return name, (7.10 if vram_gb >= 60.0 else 5.35)
    if "L4" in up: return name, 4.82
    if "T4" in up: return name, 1.96
    return name, 8.0


def parse_mix_spec(mix_spec: str, data_dir: Path) -> list[tuple[str, np.ndarray, float]]:
    """Load one or more `<u2` memmap shards with normalized sampling weights."""
    items: list[tuple[str, np.ndarray, float]] = []
    if mix_spec.strip():
        for raw in mix_spec.split(","):
            raw = raw.strip()
            if not raw: continue
            if ":" in raw and not (len(raw) > 2 and raw[1] == ":" and raw.count(":") == 1):
                p_str, w_str = raw.rsplit(":", 1)
                w = float(w_str)
            else:
                p_str, w = raw, 1.0
            p = Path(p_str)
            if not p.is_file():
                p = data_dir / p_str
            if not p.is_file():
                raise FileNotFoundError(f"Missing shard in --mix-spec: {p_str}")
            mm = np.memmap(p, dtype="<u2", mode="r")
            if mm.size >= 8:
                items.append((p.name, mm, max(1e-6, w)))
    else:
        p = data_dir / "train.bin"
        if not p.is_file():
            raise FileNotFoundError(f"Missing train.bin in {data_dir}")
        items.append((p.name, np.memmap(p, dtype="<u2", mode="r"), 1.0))
    tot_w = sum(w for _, _, w in items)
    return [(n, mm, w / tot_w) for n, mm, w in items]


def sample_mix_batch(
    shards: list[tuple[str, np.ndarray, float]], batch_size: int, seq_len: int,
    rng: np.random.Generator, vocab: int,
) -> torch.Tensor:
    span = seq_len + 1
    out = np.empty((batch_size, span), dtype=np.int64)
    probs = np.array([w for _, _, w in shards], dtype=np.float64)
    chosen = rng.choice(len(shards), size=batch_size, p=probs) if len(shards) > 1 else np.zeros(batch_size, dtype=np.int64)
    for i, sidx in enumerate(chosen):
        mm = shards[int(sidx)][1]
        n = int(mm.size)
        if n > span:
            st = int(rng.integers(0, n - span))
            out[i] = mm[st : st + span]
        else:
            st = int(rng.integers(0, n))
            out[i] = mm[(st + np.arange(span, dtype=np.int64)) % n]
    np.clip(out, 0, vocab - 1, out=out)
    return torch.from_numpy(out)


def build_bot_sft_mask(inp_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
    """Mask=1.0 ONLY on assistant target tokens after `<BOT>` (4) through `<EOS>` (2) before next `<USER>` (3)."""
    B, T = inp_ids.shape
    inp_np, tgt_np = inp_ids.detach().cpu().numpy(), tgt_ids.detach().cpu().numpy()
    mask = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        in_bot = False
        for t in range(T):
            ti, tt = int(inp_np[b, t]), int(tgt_np[b, t])
            if ti == TOK_BOT:
                in_bot = True
            elif ti == TOK_USER:
                in_bot = False
            if in_bot:
                mask[b, t] = 1.0
                if tt == TOK_EOS:
                    in_bot = False
        if mask[b].sum() < 1.0:
            mask[b, :] = 1.0  # Safe fallback for foundation paragraphs without chat tags
    return torch.from_numpy(mask).to(device=inp_ids.device, non_blocking=True)


@torch.no_grad()
def evaluate_val_ce(
    model: GenesisV9V41Native, val_mm: np.ndarray, batch_size: int, seq_len: int,
    n_batches: int, device: str, use_amp: bool, recur_steps: int,
) -> float:
    was_train = model.training
    model.eval()
    rng = np.random.default_rng(1337)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    sh = [("val", val_mm, 1.0)]
    losses: list[float] = []
    for _ in range(max(1, n_batches)):
        bt = sample_mix_batch(sh, batch_size, seq_len, rng, model.cfg.vocab).to(device)
        with amp_ctx:
            logits, _ = model(bt[:, :-1], recur_steps=recur_steps)
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), bt[:, 1:].reshape(-1))
        losses.append(float(ce.item()))
    if was_train:
        model.train()
    return float(sum(losses) / max(1, len(losses)))


def load_warmstart_weights(model: GenesisV9V41Native, init_from: Path, device: str) -> dict[str, Any]:
    ck = torch.load(init_from, map_location="cpu", weights_only=False)
    sd = ck.get("model_state", ck)
    cur = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k in cur and tuple(v.shape) == tuple(cur[k].shape):
            matched[k] = v.to(device=device, dtype=cur[k].dtype)
    missing, unexpected = model.load_state_dict(matched, strict=False)
    return {"init_from": str(init_from), "loaded_tensors": len(matched), "missing_new_heads": len(missing), "unexpected": len(unexpected)}


def save_ckpt_atomic(path: Path, model: GenesisV9V41Native, opt: MuonHybridOptimizer, meta: dict[str, Any], save_opt: bool = True) -> None:
    _guard_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    payload: dict[str, Any] = {
        "format": "genesis_v9_ultra_ckpt_v2", "config": asdict(model.cfg), "fsp_horizon": model.fsp_horizon,
        "model_state": {k: (v.detach().to(torch.bfloat16).cpu() if v.is_floating_point() else v.detach().cpu()) for k, v in model.state_dict().items()},
        **meta,
    }
    if save_opt:
        payload["opt_state"] = opt.state_dict()
    torch.save(payload, tmp)
    os.replace(tmp, path)


def export_inference_bf16(model: GenesisV9V41Native, out_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    _guard_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    was_train = model.training
    model.eval()
    dev = next(model.parameters()).device
    px = torch.arange(16, min(model.cfg.vocab, 48), device=dev, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        lo0, _ = model(px, recur_steps=model.cfg.recur_steps)
        model.strip_training_heads()
        lo1, _ = model(px, recur_steps=model.cfg.recur_steps)
    diff = float((lo0 - lo1).abs().max().item())
    if diff != 0.0:
        raise AssertionError(f"strip_training_heads changed logits: {diff}")
    sd = {k: (v.detach().to(torch.bfloat16).cpu() if v.is_floating_point() else v.detach().cpu()) for k, v in model.state_dict().items()}
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save({
        "format": "genesis_v9_v41_inference_v1", "config": asdict(model.cfg), "stripped_training_heads": True,
        "dtype": "bfloat16", "param_count_inference": sum(p.numel() for p in model.parameters()),
        "strip_max_abs_logit_diff": diff, "model_state": sd, **meta,
    }, tmp)
    os.replace(tmp, out_path)
    if was_train:
        model.train()
    return {"path": str(out_path), "strip_diff": diff, "mb": round(out_path.stat().st_size / 1e6, 2)}


def run_ultra_training(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = _guard_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dev_label, est_cu = estimate_ultra_cu_rate(device)
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if use_amp else 0.0
    prof = recommend_gpu_profile(dev_label, vram_gb) if use_amp else None
    bs, sl, ga = args.batch_size, args.seq_len, max(1, args.grad_accum)
    delta_chunk, compile_mode = 16, "default"
    if prof is not None:
        delta_chunk, compile_mode = prof.delta_chunk, prof.compile_mode
        if args.auto_profile:
            bs, ga = prof.batch_size, prof.grad_accum
            if sl >= 2048:
                bs, ga = max(2, bs // 2), ga * 2
    cu_rate = args.cu_rate if args.cu_rate > 0.0 else est_cu
    shards = parse_mix_spec(args.mix_spec, args.data_dir)
    val_p = args.val_shard if (args.val_shard and args.val_shard.is_file()) else (args.data_dir / "val.bin")
    val_mm = np.memmap(val_p, dtype="<u2", mode="r") if val_p.is_file() else shards[0][1]

    cfg = B200_PRESETS[args.preset]
    if prof is None:
        delta_chunk = cfg.delta_chunk
    model = GenesisV9V41Native(cfg, fsp_horizon=16, include_train_heads=True).to(device)
    tok_used = ""
    if cfg.vocab == 16384:
        for tp in (args.tokenizer, args.data_dir / "v9_tok_16k.json", DEFAULT_V9_TOK_PATH):
            if tp is not None and tp.is_file():
                model.set_canonical_lut(load_v9_canon_lut_from_tokenizer(tp, cfg.vocab))
                tok_used = str(tp)
                break
    kcounts = apply_fast_b200_kernels(model, delta_chunk=delta_chunk) if use_amp else {"cpu_ref": 1}

    init_info: dict[str, Any] = {}
    opt = build_fast_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, sparse_lr_mult=1.5)
    start_step = cum_tokens = 0
    best_val, cu_prior = float("inf"), 0.0
    latest_ckpt = out_dir / "latest_ckpt.pt"
    if not args.no_resume and latest_ckpt.is_file():
        ck = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        cur = model.state_dict()
        model.load_state_dict({k: v.to(device=device, dtype=cur[k].dtype) for k, v in ck["model_state"].items()}, strict=True)
        if "opt_state" in ck:
            opt.load_state_dict(ck["opt_state"])
        start_step = int(ck.get("step", 0))
        cum_tokens = int(ck.get("cum_tokens", 0))
        best_val = float(ck.get("best_val", float("inf")))
        cu_prior = float(ck.get("cu_used", 0.0))
        init_info = {"resumed_from": str(latest_ckpt), "step": start_step}
    elif args.init_from is not None and args.init_from.is_file():
        init_info = load_warmstart_weights(model, args.init_from, device)

    total_steps = max(int(args.steps), start_step)
    min_lr_ratio = max(0.0, min(1.0, args.lr_min / max(args.lr, 1e-12)))
    wsd = WSDScheduler(max(1, total_steps), lr_max=args.lr,
                       warmup_steps=max(1, min(200, max(2, total_steps // 20))),
                       decay_ratio=args.decay_ratio, min_lr_ratio=min_lr_ratio)
    if use_amp and not args.no_compile:
        train_mod, compiled_ok = compile_v9_fast(model, mode=compile_mode, allow_cudagraphs=False)
    else:
        train_mod, compiled_ok = model, False

    toks_per_step = bs * sl * ga
    r1_until = int(total_steps * max(0.0, min(1.0, args.r1_warmup_frac)))
    eval_every = args.eval_every if args.eval_every > 0 else max(1, min(200, max(5, total_steps // 10)))
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    val_before = last_val = evaluate_val_ce(model, val_mm, min(bs, 4), sl, 2, device, use_amp, args.recur_steps)
    log_path, t0 = out_dir / "train_telemetry.jsonl", time.perf_counter()
    last_save_t, cu_used, final_step, stopped_cu = t0, cu_prior, start_step, False
    print(json.dumps({"event": "start", "device": dev_label, "cu_rate": cu_rate, "compiled": compiled_ok,
                      "bs": bs, "ga": ga, "sl": sl, "delta_chunk": delta_chunk, "kernels": kcounts, "init": init_info,
                      "shards": [(n, int(m.size), round(w, 3)) for n, m, w in shards],
                      "val_before": round(val_before, 4), "r1_until_step": r1_until}, ensure_ascii=False), flush=True)

    model.train()
    loss_acc = ZeroSyncLossAccumulator(device=device)
    with AsyncMixPrefetcher(shards, bs, sl, cfg.vocab, device=device, seed=args.seed + start_step * 9973,
                            sft_mask_bot=args.sft_mask_bot, queue_size=4) as prefetcher:
        for step in range(start_step + 1, total_steps + 1):
            st_t0 = time.perf_counter()
            r_now = 1 if (step <= r1_until and args.recur_steps > 1) else args.recur_steps
            phase, cur_lr = wsd.phase_at(step), wsd.step_optimizer(opt, step)
            opt.zero_grad(set_to_none=True)
            loss_acc.reset()
            for _ in range(ga):
                inp, tgt, tmask = prefetcher.next_batch()
                with amp_ctx:
                    out = train_mod(inp, recur_steps=r_now, targets=tgt, return_details=True)
                    loss, lstats = compute_v9_v41_loss_nosync(model, out, tgt, token_mask=tmask)
                    (loss / float(ga)).backward()
                loss_acc.update(lstats)
            gnorm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            opt.step()
            metrics = loss_acc.finalize(grad_norm=gnorm_t)
            gnorm = float(metrics.pop("gnorm", 0.0))

            cum_tokens += toks_per_step
            final_step = step
            elapsed_s = time.perf_counter() - t0
            cu_used = cu_prior + (elapsed_s / 3600.0) * cu_rate
            tok_s = toks_per_step / max(1e-6, time.perf_counter() - st_t0)
            need_eval = (step % eval_every == 0) or (step == total_steps)
            if need_eval:
                last_val = evaluate_val_ce(model, val_mm, min(bs, 4), sl, 2, device, use_amp, args.recur_steps)
            rec = {
                "ts": int(time.time()), "step": step, "r": r_now, "phase": phase, "lr": round(cur_lr, 7),
                **{k: round(float(v), 4) for k, v in metrics.items()}, "gnorm": round(gnorm, 3),
                "val_ce": round(last_val, 4), "tok_s": round(tok_s, 1), "cum_tokens": cum_tokens, "cu_used": round(cu_used, 3),
            }
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(json.dumps(rec) + "\n")
            if step == start_step + 1 or need_eval or step % max(1, total_steps // 20) == 0:
                print(f"[v9-ultra] step={step}/{total_steps} R={r_now} ({phase}) lr={cur_lr:.2e} "
                      f"ce={metrics['final_ce']:.4f} val={last_val:.4f} tok/s={tok_s:.0f} Mtok={cum_tokens/1e6:.1f} cu={cu_used:.2f}/{args.max_cu:.1f}", flush=True)

            hit_cu = cu_rate > 0.0 and cu_used >= args.max_cu
            need_save = (time.perf_counter() - last_save_t) >= max(0.05, args.save_every_mins) * 60.0
            meta = {"preset": args.preset, "step": step, "cum_tokens": cum_tokens, "best_val": min(best_val, last_val), "cu_used": cu_used}
            if need_eval and last_val < best_val:
                best_val = last_val
                save_ckpt_atomic(out_dir / "best_weights_bf16.pt", model, opt, meta, save_opt=False)
            if need_save or step == total_steps or hit_cu:
                save_ckpt_atomic(latest_ckpt, model, opt, meta, save_opt=True)
                last_save_t = time.perf_counter()
            if hit_cu:
                stopped_cu = True
                break

    val_after = evaluate_val_ce(model, val_mm, min(bs, 4), sl, 2, device, use_amp, args.recur_steps)
    exp_meta = {} if args.no_export else export_inference_bf16(
        model, out_dir / "genesis_v9_1_5b_ultra_inference_bf16.pt",
        meta={"preset": args.preset, "step": final_step, "cum_tokens": cum_tokens, "val_ce": round(val_after, 4)},
    )
    summary = {
        "preset": args.preset, "device": dev_label, "cu_rate": cu_rate, "cu_used": round(cu_used, 3),
        "stopped_on_cu": stopped_cu, "final_step": final_step, "cum_tokens": cum_tokens,
        "val_before": round(val_before, 4), "val_after": round(val_after, 4), "best_val": round(min(best_val, val_after), 4),
        "tokenizer": tok_used, "init": init_info, "inference_export": exp_meta,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Genesis-v9 1.5B Colab Ultra 1000-CU Production Trainer")
    ap.add_argument("--preset", default="v9_1_5b_b200", choices=list(B200_PRESETS.keys()))
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--mix-spec", default="", help="Comma list of shard.bin:weight")
    ap.add_argument("--val-shard", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--decay-ratio", type=float, default=0.15)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--recur-steps", type=int, default=2)
    ap.add_argument("--r1-warmup-frac", type=float, default=0.0)
    ap.add_argument("--sft-mask-bot", action="store_true")
    ap.add_argument("--save-every-mins", type=float, default=25.0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--max-cu", type=float, default=550.0)
    ap.add_argument("--cu-rate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--auto-profile", action="store_true", help="Auto-pick optimal bs/ga/delta_chunk for G4/H100/A100")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    print(json.dumps(run_ultra_training(ap.parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

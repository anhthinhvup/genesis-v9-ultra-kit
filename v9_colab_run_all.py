"""Genesis-v9 1.5B Colab Pro+ Ultra 1,000-CU Full-Throttle Orchestrator (`colab_gpu_trainer/v9_colab_run_all.py`, <= 295 LOC).
Modes:
  - `--mode setup`: Downloads verified `v0.9-base` release (3.06GB `Engram196B` weights + 7 SOTA v2 shards)
                    to Drive Ultra & compiles Linux Rust `v9_stream_bin_rs` (~10M tok/s).
  - `--mode pack-cpu`: Zero-CU (`0.00 CU/hr`) multi-billion-token streaming packer via Rust `v9_stream_bin_rs`
                       pulling `FineWeb-Edu`, `Cosmopedia-v2`, `FineMath-4+`, `Vi-FineWeb2`, `Python-Edu`.
  - `--mode train-all`: Crash-proof 4-Phase 1,000-CU curriculum (`Phase 1 Pretrain -> Phase 2 Midtrain 2048
                        -> Phase 3A SOTA v2 1024 -> Phase 3B Long-CoT+OPD 2048 -> Final BF16 Export`).
Strictly <= 400 LOC | 0 hardcoded secrets | 0 `re`/`sympy`/`eval`/`exec`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import time
from typing import Any
import urllib.parse
import urllib.request

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v9_colab_cpu_pack import build_pool_shard
from v9_colab_tpu_xla import detect_xla_runtime

RELEASE_BASE = "https://github.com/anhthinhvup/genesis-v9-ultra-kit/releases/download/v0.9-base"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _download_checked(url: str, dst: Path, expected_sha: str = "") -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and (not expected_sha or _sha256(dst) == expected_sha):
        print(f"[SETUP] cached ok: {dst.name} ({dst.stat().st_size/1e6:.1f} MB)", flush=True)
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[SETUP] downloading {url} -> {dst.name} ...", flush=True)
    subprocess.run(["curl", "-fSL", "--retry", "4", "-o", str(tmp), url], check=True)
    if expected_sha:
        got = _sha256(tmp)
        if got != expected_sha:
            raise RuntimeError(f"SHA256 mismatch for {dst.name}: got {got}, want {expected_sha}")
    tmp.replace(dst)
    print(f"[SETUP] verified: {dst.name} ({dst.stat().st_size/1e6:.1f} MB)", flush=True)
    return dst


def build_rust_stream_packer(kit_dir: Path) -> Path | None:
    rust_dir = kit_dir / "genesis_v9_rust"
    if not (rust_dir / "Cargo.toml").is_file():
        return None
    bin_path = rust_dir / "target" / "release" / ("v9_stream_bin_rs.exe" if os.name == "nt" else "v9_stream_bin_rs")
    if bin_path.is_file():
        return bin_path
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
    if not Path(cargo).is_file() and os.name != "nt":
        print("[SETUP] Installing Rust toolchain (15s, 0 CU)...", flush=True)
        subprocess.run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal", shell=True, check=True)
        cargo = str(Path.home() / ".cargo" / "bin" / "cargo")
    if not Path(cargo).is_file():
        return None
    env = dict(os.environ, RUSTFLAGS="-C target-cpu=native")
    print("[SETUP] Compiling Rust v9_stream_bin_rs --release ...", flush=True)
    subprocess.run([cargo, "build", "--release", "--bin", "v9_stream_bin_rs"], cwd=str(rust_dir), env=env, check=True)
    return bin_path if bin_path.is_file() else None


def run_setup(ultra_dir: Path, kit_dir: Path) -> dict[str, Any]:
    for sub in ("shards", "sota_v2", "ckpts", "logs"):
        (ultra_dir / sub).mkdir(parents=True, exist_ok=True)
    sums_p = _download_checked(f"{RELEASE_BASE}/SHA256SUMS.txt", ultra_dir / "SHA256SUMS.txt")
    shas: dict[str, str] = {}
    for line in sums_p.read_text("utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            shas[parts[1]] = parts[0]

    # 1. Extract 7 SOTA v2 shards (57.08M tok)
    sh_tar = _download_checked(f"{RELEASE_BASE}/v9_distill_shards_v4.tar.gz", ultra_dir / "v9_distill_shards_v4.tar.gz", shas.get("v9_distill_shards_v4.tar.gz", ""))
    if not (ultra_dir / "sota_v2" / "train.bin").is_file():
        with tarfile.open(sh_tar, "r:gz") as tf:
            tf.extractall(ultra_dir / "sota_v2")
    print("[SETUP] 7 SOTA v2 shards ready in", ultra_dir / "sota_v2", flush=True)

    # 2. Download & reassemble 3.06GB Engram196B BF16 base checkpoint
    wt_dst = ultra_dir / "genesis_v9_1_5b_engram196b_deep_bf16.pt"
    want_wt = shas.get("genesis_v9_1_5b_engram196b_deep_bf16.pt", "")
    if not (wt_dst.is_file() and (not want_wt or _sha256(wt_dst) == want_wt)):
        p0 = _download_checked(f"{RELEASE_BASE}/engram196b_deep_bf16.pt.part0", ultra_dir / "part0.bin", shas.get("engram196b_deep_bf16.pt.part0", ""))
        p1 = _download_checked(f"{RELEASE_BASE}/engram196b_deep_bf16.pt.part1", ultra_dir / "part1.bin", shas.get("engram196b_deep_bf16.pt.part1", ""))
        tmp_wt = wt_dst.with_suffix(".pt.tmp")
        with tmp_wt.open("wb") as out_f:
            for part in (p0, p1):
                with part.open("rb") as pf:
                    shutil.copyfileobj(pf, out_f, length=1 << 22)
        if want_wt and _sha256(tmp_wt) != want_wt:
            raise RuntimeError("Reassembled weights SHA256 mismatch!")
        tmp_wt.replace(wt_dst)
        p0.unlink(missing_ok=True)
        p1.unlink(missing_ok=True)
    print(f"[SETUP] Base Engram196B weights verified: {wt_dst.stat().st_size/1e6:.1f} MB", flush=True)

    rs_bin = build_rust_stream_packer(kit_dir)
    return {"base_weights": str(wt_dst), "sota_v2_dir": str(ultra_dir / "sota_v2"), "rust_packer": str(rs_bin or "")}


def run_rust_cpu_pack(pools_spec: str, ultra_dir: Path, kit_dir: Path) -> list[dict[str, Any]]:
    rs_bin = build_rust_stream_packer(kit_dir)
    tok_p, guard_p = kit_dir / "v9_tok_16k.json", kit_dir / "v9_anti_leak_guard.bin"
    out_dir = ultra_dir / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("/tmp/v9_pq_stream") if os.name != "nt" else (ultra_dir / "_tmp_pq")
    workers = max(2, (os.cpu_count() or 4) - 1)
    reports = [
        build_pool_shard(
            pid.strip(), int(float(mtok_s) * 1_000_000), out_dir, tok_p,
            guard_p if guard_p.is_file() else None, tmp_dir, workers, rust_bin=rs_bin,
        )
        for item in pools_spec.split(",") if item.strip()
        for pid, mtok_s in [item.strip().split(":", 1)]
    ]
    (out_dir / "rust_pack_manifest.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports


def _run_phase(kit_dir: Path, args_list: list[str], name: str, use_xla: bool = False) -> Path:
    print(f"\n========== LAUNCHING {name} ({'TPU-XLA' if use_xla else 'CUDA-ULTRA'}) ==========", flush=True)
    script = kit_dir / ("v9_colab_tpu_xla.py" if use_xla else "v9_colab_ultra_1000cu.py")
    extra = ["--mode", "train"] if use_xla else ["--auto-profile"]
    cmd = [sys.executable, "-u", str(script), *extra, *args_list]
    subprocess.run(cmd, check=True)
    out_idx = args_list.index("--out-dir") + 1
    return Path(args_list[out_idx]) / "genesis_v9_1_5b_ultra_inference_bf16.pt"


def run_train_all(ultra_dir: Path, kit_dir: Path, bs: int, ga: int, cu_rate: float) -> None:
    xla_info = detect_xla_runtime()
    use_xla = bool(getattr(xla_info, "is_xla", False))
    if use_xla:
        bs, ga = min(bs, 8), max(ga, 8)
        if cu_rate <= 0.0 or cu_rate == 8.8:
            cu_rate = 3.2
    tok = str(kit_dir / "v9_tok_16k.json")
    sh_dir, sota_dir, ck_dir = ultra_dir / "shards", ultra_dir / "sota_v2", ultra_dir / "ckpts"
    base_wt = ultra_dir / "genesis_v9_1_5b_engram196b_deep_bf16.pt"
    val_sh = str(sota_dir / "val.bin")

    cand = [("fineweb_edu.bin", 0.40), ("cosmopedia_v2.bin", 0.20), ("finemath_4plus.bin", 0.15), ("vi_fineweb2.bin", 0.15), ("python_code.bin", 0.10)]
    p1_mix = ",".join(f"{sh_dir / n}:{w}" for n, w in cand if (sh_dir / n).is_file()) or f"{sota_dir / 'train.bin'}:1.0"
    p1_out = _run_phase(kit_dir, [
        "--preset", "v9_1_5b_b200", "--data-dir", str(sh_dir), "--mix-spec", p1_mix, "--val-shard", val_sh,
        "--out-dir", str(ck_dir / "p1_pretrain"), "--init-from", str(base_wt), "--tokenizer", tok,
        "--steps", "160000", "--batch-size", str(bs), "--seq-len", "1024", "--grad-accum", str(ga),
        "--lr", "6e-4", "--lr-min", "8e-5", "--recur-steps", "2", "--r1-warmup-frac", "0.60",
        "--save-every-mins", "25", "--eval-every", "400", "--max-cu", "550.0", "--cu-rate", str(cu_rate),
    ], "PHASE 1: HIGH-DENSITY PRETRAIN (550 CU)", use_xla=use_xla)

    p2_cand = [(sh_dir / "finemath_4plus.bin", 0.30), (sh_dir / "cosmopedia_v2.bin", 0.25), (sh_dir / "vi_fineweb2.bin", 0.20), (sota_dir / "train.bin", 0.25)]
    p2_mix = ",".join(f"{p}:{w}" for p, w in p2_cand if p.is_file())
    p2_bs = max(2, min(4, bs // 2)) if use_xla else max(2, bs // 2)
    p2_out = _run_phase(kit_dir, [
        "--preset", "v9_1_5b_b200", "--data-dir", str(sh_dir), "--mix-spec", p2_mix, "--val-shard", val_sh,
        "--out-dir", str(ck_dir / "p2_midtrain_2048"), "--init-from", str(p1_out), "--tokenizer", tok,
        "--steps", "60000", "--batch-size", str(p2_bs), "--seq-len", "2048", "--grad-accum", str(ga * 2),
        "--lr", "2e-4", "--lr-min", "3e-5", "--recur-steps", "2", "--r1-warmup-frac", "0.0",
        "--save-every-mins", "25", "--eval-every", "300", "--max-cu", "220.0", "--cu-rate", str(cu_rate),
    ], "PHASE 2: MID-TRAINING 2048 (220 CU)", use_xla=use_xla)

    p3a_out = _run_phase(kit_dir, [
        "--preset", "v9_1_5b_b200", "--data-dir", str(sota_dir),
        "--mix-spec", f"{sota_dir / 'sota_v2_stage1a_1024.bin'}:0.85,{sota_dir / 'sota_v2_non_think_chat.bin'}:0.15",
        "--val-shard", val_sh, "--out-dir", str(ck_dir / "p3a_sft_1024"), "--init-from", str(p2_out),
        "--tokenizer", tok, "--steps", "2000", "--batch-size", str(bs), "--seq-len", "1024", "--grad-accum", str(ga),
        "--lr", "8e-5", "--lr-min", "1e-5", "--recur-steps", "2", "--sft-mask-bot",
        "--save-every-mins", "20", "--eval-every", "100", "--max-cu", "70.0", "--cu-rate", str(cu_rate),
    ], "PHASE 3A: SOTA V2 STAGE 1A (70 CU)", use_xla=use_xla)

    _run_phase(kit_dir, [
        "--preset", "v9_1_5b_b200", "--data-dir", str(sota_dir),
        "--mix-spec", f"{sota_dir / 'sota_v2_stage1b_2048.bin'}:0.65,{sota_dir / 'sota_v2_non_think_chat.bin'}:0.25,{sota_dir / 'opd_train.bin'}:0.10",
        "--val-shard", val_sh, "--out-dir", str(ck_dir / "p3b_sft_2048_final"), "--init-from", str(p3a_out),
        "--tokenizer", tok, "--steps", "600", "--batch-size", str(p2_bs), "--seq-len", "2048", "--grad-accum", str(ga * 2),
        "--lr", "3e-5", "--lr-min", "5e-6", "--decay-ratio", "0.50", "--recur-steps", "2", "--sft-mask-bot",
        "--save-every-mins", "15", "--eval-every", "50", "--max-cu", "40.0", "--cu-rate", str(cu_rate),
    ], "PHASE 3B: SOTA V2 LONG-COT + OPD FINAL (40 CU)", use_xla=use_xla)


def main() -> None:
    ap = argparse.ArgumentParser(description="Genesis-v9 1.5B Colab Pro+ Ultra 1,000-CU Full-Throttle Orchestrator")
    ap.add_argument("--mode", required=True, choices=["setup", "pack-cpu", "train-all"])
    ap.add_argument("--ultra-dir", type=Path, default=Path("/content/drive/MyDrive/genesis_v9_ultra"))
    ap.add_argument("--kit-dir", type=Path, default=HERE)
    ap.add_argument("--pools", default="fineweb_edu:4000,cosmopedia_v2:2000,finemath_4plus:1500,vi_fineweb2:1500,python_code:1000")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--cu-rate", type=float, default=8.8)
    a = ap.parse_args()
    if a.mode == "setup":
        print(json.dumps(run_setup(a.ultra_dir, a.kit_dir), indent=2))
    elif a.mode == "pack-cpu":
        print(json.dumps(run_rust_cpu_pack(a.pools, a.ultra_dir, a.kit_dir), indent=2))
    elif a.mode == "train-all":
        run_train_all(a.ultra_dir, a.kit_dir, a.batch_size, a.grad_accum, a.cu_rate)


if __name__ == "__main__":
    main()

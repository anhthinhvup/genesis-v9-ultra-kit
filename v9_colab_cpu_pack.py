"""Zero-CU Colab CPU Multi-Billion-Token `v9_tok_16k` Shard Builder (`colab_gpu_trainer/v9_colab_cpu_pack.py`, <= 400 LOC).
Pulls 100% open-license ungated (`ODC-BY / CC-BY-SA-3.0`) Parquet shards from Hugging Face via live
Hub Tree API + `datasets-server` fallback, streams row-groups in constant RAM, applies CJK/length
hygiene + exact `GV9GRD01` anti-leak filtering, and packs `<u2` (`uint16`) shards via Rust
`v9_stream_bin_rs` (~6M-12M tok/s) or multiprocessing Python fallback.
Strictly <= 400 LOC | 0 hardcoded secrets | 0 `re`/`sympy`/`eval`/`exec`.
"""
from __future__ import annotations

import argparse, concurrent.futures, hashlib, io, json, os, struct, subprocess, time
from pathlib import Path
from typing import Any, Iterator
import urllib.error, urllib.parse, urllib.request

import numpy as np
import pyarrow.parquet as pq

UA = "genesis-v9-ultra-cpu/2.0"
_PUNCT = frozenset(".,!?;:-'\"()[]{}<>/*+=#@&$%^~`|\\")

# 100% Open-License Ungated Direct-Parquet Pretrain & Midtrain Pools (live-verified HTTP 200)
ULTRA_POOLS: dict[str, dict[str, Any]] = {
    "fineweb_edu": {"repo": "HuggingFaceFW/fineweb-edu", "cfg": "sample-10BT", "tree_path": "sample/10BT",
                    "col": "text", "license": "ODC-BY", "lang": "en", "min_chars": 240, "max_chars": 16000},
    "cosmopedia_v2": {"repo": "HuggingFaceTB/smollm-corpus", "cfg": "cosmopedia-v2", "tree_path": "cosmopedia-v2",
                      "col": "text", "license": "ODC-BY", "lang": "en", "min_chars": 240, "max_chars": 16000},
    "finemath_4plus": {"repo": "HuggingFaceTB/finemath", "cfg": "finemath-4plus", "tree_path": "finemath-4plus",
                       "col": "text", "license": "ODC-BY", "lang": "en", "min_chars": 160, "max_chars": 14000},
    "vi_fineweb2": {"repo": "HuggingFaceFW/fineweb-2", "cfg": "vie_Latn", "tree_path": "data/vie_Latn/train",
                    "col": "text", "license": "ODC-BY", "lang": "vi", "min_chars": 200, "max_chars": 14000},
    "vi_wiki": {"repo": "wikimedia/wikipedia", "cfg": "20231101.vi", "tree_path": "20231101.vi",
                "col": "text", "license": "CC-BY-SA-3.0", "lang": "vi", "min_chars": 240, "max_chars": 14000},
    "python_code": {"repo": "Avelina/python-edu-cleaned", "cfg": "default", "tree_path": "data",
                    "col": "text", "license": "ODC-BY", "lang": "en", "min_chars": 120, "max_chars": 14000},
}

_G_CODEC: tuple[dict[bytes, int], dict[bytes, int], int] | None = None
_G_GUARD: tuple[set[bytes], set[int], int] = (set(), set(), 8)


def _fnv64_words(words: list[str], start: int, n: int) -> int:
    h = 0xCBF29CE484222325
    for k in range(n):
        if k:
            h = ((h ^ 32) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        for b in words[start + k].encode("utf-8", errors="ignore"):
            h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _tokenize_clean_words(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for ch in text:
        if ch in _PUNCT or ch.isspace():
            if cur: out.append("".join(cur)); cur.clear()
        else:
            cur.append(ch.lower())
    if cur: out.append("".join(cur))
    return out


def _load_guard(guard_path: Path | None) -> tuple[set[bytes], set[int], int]:
    if guard_path is None or not guard_path.is_file(): return set(), set(), 8
    raw = guard_path.read_bytes()
    if len(raw) < 20 or raw[:8] != b"GV9GRD01": raise ValueError(f"Invalid GV9GRD01 guard in {guard_path}")
    n_sha, n_sh, sh_n = struct.unpack_from("<III", raw, 8)
    if len(raw) < 20 + n_sha * 32 + n_sh * 8: raise ValueError(f"Truncated GV9GRD01 guard {guard_path}")
    shas = {raw[20 + i * 32 : 20 + (i + 1) * 32] for i in range(n_sha)}
    shingles = set(struct.unpack_from(f"<{n_sh}Q", raw, 20 + n_sha * 32)) if n_sh else set()
    return shas, shingles, max(1, sh_n)


def _has_cjk(text: str) -> bool:
    return any(0x4E00 <= ord(c) <= 0x9FFF or 0x3400 <= ord(c) <= 0x4DBF or 0x3040 <= ord(c) <= 0x30FF or 0xAC00 <= ord(c) <= 0xD7AF for c in text)


def _is_leaked(text: str, guard: tuple[set[bytes], set[int], int]) -> bool:
    shas, shingles, sh_n = guard
    s = text.strip()
    if (not shas and not shingles) or len(s) < 8:
        return False
    if hashlib.sha256(" ".join(s.split()).lower().encode("utf-8", errors="ignore")).digest() in shas:
        return True
    if not shingles:
        return False
    words, hits = _tokenize_clean_words(s), 0
    for i in range(max(0, len(words) - sh_n + 1)):
        if _fnv64_words(words, i, sh_n) in shingles:
            hits += 1
            if hits >= 2:
                return True
    return False


def _init_worker(tok_path_str: str, guard_path_str: str) -> None:
    global _G_CODEC, _G_GUARD
    obj = json.loads(Path(tok_path_str).read_text(encoding="utf-8"))
    p2i: dict[bytes, int] = {bytes([b]): 16 + b for b in range(256)}
    for idx, (h, _, _) in enumerate(obj["pieces"]):
        p2i[bytes.fromhex(h)] = 272 + idx
    sp_map = {s.encode("utf-8"): i for i, s in enumerate(obj.get("special_tokens", []))}
    _G_CODEC = (p2i, sp_map, max((len(k) for k in p2i), default=24))
    _G_GUARD = _load_guard(Path(guard_path_str) if guard_path_str else None)


def _encode_chunk(args: tuple[list[str], int, int]) -> tuple[bytes, int, int]:
    texts, min_c, max_c = args
    assert _G_CODEC is not None
    p2i, sp_map, max_l = _G_CODEC
    out_ids: list[int] = []
    kept = leaks = 0
    for txt in texts:
        if not txt or len(txt) < min_c:
            continue
        t = txt[:max_c].strip()
        if len(t) < min_c or _has_cjk(t):
            continue
        if _is_leaked(t, _G_GUARD):
            leaks += 1; continue
        raw = t.encode("utf-8", errors="ignore")
        ids, i, n = [1], 0, len(raw)
        while i < n:
            if raw[i] == 60:
                gt = raw.find(62, i + 1, min(n, i + 18))
                if gt != -1 and (stid := sp_map.get(raw[i : gt + 1])) is not None:
                    ids.append(stid); i = gt + 1; continue
            for l in range(min(max_l, n - i), 0, -1):
                if (tid := p2i.get(raw[i : i + l])) is not None:
                    ids.append(tid); i += l; break
            else:
                ids.append(16 + raw[i]); i += 1
        if ids[-1] != 2:
            ids.append(2)
        out_ids.extend(ids); kept += 1
    return np.asarray(out_ids, dtype="<u2").tobytes(), kept, leaks


def _parse_next_link(link_hdr: str) -> str | None:
    for part in (link_hdr or "").split(","):
        if 'rel="next"' in part or "rel=next" in part:
            s, e = part.find("<"), part.find(">")
            if 0 <= s < e:
                return part[s + 1 : e].strip()
    return None


def _http_json(url: str, timeout: int = 30, retries: int = 4) -> tuple[Any, str]:
    last_err: Exception | None = None
    for att in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8")), r.headers.get("Link", "")
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (400, 401, 403, 404):
                raise
        except Exception as e:
            last_err = e
        if att + 1 < retries:
            time.sleep(1.5 * (2**att))
    raise RuntimeError(f"HTTP JSON fetch failed for {url}: {last_err}")


def _tree_parquet_urls(repo: str, tree_path: str, max_files: int, recursive: bool = False, cfg_hint: str = "") -> list[str]:
    base = f"https://huggingface.co/api/datasets/{repo}/tree/main"
    sub = urllib.parse.quote(tree_path.strip("/"), safe="/")
    url: str | None = (f"{base}/{sub}" if sub else base) + ("?recursive=true" if recursive else "")
    paths: list[str] = []
    while url and len(paths) < max(max_files * 2, 256):
        data, link = _http_json(url, timeout=30)
        if not isinstance(data, list):
            break
        paths.extend(str(it["path"]) for it in data if it.get("type") == "file" and str(it.get("path", "")).endswith(".parquet"))
        url = _parse_next_link(link)
    if recursive and cfg_hint and cfg_hint != "default":
        lo_a, lo_b = cfg_hint.lower(), cfg_hint.lower().replace("-", "/")
        paths = [p for p in paths if lo_a in p.lower() or lo_b in p.lower()] or paths
        paths = [p for p in paths if "train" in p.lower()] or paths
    paths.sort()
    return [f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(p, safe='/')}" for p in paths[:max_files]]


def _fetch_parquet_urls(repo: str, cfg: str, max_files: int = 64, tree_path: str = "") -> list[str]:
    if not tree_path:
        tree_path = next((str(s.get("tree_path", "")) for s in ULTRA_POOLS.values() if s["repo"] == repo and s["cfg"] == cfg), "")
    if tree_path:
        try:
            if urls := _tree_parquet_urls(repo, tree_path, max_files):
                return urls
        except Exception:
            pass
    try:
        u = f"https://datasets-server.huggingface.co/parquet?dataset={urllib.parse.quote(repo, safe='')}&config={urllib.parse.quote(cfg, safe='')}"
        data, _ = _http_json(u, timeout=30)
        raw_files = data.get("parquet_files", []) if isinstance(data, dict) else []
        files = [f for f in raw_files if f.get("split") == "train"] or raw_files
        if files:
            return [str(f["url"]) for f in files[:max_files]]
    except Exception:
        pass
    if urls := _tree_parquet_urls(repo, "", max_files, recursive=True, cfg_hint=cfg):
        return urls
    raise RuntimeError(f"No live Parquet files discovered for {repo} (cfg={cfg}, tree={tree_path})")


def _download_tmp(url: str, dst: Path, retries: int = 4) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    last_err: Exception | None = None
    for att in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=240) as resp, part.open("wb") as fh:
                while buf := resp.read(1 << 20):
                    fh.write(buf)
            part.replace(dst)
            return dst
        except Exception as e:
            last_err = e; part.unlink(missing_ok=True)
            if att + 1 < retries:
                time.sleep(2.0 * (2**att))
    raise RuntimeError(f"Download failed after {retries} attempts for {url}: {last_err}")


class _HttpRangeFile(io.RawIOBase):
    """Minimal seekable HTTP Range reader for Parquet row-group streaming."""
    def __init__(self, url: str, timeout: int = 60) -> None:
        self._url, self._timeout, self._pos = url, timeout, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA}), timeout=timeout) as r:
            self._size = int(r.headers.get("Content-Length", "0"))
        if self._size <= 0: raise RuntimeError(f"HTTP HEAD missing Content-Length for {url}")
    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def tell(self) -> int: return self._pos
    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = 0 if whence == io.SEEK_SET else (self._pos if whence == io.SEEK_CUR else self._size)
        self._pos = max(0, min(self._size, base + offset)); return self._pos
    def readinto(self, b: Any) -> int:
        if self._pos >= self._size or not len(b): return 0
        end = min(self._size - 1, self._pos + len(b) - 1)
        for att in range(4):
            try:
                req = urllib.request.Request(self._url, headers={"User-Agent": UA, "Range": f"bytes={self._pos}-{end}"})
                with urllib.request.urlopen(req, timeout=self._timeout) as r: data = r.read()
                n = len(data); b[:n] = data; self._pos += n; return n
            except Exception:
                if att == 3: raise
                time.sleep(1.5 * (2**att))
        return 0


def _iter_parquet_texts(src: Any, col: str, batch_size: int = 8192, max_rows: int = 0) -> Iterator[list[str]]:
    pf = pq.ParquetFile(src)
    try:
        if col not in pf.schema_arrow.names:
            raise ValueError(f"Parquet missing column {col!r}; available={list(pf.schema_arrow.names)}")
        seen = 0
        for rb in pf.iter_batches(batch_size=batch_size, columns=[col]):
            vals = [str(x or "") for x in rb.column(0).to_pylist()]
            if max_rows > 0 and seen + len(vals) > max_rows: vals = vals[: max_rows - seen]
            if vals: yield vals; seen += len(vals)
            if max_rows > 0 and seen >= max_rows: break
    finally:
        pf.close()


def _resolve_rust_bin(explicit: Path | None) -> Path | None:
    if explicit is not None: return explicit if explicit.is_file() else None
    exe = "v9_stream_bin_rs.exe" if os.name == "nt" else "v9_stream_bin_rs"
    here = Path(__file__).resolve().parent
    cands = [here / "genesis_v9_rust/target/release" / exe, here.parent / "genesis_v9_rust/target/release" / exe,
             here.parent / ".tmp_e/cargo_v9_target/release" / exe]
    if os.environ.get("CARGO_TARGET_DIR"): cands.insert(0, Path(os.environ["CARGO_TARGET_DIR"]) / "release" / exe)
    return next((p for p in cands if p.is_file()), None)


def _pack_batches_rust(
    rs_bin: Path, tok_path: Path, guard_path: Path | None, out_bin: Path,
    rem_tokens: int, batches: Iterator[list[str]], min_c: int, max_c: int,
) -> dict[str, Any]:
    g_arg = str(guard_path) if (guard_path and guard_path.is_file()) else "none"
    proc = subprocess.Popen([str(rs_bin), str(tok_path), g_arg, str(out_bin), str(max(1, rem_tokens))],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for batch in batches:
            buf = bytearray()
            for tx in batch:
                if tx and len(tx) >= min_c and len(s := tx[:max_c].strip()) >= min_c:
                    b = s.encode("utf-8", errors="ignore"); buf += struct.pack("<I", len(b)) + b
            if buf: proc.stdin.write(buf)
    except (BrokenPipeError, OSError): pass
    finally:
        try: proc.stdin.close()
        except OSError: pass
    out_s = proc.stdout.read().decode("utf-8", errors="replace").strip() if proc.stdout else ""
    err_s = proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
    if (rc := proc.wait()) != 0: raise RuntimeError(f"v9_stream_bin_rs failed ({rc}): {err_s}")
    return json.loads(out_s.splitlines()[-1]) if out_s else {}


def build_pool_shard(
    pool_id: str, target_tokens: int, out_dir: Path, tok_path: Path, guard_path: Path | None,
    tmp_dir: Path, workers: int, rust_bin: Path | None = None, range_stream: bool = False, max_rows_per_shard: int = 0,
) -> dict[str, Any]:
    spec = ULTRA_POOLS[pool_id]
    out_bin = out_dir / f"{pool_id}.bin"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    existing_toks = (out_bin.stat().st_size // 2) if out_bin.is_file() else 0
    if existing_toks >= target_tokens:
        return {"pool": pool_id, "status": "already_complete", "tokens": existing_toks, "path": str(out_bin)}
    urls = _fetch_parquet_urls(spec["repo"], spec["cfg"], max_files=64, tree_path=spec.get("tree_path", ""))
    rs_bin = _resolve_rust_bin(rust_bin)
    total_toks, total_docs, total_leaks, files_used = existing_toks, 0, 0, 0
    ex = None if rs_bin else concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, workers), initializer=_init_worker, initargs=(str(tok_path), str(guard_path) if guard_path else ""))
    try:
        for idx, url in enumerate(urls):
            if total_toks >= target_tokens:
                break
            pq_file = None if range_stream else _download_tmp(url, tmp_dir / f"{pool_id}_{idx:03d}.parquet")
            src: Any = io.BufferedReader(_HttpRangeFile(url), buffer_size=1 << 20) if range_stream else pq_file
            batches: Any = None
            try:
                batches = _iter_parquet_texts(src, spec["col"], max_rows=max_rows_per_shard)
                if rs_bin is not None:
                    st = _pack_batches_rust(rs_bin, tok_path, guard_path, out_bin, target_tokens - total_toks,
                                            batches, spec["min_chars"], spec["max_chars"])
                    total_toks = (out_bin.stat().st_size // 2) if out_bin.is_file() else total_toks
                    total_docs += int(st.get("docs_kept", 0)); total_leaks += int(st.get("leaks_blocked", 0))
                else:
                    assert ex is not None
                    with out_bin.open("ab") as fout:
                        for batch in batches:
                            csz = max(64, len(batch) // max(1, workers * 4))
                            jobs = [(batch[j : j + csz], spec["min_chars"], spec["max_chars"]) for j in range(0, len(batch), csz)]
                            for raw_b, kept, leaks in ex.map(_encode_chunk, jobs):
                                if raw_b:
                                    rem_b = max(0, (target_tokens - total_toks) * 2)
                                    fout.write(raw_b[:rem_b]); total_toks += min(len(raw_b), rem_b) // 2
                                total_docs += kept; total_leaks += leaks
                                if total_toks >= target_tokens:
                                    break
                            if total_toks >= target_tokens:
                                break
                        fout.flush()
            finally:
                if batches is not None and hasattr(batches, "close"): batches.close()
                if range_stream and hasattr(src, "close"): src.close()
                if pq_file is not None and pq_file.exists(): pq_file.unlink()
            files_used += 1
            dt = time.perf_counter() - t0
            print(f"[{'RUST' if rs_bin else 'PY'}-PACK] {pool_id} shard#{files_used} Mtok={total_toks/1e6:.2f}/{target_tokens/1e6:.2f} "
                  f"docs={total_docs} leaks={total_leaks} speed={(total_toks-existing_toks)/max(1e-3,dt)/1e3:.0f}k tok/s", flush=True)
    finally:
        if ex is not None: ex.shutdown(wait=True)
    return {
        "pool": pool_id, "repo": spec["repo"], "license": spec["license"], "path": str(out_bin),
        "backend": "rust" if rs_bin else "python", "files_used": files_used, "urls_discovered": len(urls),
        "docs_added": total_docs, "leaks_blocked": total_leaks, "tokens": total_toks, "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def _default_tmp_dir() -> Path:
    return Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or ("/tmp" if os.name != "nt" else ".")) / "v9_pq_stream"


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-CU Colab CPU Multi-Billion-Token v9_tok_16k Shard Builder")
    ap.add_argument("--pools", default="fineweb_edu:4000,cosmopedia_v2:2000,finemath_4plus:1500,vi_fineweb2:1500,python_code:1000")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--guard", type=Path, default=None)
    ap.add_argument("--tmp-dir", type=Path, default=_default_tmp_dir())
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--rust-bin", type=Path, default=None)
    ap.add_argument("--range-stream", action="store_true")
    ap.add_argument("--max-rows-per-shard", type=int, default=0)
    a = ap.parse_args()
    reports = [build_pool_shard(
        p.strip(), int(float(m) * 1_000_000), a.out_dir, a.tokenizer, a.guard, a.tmp_dir, a.workers,
        rust_bin=a.rust_bin, range_stream=a.range_stream, max_rows_per_shard=a.max_rows_per_shard,
    ) for item in a.pools.split(",") if item.strip() for p, m in [item.strip().split(":", 1)]]
    (a.out_dir / "cpu_pack_manifest.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

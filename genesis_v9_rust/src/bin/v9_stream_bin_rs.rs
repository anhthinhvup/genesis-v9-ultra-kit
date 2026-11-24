//! Ultra-Fast Multi-Threaded Streaming `v9_tok_16k` Binary Packer (`<= 140` LOC).
//!
//! Reads `[u32 LE byte_len][UTF-8 bytes]...` from `stdin` (or file), runs multi-threaded CJK filter +
//! 124K-shingle `FastAntiLeakGuard` + exact `V9Tok16kTrie` encoding across ALL CPU cores (`~6M-12M tok/s`),
//! and appends packed `<u2` (`uint16`) token IDs directly to `out_bin`.
//! Zero external crates | Designed for Zero-CU Colab CPU multi-billion-token shard building.

use std::fs::OpenOptions;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use genesis_v9::sha256_fast::FastAntiLeakGuard;
use genesis_v9::tok_trie::V9Tok16kTrie;

fn has_cjk(s: &str) -> bool {
    s.chars().any(|c| {
        let u = c as u32;
        (0x4E00..=0x9FFF).contains(&u)
            || (0x3400..=0x4DBF).contains(&u)
            || (0x3040..=0x30FF).contains(&u)
            || (0xAC00..=0xD7AF).contains(&u)
    })
}

fn read_batch(r: &mut impl Read, max_docs: usize, max_bytes: usize) -> Result<Vec<String>, String> {
    let mut docs = Vec::with_capacity(max_docs.min(4096));
    let mut tot_b = 0usize;
    let mut len_buf = [0u8; 4];
    while docs.len() < max_docs && tot_b < max_bytes {
        match r.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("read len: {e}")),
        }
        let n = u32::from_le_bytes(len_buf) as usize;
        if n == 0 {
            continue;
        }
        let mut buf = vec![0u8; n];
        r.read_exact(&mut buf).map_err(|e| format!("read payload: {e}"))?;
        tot_b += n;
        docs.push(String::from_utf8_lossy(&buf).into_owned());
    }
    Ok(docs)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("Usage: v9_stream_bin_rs <tok_json> <guard_bin_or_none> <out_bin> [max_tokens]".into());
    }
    let t0 = Instant::now();
    let tok = V9Tok16kTrie::load(&PathBuf::from(&args[1]))?;
    let guard = if args[2] != "none" && PathBuf::from(&args[2]).is_file() {
        Some(FastAntiLeakGuard::load(&PathBuf::from(&args[2]))?)
    } else {
        None
    };
    let out_path = PathBuf::from(&args[3]);
    if let Some(p) = out_path.parent() {
        if !p.as_os_str().is_empty() {
            std::fs::create_dir_all(p).map_err(|e| format!("mkdir: {e}"))?;
        }
    }
    let max_tokens: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(u64::MAX);
    let out_file = OpenOptions::new().create(true).append(true).open(&out_path).map_err(|e| format!("open out: {e}"))?;
    let mut writer = BufWriter::with_capacity(1 << 22, out_file);
    let stdin = io::stdin();
    let mut reader = BufReader::with_capacity(1 << 22, stdin.lock());

    let n_threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4).min(16);
    let (mut docs_in, mut docs_kept, mut leaks, mut cjk_drop, mut toks_written) = (0u64, 0u64, 0u64, 0u64, 0u64);

    loop {
        if toks_written >= max_tokens {
            break;
        }
        let batch = read_batch(&mut reader, 4096, 32 << 20)?;
        if batch.is_empty() {
            break;
        }
        docs_in += batch.len() as u64;
        let chunk = (batch.len() + n_threads - 1) / n_threads.max(1);
        let mut outs: Vec<(Vec<u16>, u64, u64, u64)> = (0..n_threads).map(|_| (Vec::new(), 0, 0, 0)).collect();
        std::thread::scope(|s| {
            for (ti, slot) in outs.iter_mut().enumerate() {
                let st = ti * chunk;
                let en = (st + chunk).min(batch.len());
                if st >= en { continue; }
                let sl = &batch[st..en];
                let tok_ref = &tok;
                let g_ref = guard.as_ref();
                s.spawn(move || {
                    let mut ids_all = Vec::with_capacity(sl.len() * 256);
                    let (mut kept, mut lk, mut cj) = (0u64, 0u64, 0u64);
                    for d in sl {
                        if has_cjk(d) { cj += 1; continue; }
                        if let Some(g) = g_ref {
                            if !g.is_clean(d) { lk += 1; continue; }
                        }
                        let ids = tok_ref.encode(d, true, true);
                        ids_all.extend_from_slice(&ids);
                        kept += 1;
                    }
                    *slot = (ids_all, kept, lk, cj);
                });
            }
        });
        for (ids, kept, lk, cj) in outs {
            docs_kept += kept;
            leaks += lk;
            cjk_drop += cj;
            if ids.is_empty() { continue; }
            let rem = max_tokens.saturating_sub(toks_written) as usize;
            let take = ids.len().min(rem);
            let raw = unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const u8, take * 2) };
            writer.write_all(raw).map_err(|e| format!("write: {e}"))?;
            toks_written += take as u64;
            if toks_written >= max_tokens { break; }
        }
    }
    writer.flush().map_err(|e| format!("flush: {e}"))?;
    let dt = t0.elapsed().as_secs_f64().max(1e-6);
    println!(
        "{{\"docs_in\":{docs_in},\"docs_kept\":{docs_kept},\"leaks_blocked\":{leaks},\"cjk_blocked\":{cjk_drop},\"tokens_written\":{toks_written},\"elapsed_sec\":{dt:.3},\"tok_per_sec\":{:.1}}}",
        (toks_written as f64) / dt
    );
    Ok(())
}

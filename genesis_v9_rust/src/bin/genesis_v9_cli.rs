//! Genesis-v9 1.5B Pure-Rust AVX2 CLI (`<= 190` LOC): `bench`, `eval-val`, `parity`, `generate`.

use std::path::PathBuf;
use std::time::Instant;

use genesis_v9::gv9_ffi::eval_val_tokens;
use genesis_v9::gv9_format::GenesisV9Weights;
use genesis_v9::gv9_model::{step_token, V9RuntimeState};
use genesis_v9::tok_trie::V9Tok16kTrie;

fn read_u16_file(path: &PathBuf) -> Result<Vec<u16>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let n = raw.len() / 2;
    let mut out = vec![0u16; n];
    for i in 0..n { out[i] = u16::from_le_bytes([raw[2*i], raw[2*i+1]]); }
    Ok(out)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err("Usage: genesis_v9_cli <bench|eval-val|parity|generate> <model.gv9> [args...]".into());
    }
    let cmd = args[1].as_str();
    let model_path = PathBuf::from(&args[2]);
    let t_load = Instant::now();
    let w = GenesisV9Weights::load(&model_path)?;
    let load_sec = t_load.elapsed().as_secs_f64();
    let mut st = V9RuntimeState::new(&w);

    match cmd {
        "bench" => {
            let pre_n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
            let dec_n: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
            // Warmup
            step_token(&w, &mut st, 1, 1, 1, true);
            st.reset();
            // 1. Asymmetric CED Prefill (`top_k=1, R=1`)
            let t0 = Instant::now();
            for i in 0..pre_n {
                step_token(&w, &mut st, 272 + (i % 500), 1, 1, i + 1 == pre_n);
            }
            let dt_pre_ced = t0.elapsed().as_secs_f64().max(1e-6);
            // 2. Decode R=1 (`top_k=3`)
            let t1 = Instant::now();
            let mut cur = 272usize;
            for _ in 0..dec_n {
                step_token(&w, &mut st, cur, 1, w.cfg.top_k, true);
                cur = st.logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|x| x.0).unwrap_or(2);
            }
            let dt_r1 = t1.elapsed().as_secs_f64().max(1e-6);
            // 3. Decode R=2 (`top_k=3`, full flagship depth!)
            st.reset();
            for i in 0..32 { step_token(&w, &mut st, 272 + i, 1, 1, false); }
            let t2 = Instant::now();
            cur = 304;
            for _ in 0..dec_n {
                step_token(&w, &mut st, cur, 2, w.cfg.top_k, true);
                cur = st.logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|x| x.0).unwrap_or(2);
            }
            let dt_r2 = t2.elapsed().as_secs_f64().max(1e-6);
            println!(
                "{{\"cmd\":\"bench\",\"load_sec\":{load_sec:.3},\"prefill_ced_toks\":{pre_n},\"prefill_ced_tok_s\":{:.2},\"decode_r1_tok_s\":{:.2},\"decode_r2_tok_s\":{:.2},\"fp4_kv_bytes_per_tok\":{},\"vocab\":{},\"d_model\":{}}}",
                (pre_n as f64) / dt_pre_ced, (dec_n as f64) / dt_r1, (dec_n as f64) / dt_r2,
                st.fp4_kv_bytes_per_token(), w.cfg.vocab, w.cfg.d_model
            );
        }
        "eval-val" => {
            if args.len() < 4 { return Err("Usage: eval-val <model.gv9> <val.bin> [n_seqs] [seq_len]".into()); }
            let val_path = PathBuf::from(&args[3]);
            let n_seqs: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            let seq_len: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(256);
            let val_u16 = read_u16_file(&val_path)?;
            let t1 = Instant::now();
            let (ce_r1, _, n1) = eval_val_tokens(&w, &mut st, &val_u16, n_seqs, seq_len, 1, w.cfg.top_k);
            let dt1 = t1.elapsed().as_secs_f64().max(1e-6);
            let t2 = Instant::now();
            let (ce_r2, _, n2) = eval_val_tokens(&w, &mut st, &val_u16, n_seqs, seq_len, 2, w.cfg.top_k);
            let dt2 = t2.elapsed().as_secs_f64().max(1e-6);
            println!(
                "{{\"cmd\":\"eval-val\",\"load_sec\":{load_sec:.3},\"n_seqs\":{n_seqs},\"seq_len\":{seq_len},\"tokens_scored\":{n1},\"val_R1_ce\":{ce_r1:.5},\"val_R2_ce\":{ce_r2:.5},\"delta_R2_minus_R1\":{:.5},\"r1_sec\":{dt1:.2},\"r2_sec\":{dt2:.2},\"r1_tok_s\":{:.2},\"r2_tok_s\":{:.2}}}",
                ce_r2 - ce_r1, (n1 as f64) / dt1, (n2 as f64) / dt2
            );
        }
        "parity" => {
            // Runs fixed token sequence and dumps last-token logits binary (`f32`) for exact PyTorch parity check!
            if args.len() < 5 { return Err("Usage: parity <model.gv9> <recur> <out_logits.f32> <id0,id1,...>".into()); }
            let recur: usize = args[3].parse().unwrap_or(2);
            let out_p = PathBuf::from(&args[4]);
            let ids: Vec<usize> = args[5].split(',').filter_map(|s| s.trim().parse().ok()).collect();
            st.reset();
            for (i, &tid) in ids.iter().enumerate() {
                step_token(&w, &mut st, tid, recur, w.cfg.top_k, i + 1 == ids.len());
            }
            let bytes = unsafe { std::slice::from_raw_parts(st.logits.as_ptr() as *const u8, st.logits.len() * 4) };
            std::fs::write(&out_p, bytes).map_err(|e| format!("write parity: {e}"))?;
            let top1 = st.logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
            println!("{{\"cmd\":\"parity\",\"recur\":{recur},\"n_toks\":{},\"top1_id\":{},\"top1_logit\":{:.5}}}", ids.len(), top1.0, top1.1);
        }
        "generate" => {
            if args.len() < 5 { return Err("Usage: generate <model.gv9> <v9_tok_16k.json> <prompt> [max_new] [recur]".into()); }
            let tok = V9Tok16kTrie::load(&PathBuf::from(&args[3]))?;
            let prompt = &args[4];
            let max_new: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);
            let recur: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(2);
            let p_ids = tok.encode(prompt, true, false);
            let t0 = Instant::now();
            for (i, &t) in p_ids.iter().enumerate() {
                step_token(&w, &mut st, t as usize, 1, 1, i + 1 == p_ids.len());
            }
            let dt_pre = t0.elapsed().as_secs_f64().max(1e-6);
            let mut gen = Vec::with_capacity(max_new);
            let t1 = Instant::now();
            for _ in 0..max_new {
                let nxt = st.logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|x| x.0 as u16).unwrap_or(2);
                if nxt == 2 { break; }
                gen.push(nxt);
                step_token(&w, &mut st, nxt as usize, recur, w.cfg.top_k, true);
            }
            let dt_gen = t1.elapsed().as_secs_f64().max(1e-6);
            let txt = tok.decode(&gen);
            println!(
                "{{\"cmd\":\"generate\",\"prompt_toks\":{},\"gen_toks\":{},\"prefill_tok_s\":{:.2},\"decode_tok_s\":{:.2},\"output\":{:?}}}",
                p_ids.len(), gen.len(), (p_ids.len() as f64) / dt_pre, (gen.len() as f64) / dt_gen, txt
            );
        }
        _ => return Err(format!("unknown command: {cmd}")),
    }
    Ok(())
}

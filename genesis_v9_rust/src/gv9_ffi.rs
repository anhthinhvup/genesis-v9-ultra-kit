//! Clean C-ABI FFI Exports for `genesis_v9.dll` (`<= 140` LOC, Zero External Crates).

use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::Path;

use crate::gv9_format::GenesisV9Weights;
use crate::gv9_model::{step_token, V9RuntimeState};

pub struct Gv9Handle {
    pub weights: GenesisV9Weights,
    pub state: V9RuntimeState,
}

#[no_mangle]
pub unsafe extern "C" fn gv9_load(path_utf8: *const c_char) -> *mut Gv9Handle {
    if path_utf8.is_null() { return std::ptr::null_mut(); }
    let Ok(s) = CStr::from_ptr(path_utf8).to_str() else { return std::ptr::null_mut(); };
    let Ok(weights) = GenesisV9Weights::load(Path::new(s)) else { return std::ptr::null_mut(); };
    let state = V9RuntimeState::new(&weights);
    Box::into_raw(Box::new(Gv9Handle { weights, state }))
}

#[no_mangle]
pub unsafe extern "C" fn gv9_free(h: *mut Gv9Handle) {
    if !h.is_null() { drop(Box::from_raw(h)); }
}

#[no_mangle]
pub unsafe extern "C" fn gv9_reset(h: *mut Gv9Handle) {
    if !h.is_null() { (*h).state.reset(); }
}

#[no_mangle]
pub unsafe extern "C" fn gv9_vocab_size(h: *const Gv9Handle) -> u32 {
    if h.is_null() { 0 } else { (*h).weights.cfg.vocab as u32 }
}

#[no_mangle]
pub unsafe extern "C" fn gv9_step_token(
    h: *mut Gv9Handle, tok_id: u32, recur_steps: u32, top_k: u32, compute_logits: u32, out_logits: *mut f32,
) -> i32 {
    if h.is_null() { return -1; }
    let handle = &mut *h;
    let r = if recur_steps == 0 { handle.weights.cfg.recur_steps } else { recur_steps as usize };
    let k = if top_k == 0 { handle.weights.cfg.top_k } else { top_k as usize };
    let need = compute_logits != 0;
    step_token(&handle.weights, &mut handle.state, tok_id as usize, r, k, need);
    if need && !out_logits.is_null() {
        std::ptr::copy_nonoverlapping(handle.state.logits.as_ptr(), out_logits, handle.state.logits.len());
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn gv9_prefill_ced(
    h: *mut Gv9Handle, toks: *const u16, n_toks: u32, prefill_top_k: u32, prefill_recur: u32, out_last_logits: *mut f32,
) -> i32 {
    if h.is_null() || toks.is_null() || n_toks == 0 { return -1; }
    let handle = &mut *h;
    let slice = std::slice::from_raw_parts(toks, n_toks as usize);
    let pk = if prefill_top_k == 0 { 1 } else { prefill_top_k as usize };
    let pr = if prefill_recur == 0 { 1 } else { prefill_recur as usize };
    for (i, &t) in slice.iter().enumerate() {
        let is_last = i + 1 == slice.len();
        step_token(&handle.weights, &mut handle.state, t as usize, pr, pk, is_last && !out_last_logits.is_null());
    }
    if !out_last_logits.is_null() {
        std::ptr::copy_nonoverlapping(handle.state.logits.as_ptr(), out_last_logits, handle.state.logits.len());
    }
    0
}

/// Evaluates exact cross-entropy loss (nats & bpb) over `n_seqs` slices of length `seq_len` from `u16` buffer.
pub fn eval_val_tokens(
    w: &GenesisV9Weights, st: &mut V9RuntimeState, val_u16: &[u16],
    n_seqs: usize, seq_len: usize, recur_steps: usize, top_k: usize,
) -> (f64, f64, usize) {
    let mut ce_sum = 0.0f64;
    let mut tok_cnt = 0usize;
    let max_seqs = n_seqs.min(val_u16.len() / (seq_len + 1));
    for si in 0..max_seqs {
        st.reset();
        let base = si * (seq_len + 1);
        for t in 0..seq_len {
            let x_id = val_u16[base + t] as usize;
            let y_id = val_u16[base + t + 1] as usize;
            step_token(w, st, x_id, recur_steps, top_k, true);
            let mut m = f32::NEG_INFINITY;
            for &v in &st.logits { if v > m { m = v; } }
            let mut se = 0.0f64;
            for &v in &st.logits { se += ((v - m) as f64).exp(); }
            let log_z = (m as f64) + se.ln();
            ce_sum += log_z - (st.logits[y_id.min(st.logits.len() - 1)] as f64);
            tok_cnt += 1;
        }
    }
    let ce = ce_sum / (tok_cnt.max(1) as f64);
    let bpb = ce / 2.0f64.ln() / 3.35; // approximate chars/tok or raw bits/tok below
    (ce, bpb, tok_cnt)
}

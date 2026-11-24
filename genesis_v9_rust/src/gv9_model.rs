//! Genesis-v9 1.5B Pure-Rust AVX2+FMA+TL2 Forward Engine (`<= 340` LOC, Zero External Crates).
//!
//! Exact mathematical implementation of `GenesisV9V41Native`:
//!   - `ChannelGatedDeltaNetV9` (per-channel diagonal decay) + `BitNetMoEV9` (TL2 AVX2 LUT, shared LUT across top_k!)
//!   - `HuginnSharedKVAttnBlockV9` + `FP4KVCacheV9` (E2M1 block-32, `1,440 B/tok`, Shared-KV across recursion steps)
//!   - `EngramV41Module` (2/3/4-gram NFKC hash + grouped `alpha_g` + low-rank `ctx` + streaming `Conv1d(k=4)`)
//!   - `PeerLayerV9` (Query-normed split-key top-8x8 -> top-8 product keys + AVX2 INT8 `u_exp`/`v_exp`)
//!   - TRM Dual-State Recursive Core `(z, y)` + Asymmetric CED (`prefill_top_k=1, R=1` vs `decode_top_k=3, R=2`).

use crate::gv9_format::{
    DeltaNetMoeBlock, EngramV41Weights, GenesisV9Weights, HuginnAttnBlock, PeerWeights,
};
use crate::gv9_fp4_kv::{apply_rope_pos, Fp4LayerKv};
use crate::gv9_simd::{
    axpy_i8_row, deltanet_head_step, dot_i8_row, f16_to_f32, l2_norm_inplace, lm_head_f16_parallel,
    logsigmoid_clamped, matvec_f16, relu_sq_clamp6, rms_norm, sigmoid, silu, softmax_inplace,
};
use crate::gv9_tl2::{build_tl2_lut_avx2, Tl2LutPair};

const HASH_PRIMES: [(i64, i64, i64); 8] = [
    (1000003, 9176, 17), (15485863, 32452843, 53), (49979687, 67867967, 97), (86028121, 104395301, 131),
    (2654435761, 1315423911, 2246822519), (2166136261, 16777619, 3266489917),
    (374761393, 668265263, 1274126177), (2860486313, 3267000013, 1500450271),
];

pub struct V9RuntimeState {
    pub pos: usize,
    pub kv_prelude: Fp4LayerKv, pub kv_core: Fp4LayerKv,
    pub pre_s: Vec<Vec<f32>>, pub core_s: Vec<Vec<f32>>, pub core_inited: Vec<bool>, pub coda_s: Vec<Vec<f32>>,
    pub eng_prev_cids: [i64; 3], pub eng_conv_tail: Vec<f32>,
    pub xn: Vec<f32>, pub q: Vec<f32>, pub k: Vec<f32>, pub v: Vec<f32>,
    pub la: Vec<f32>, pub beta: Vec<f32>, pub dn_out: Vec<f32>, pub tmp_d: Vec<f32>, pub ff_h: Vec<f32>,
    pub router_p: Vec<f32>, pub moe_h: Vec<f32>, pub moe_e: Vec<f32>, pub moe_acc: Vec<f32>,
    pub luts_d: Vec<Tl2LutPair>, pub luts_ff: Vec<Tl2LutPair>,
    pub eng_raw: Vec<f32>, pub eng_k: Vec<f32>, pub eng_v: Vec<f32>, pub eng_hn: Vec<f32>, pub eng_kn: Vec<f32>,
    pub eng_ctx: Vec<f32>, pub eng_gamma: Vec<f32>, pub eng_vg: Vec<f32>, pub eng_vn: Vec<f32>,
    pub logits: Vec<f32>,
}

impl V9RuntimeState {
    pub fn new(w: &GenesisV9Weights) -> Self {
        let c = &w.cfg;
        let d = c.d_model;
        let dn_hd = d / c.delta_heads;
        let attn_hd = d / c.attn_heads;
        let s_len = c.delta_heads * dn_hd * dn_hd;
        let pairs_d = (((d + 2) / 3) + 1) / 2;
        let pairs_ff = (((c.d_ff + 2) / 3) + 1) / 2;
        let eng_tot = c.engram_heads * c.engram_dim;
        let eng_rank = (d / 4).max(16);
        Self {
            pos: 0,
            kv_prelude: Fp4LayerKv::new(c.attn_heads, attn_hd, 32.min(attn_hd), c.attn_window),
            kv_core: Fp4LayerKv::new(c.attn_heads, attn_hd, 32.min(attn_hd), c.attn_window),
            pre_s: (0..c.n_prelude_dn).map(|_| vec![0.0f32; s_len]).collect(),
            core_s: (0..(c.n_core_dn * c.max_recur_steps)).map(|_| vec![0.0f32; s_len]).collect(),
            core_inited: vec![false; c.n_core_dn * c.max_recur_steps],
            coda_s: (0..c.n_coda_dn).map(|_| vec![0.0f32; s_len]).collect(),
            eng_prev_cids: [0; 3],
            eng_conv_tail: vec![0.0f32; d * (c.engram_conv_k - 1)],
            xn: vec![0.0; d], q: vec![0.0; d], k: vec![0.0; d], v: vec![0.0; d],
            la: vec![0.0; d], beta: vec![0.0; c.delta_heads], dn_out: vec![0.0; d],
            tmp_d: vec![0.0; d], ff_h: vec![0.0; c.d_attn_ff],
            router_p: vec![0.0; c.n_experts], moe_h: vec![0.0; c.d_ff],
            moe_e: vec![0.0; d], moe_acc: vec![0.0; d],
            luts_d: vec![Tl2LutPair([[0u8; 16]; 6]); pairs_d],
            luts_ff: vec![Tl2LutPair([[0u8; 16]; 6]); pairs_ff],
            eng_raw: vec![0.0; eng_tot], eng_k: vec![0.0; d], eng_v: vec![0.0; d],
            eng_hn: vec![0.0; d], eng_kn: vec![0.0; d], eng_ctx: vec![0.0; eng_rank],
            eng_gamma: vec![0.0; d], eng_vg: vec![0.0; d], eng_vn: vec![0.0; d],
            logits: vec![0.0; c.vocab],
        }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        self.kv_prelude.reset();
        self.kv_core.reset();
        for s in &mut self.pre_s { s.fill(0.0); }
        for s in &mut self.core_s { s.fill(0.0); }
        self.core_inited.fill(false);
        for s in &mut self.coda_s { s.fill(0.0); }
        self.eng_prev_cids = [0; 3];
        self.eng_conv_tail.fill(0.0);
    }

    pub fn fp4_kv_bytes_per_token(&self) -> usize {
        self.kv_prelude.bytes_per_token() + self.kv_core.bytes_per_token()
    }
}

fn topk_indices(vals: &[f32], k: usize, out: &mut [(usize, f32)]) {
    for slot in out.iter_mut().take(k) { *slot = (0, f32::NEG_INFINITY); }
    for (idx, &v) in vals.iter().enumerate() {
        if v > out[k - 1].1 {
            out[k - 1] = (idx, v);
            let mut j = k - 1;
            while j > 0 && out[j].1 > out[j - 1].1 {
                out.swap(j, j - 1);
                j -= 1;
            }
        }
    }
}

fn run_dn_moe_step(
    blk: &DeltaNetMoeBlock, dn_s: &mut [f32], x: &mut [f32], st: &mut V9RuntimeState,
    delta_heads: usize, top_k: usize,
) {
    let d = x.len();
    let hd = d / delta_heads;
    rms_norm(x, &blk.dn_norm, &mut st.xn);
    matvec_f16(&blk.wq.w, d, d, &st.xn, None, &mut st.q);
    matvec_f16(&blk.wk.w, d, d, &st.xn, None, &mut st.k);
    matvec_f16(&blk.wv.w, d, d, &st.xn, None, &mut st.v);
    matvec_f16(&blk.w_alpha.w, d, d, &st.xn, blk.w_alpha.bias.as_deref(), &mut st.la);
    matvec_f16(&blk.w_beta.w, delta_heads, d, &st.xn, blk.w_beta.bias.as_deref(), &mut st.beta);

    for h in 0..delta_heads {
        l2_norm_inplace(&mut st.q[h * hd..(h + 1) * hd]);
        l2_norm_inplace(&mut st.k[h * hd..(h + 1) * hd]);
        for j in 0..hd { st.la[h * hd + j] = logsigmoid_clamped(st.la[h * hd + j]); }
        let b_h = sigmoid(st.beta[h]);
        deltanet_head_step(
            &mut dn_s[h * hd * hd..(h + 1) * hd * hd],
            &st.q[h * hd..(h + 1) * hd], &st.k[h * hd..(h + 1) * hd],
            &st.v[h * hd..(h + 1) * hd], &st.la[h * hd..(h + 1) * hd],
            b_h, &mut st.dn_out[h * hd..(h + 1) * hd],
        );
    }
    matvec_f16(&blk.wo.w, d, d, &st.dn_out, None, &mut st.tmp_d);
    for i in 0..d { x[i] += st.tmp_d[i]; }

    // BitNet b1.58 TL2 MoE (`top_k`: 1 prefill CED or 3 decode!)
    rms_norm(x, &blk.norm2, &mut st.xn);
    let n_exp = blk.router.rows;
    matvec_f16(&blk.router.w, n_exp, d, &st.xn, None, &mut st.router_p);
    softmax_inplace(&mut st.router_p);
    let tk = top_k.clamp(1, n_exp.min(8));
    let mut top = [(0usize, 0.0f32); 8];
    topk_indices(&st.router_p, tk, &mut top);
    let w_sum: f32 = top[..tk].iter().map(|t| t.1).sum::<f32>().max(1e-6);

    st.moe_acc.fill(0.0);
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    let mut inv_sc_in = 0.0f64;
    if use_avx2 {
        #[cfg(target_arch = "x86_64")]
        unsafe { inv_sc_in = build_tl2_lut_avx2(&st.xn, d, st.luts_d.len(), st.luts_d.as_mut_ptr()); }
    }
    let d_ff = st.moe_h.len();
    for m in 0..tk {
        let (eid, raw_w) = top[m];
        let ew = raw_w / w_sum;
        if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe { blk.exp_up[eid].matvec_with_lut_avx2(&st.luts_d, inv_sc_in, &mut st.moe_h); }
        } else {
            blk.exp_up[eid].matvec_scalar(&st.xn, &mut st.moe_h);
        }
        for v in st.moe_h.iter_mut() { *v = relu_sq_clamp6(*v); }
        if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let inv_h = build_tl2_lut_avx2(&st.moe_h, d_ff, st.luts_ff.len(), st.luts_ff.as_mut_ptr());
                blk.exp_down[eid].matvec_with_lut_avx2(&st.luts_ff, inv_h, &mut st.moe_e);
            }
        } else {
            blk.exp_down[eid].matvec_scalar(&st.moe_h, &mut st.moe_e);
        }
        for i in 0..d { st.moe_acc[i] += ew * st.moe_e[i]; }
    }
    for i in 0..d { x[i] += st.moe_acc[i]; }
}

fn run_attn_step(
    blk: &HuginnAttnBlock, kv: &mut Fp4LayerKv, x: &mut [f32], st: &mut V9RuntimeState,
    attn_heads: usize, pos: usize, write_kv: bool,
) {
    let d = x.len();
    let hd = d / attn_heads;
    rms_norm(x, &blk.norm1, &mut st.xn);
    matvec_f16(&blk.wq.w, d, d, &st.xn, None, &mut st.q);
    apply_rope_pos(&mut st.q, attn_heads, hd, pos);
    if write_kv {
        matvec_f16(&blk.wk.w, d, d, &st.xn, None, &mut st.k);
        apply_rope_pos(&mut st.k, attn_heads, hd, pos);
        matvec_f16(&blk.wv.w, d, d, &st.xn, None, &mut st.v);
        kv.append(&st.k, &st.v);
    }
    kv.attend_step(&st.q, &mut st.dn_out);
    matvec_f16(&blk.wo.w, d, d, &st.dn_out, None, &mut st.tmp_d);
    for i in 0..d { x[i] += st.tmp_d[i]; }

    rms_norm(x, &blk.norm2, &mut st.xn);
    let d_ff = blk.ff_up.rows;
    matvec_f16(&blk.ff_up.w, d_ff, d, &st.xn, None, &mut st.ff_h);
    for v in st.ff_h.iter_mut() { *v = relu_sq_clamp6(*v); }
    matvec_f16(&blk.ff_dn.w, d, d_ff, &st.ff_h, None, &mut st.tmp_d);
    for i in 0..d { x[i] += st.tmp_d[i]; }
}

fn run_engram_step(eng: &EngramV41Weights, tok_id: usize, x: &mut [f32], st: &mut V9RuntimeState, vocab: usize, buckets: usize, edim: usize, conv_k: usize) {
    let d = x.len();
    let cid = eng.canon_lut[tok_id.min(vocab - 1)] as i64;
    let [p3, p2, p1] = st.eng_prev_cids;
    st.eng_prev_cids = [p2, p1, cid];
    for (i, tbl) in eng.tables.iter().enumerate() {
        let (m0, m1, m2) = HASH_PRIMES[i % HASH_PRIMES.len()];
        let hv = if i % 3 == 0 {
            cid.wrapping_mul(m0) ^ (p1.wrapping_mul(m1).wrapping_add(m2))
        } else if i % 3 == 1 {
            cid.wrapping_mul(m0) ^ p1.wrapping_mul(m1) ^ p2.wrapping_mul(m2)
        } else {
            cid.wrapping_mul(m0) ^ p1.wrapping_mul(m1) ^ p2.wrapping_mul(m2) ^ p3.wrapping_mul(m0 ^ m2)
        };
        let b = ((hv & 0x7FFF_FFFF) as usize) % buckets;
        let sc = f16_to_f32(tbl.scales[b]);
        let row = &tbl.q[b * edim..(b + 1) * edim];
        let dst = &mut st.eng_raw[i * edim..(i + 1) * edim];
        for j in 0..edim { dst[j] = (row[j] as f32) * sc; }
    }
    let eng_tot = st.eng_raw.len();
    matvec_f16(&eng.wk.w, d, eng_tot, &st.eng_raw, None, &mut st.eng_k);
    matvec_f16(&eng.wv.w, d, eng_tot, &st.eng_raw, None, &mut st.eng_v);
    rms_norm(x, &eng.norm_h, &mut st.eng_hn);
    rms_norm(&st.eng_k, &eng.norm_k, &mut st.eng_kn);
    matvec_f16(&eng.ctx_down.w, eng.ctx_down.rows, d, &st.eng_hn, None, &mut st.eng_ctx);
    for v in st.eng_ctx.iter_mut() { *v = silu(*v); }
    matvec_f16(&eng.ctx_up.w, d, eng.ctx_up.cols, &st.eng_ctx, eng.ctx_up.bias.as_deref(), &mut st.eng_gamma);
    for v in st.eng_gamma.iter_mut() { *v = sigmoid(*v); }

    let g_dim = eng.g_dim;
    let inv_g = 1.0 / (g_dim as f32).sqrt();
    for g in 0..eng.gate_groups {
        let mut dot = 0.0f32;
        for j in 0..g_dim { dot += st.eng_hn[g * g_dim + j] * st.eng_kn[g * g_dim + j]; }
        let ag = sigmoid(dot * inv_g + eng.head_bias[g]);
        for j in 0..g_dim {
            let idx = g * g_dim + j;
            st.eng_vg[idx] = ag * st.eng_gamma[idx] * st.eng_v[idx];
        }
    }
    rms_norm(&st.eng_vg, &eng.norm_v, &mut st.eng_vn);
    let ck1 = conv_k - 1;
    for i in 0..d {
        let tb = i * ck1;
        let wb = i * conv_k;
        let mut cs = 0.0f32;
        for k in 0..ck1 { cs += st.eng_conv_tail[tb + k] * eng.conv_w[wb + k]; }
        cs += st.eng_vn[i] * eng.conv_w[wb + ck1];
        for k in 0..(ck1 - 1) { st.eng_conv_tail[tb + k] = st.eng_conv_tail[tb + k + 1]; }
        st.eng_conv_tail[tb + ck1 - 1] = st.eng_vn[i];
        x[i] += st.eng_vg[i] + silu(cs);
    }
}

fn run_peer_step(peer: &PeerWeights, x: &mut [f32], st: &mut V9RuntimeState, nh: usize, sub_k: usize, pk: usize) {
    let d = x.len();
    let sub_d = (d / nh) / 2;
    let scale = 1.0 / (sub_d as f32).sqrt();
    rms_norm(x, &peer.norm, &mut st.xn);
    matvec_f16(&peer.wq.w, d, d, &st.xn, None, &mut st.q);
    st.moe_acc.fill(0.0);
    let mut q1 = [0.0f32; 64];
    let mut q2 = [0.0f32; 64];
    let mut sa = vec![0.0f32; sub_k];
    let mut sb = vec![0.0f32; sub_k];

    for h in 0..nh {
        let qbase = h * 2 * sub_d;
        rms_norm(&st.q[qbase..qbase + sub_d], &peer.q_norm_a[h * sub_d..(h + 1) * sub_d], &mut q1[..sub_d]);
        rms_norm(&st.q[qbase + sub_d..qbase + 2 * sub_d], &peer.q_norm_b[h * sub_d..(h + 1) * sub_d], &mut q2[..sub_d]);
        let ka = &peer.keys_a[h * sub_k * sub_d..(h + 1) * sub_k * sub_d];
        let kb = &peer.keys_b[h * sub_k * sub_d..(h + 1) * sub_k * sub_d];
        matvec_f16(ka, sub_k, sub_d, &q1[..sub_d], None, &mut sa);
        matvec_f16(kb, sub_k, sub_d, &q2[..sub_d], None, &mut sb);
        for v in sa.iter_mut() { *v *= scale; }
        for v in sb.iter_mut() { *v *= scale; }
        let mut ta = [(0usize, 0.0f32); 16];
        let mut tb = [(0usize, 0.0f32); 16];
        topk_indices(&sa, pk, &mut ta);
        topk_indices(&sb, pk, &mut tb);
        let mut comb = [(0usize, 0.0f32); 64];
        let mut ci = 0usize;
        for i in 0..pk {
            for j in 0..pk {
                comb[ci] = (ta[i].0 * sub_k + tb[j].0, ta[i].1 + tb[j].1);
                ci += 1;
            }
        }
        comb[..ci].sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let mut ws = [0.0f32; 16];
        for m in 0..pk { ws[m] = comb[m].1; }
        softmax_inplace(&mut ws[..pk]);
        for m in 0..pk {
            let eid = comb[m].0;
            let u_sc = f16_to_f32(peer.u_exp.scales[eid]);
            let u_row = &peer.u_exp.q[eid * d..(eid + 1) * d];
            let act = relu_sq_clamp6(dot_i8_row(u_row, u_sc, &st.xn));
            if act > 0.0 {
                let v_sc = f16_to_f32(peer.v_exp.scales[eid]);
                let v_row = &peer.v_exp.q[eid * d..(eid + 1) * d];
                axpy_i8_row(v_row, ws[m] * act * v_sc, &mut st.moe_acc);
            }
        }
    }
    for i in 0..d { x[i] += st.moe_acc[i]; }
}

/// Executes ONE full token step through `Genesis-v9 1.5B` (supports Asymmetric CED `top_k` & `recur_steps`).
pub fn step_token(
    w: &GenesisV9Weights, st: &mut V9RuntimeState, tok_id: usize,
    recur_steps: usize, top_k: usize, compute_logits: bool,
) {
    let c = &w.cfg;
    let d = c.d_model;
    let r_steps = recur_steps.clamp(1, c.max_recur_steps);
    let tid = tok_id.min(c.vocab - 1);
    let mut x = vec![0.0f32; d];
    for i in 0..d { x[i] = f16_to_f32(w.emb[tid * d + i]); }

    for (i, blk) in w.prelude_dn.iter().enumerate() {
        let mut s_buf = std::mem::take(&mut st.pre_s[i]);
        run_dn_moe_step(blk, &mut s_buf, &mut x, st, c.delta_heads, top_k);
        st.pre_s[i] = s_buf;
    }
    let mut kv_pre = std::mem::replace(&mut st.kv_prelude, Fp4LayerKv::new(1, 2, 2, 1));
    run_attn_step(&w.prelude_attn, &mut kv_pre, &mut x, st, c.attn_heads, st.pos, true);
    st.kv_prelude = kv_pre;

    run_engram_step(&w.engram, tid, &mut x, st, c.vocab, c.engram_buckets, c.engram_dim, c.engram_conv_k);

    let s0 = x.clone();
    let mut y = x;
    let mut z = vec![0.0f32; d];
    let mut inj = vec![0.0f32; d];
    for i in 0..d { inj[i] = w.recur_inj[i].tanh(); }

    let mut kv_cor = std::mem::replace(&mut st.kv_core, Fp4LayerKv::new(1, 2, 2, 1));
    for r in 0..r_steps {
        for i in 0..d {
            let base = if r == 0 { s0[i] } else { z[i] + inj[i] * 0.5 * (s0[i] + y[i]) };
            z[i] = base + w.step_emb[r * d + i];
        }
        // core_dn[0]
        let idx0 = r;
        if r > 0 && st.pos > 0 && !st.core_inited[idx0] && st.core_inited[0] {
            let src = st.core_s[0].clone();
            st.core_s[idx0] = src;
        }
        st.core_inited[idx0] = true;
        let mut s0_buf = std::mem::take(&mut st.core_s[idx0]);
        run_dn_moe_step(&w.core_dn[0], &mut s0_buf, &mut z, st, c.delta_heads, top_k);
        st.core_s[idx0] = s0_buf;
        run_peer_step(&w.peer0, &mut z, st, c.peer_heads, c.peer_sub_k, c.peer_k);
        run_attn_step(&w.core_attn, &mut kv_cor, &mut z, st, c.attn_heads, st.pos, r == 0);

        if r == 0 { y.copy_from_slice(&z); } else { for i in 0..d { y[i] += inj[i] * z[i]; } }
        for j in 1..c.n_core_dn {
            let idxj = j * c.max_recur_steps + r;
            let basej = j * c.max_recur_steps;
            if r > 0 && st.pos > 0 && !st.core_inited[idxj] && st.core_inited[basej] {
                let src = st.core_s[basej].clone();
                st.core_s[idxj] = src;
            }
            st.core_inited[idxj] = true;
            let mut sj = std::mem::take(&mut st.core_s[idxj]);
            run_dn_moe_step(&w.core_dn[j], &mut sj, &mut y, st, c.delta_heads, top_k);
            st.core_s[idxj] = sj;
        }
        run_peer_step(&w.peer1, &mut y, st, c.peer_heads, c.peer_sub_k, c.peer_k);
    }
    st.kv_core = kv_cor;
    st.pos += 1;

    for (i, blk) in w.coda_dn.iter().enumerate() {
        let mut sc = std::mem::take(&mut st.coda_s[i]);
        run_dn_moe_step(blk, &mut sc, &mut y, st, c.delta_heads, top_k);
        st.coda_s[i] = sc;
    }
    if compute_logits {
        rms_norm(&y, &w.norm_f, &mut st.xn);
        lm_head_f16_parallel(&w.emb, c.vocab, d, &st.xn, &mut st.logits);
    }
}

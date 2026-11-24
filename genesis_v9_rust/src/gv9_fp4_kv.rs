//! Exact Genesis-v9 `FP4KVCacheV9` (E2M1 Block-32 Nibble Packing) + RoPE + Sliding Attention (`<= 175` LOC).
//!
//! 100% mathematical parity with `colab_gpu_trainer/v9_v41_upgrades.py`:
//!   - `_FP4_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)`
//!   - `_FP4_LUT  = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)`
//!   - `1,440 bytes/token` across both attention layers (`prelude_attn` + `core_attn`, `10 heads x 64 dim`).

use crate::gv9_simd::{f16_to_f32, f32_to_f16};

const FP4_MIDS: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
const FP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[inline]
fn quant_fp4_nibble(v: f32) -> u8 {
    let neg = if v < 0.0 { 8u8 } else { 0u8 };
    let a = v.abs();
    let mut mag = 0u8;
    for &m in &FP4_MIDS {
        if a >= m { mag += 1; } else { break; }
    }
    neg | mag
}

/// Applies exact RoPE at absolute position `pos` to `x[n_heads * head_dim]`.
pub fn apply_rope_pos(x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize) {
    let half = head_dim / 2;
    let p = pos as f32;
    let mut cos_v = [0.0f32; 64];
    let mut sin_v = [0.0f32; 64];
    for i in 0..half {
        let inv = 1.0 / 10000.0f32.powf((2 * i) as f32 / (head_dim as f32));
        let ang = p * inv;
        cos_v[i] = ang.cos();
        sin_v[i] = ang.sin();
    }
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let x0 = x[base + 2 * i];
            let x1 = x[base + 2 * i + 1];
            x[base + 2 * i] = x0 * cos_v[i] - x1 * sin_v[i];
            x[base + 2 * i + 1] = x0 * sin_v[i] + x1 * cos_v[i];
        }
    }
}

pub struct Fp4LayerKv {
    pub n_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub n_blocks_per_head: usize,
    pub max_win: usize,
    pub len: usize,
    // Ring buffer indexed by `slot = abs_pos % max_win`:
    // packed: [max_win, n_heads, head_dim/2], scale: [max_win, n_heads, n_blocks]
    k_packed: Vec<u8>,
    k_scale: Vec<u16>,
    v_packed: Vec<u8>,
    v_scale: Vec<u16>,
}

impl Fp4LayerKv {
    pub fn new(n_heads: usize, head_dim: usize, block_size: usize, max_win: usize) -> Self {
        let nb = head_dim / block_size;
        Self {
            n_heads, head_dim, block_size, n_blocks_per_head: nb, max_win, len: 0,
            k_packed: vec![0u8; max_win * n_heads * (head_dim / 2)],
            k_scale: vec![0u16; max_win * n_heads * nb],
            v_packed: vec![0u8; max_win * n_heads * (head_dim / 2)],
            v_scale: vec![0u16; max_win * n_heads * nb],
        }
    }

    pub fn reset(&mut self) { self.len = 0; }

    fn quantize_vec(src: &[f32], n_heads: usize, hd: usize, bq: usize, dst_p: &mut [u8], dst_s: &mut [u16]) {
        let nb = hd / bq;
        let hp = hd / 2;
        for h in 0..n_heads {
            for b in 0..nb {
                let s0 = h * hd + b * bq;
                let mut amax = 0.0f32;
                for i in 0..bq {
                    let a = src[s0 + i].abs();
                    if a > amax { amax = a; }
                }
                let sc_h = f32_to_f16((amax.max(1e-8)) / 6.0);
                let inv = 1.0 / f16_to_f32(sc_h).max(1e-8);
                dst_s[h * nb + b] = sc_h;
                let p0 = h * hp + b * (bq / 2);
                for j in 0..(bq / 2) {
                    let c0 = quant_fp4_nibble(src[s0 + 2 * j] * inv);
                    let c1 = quant_fp4_nibble(src[s0 + 2 * j + 1] * inv);
                    dst_p[p0 + j] = c0 | (c1 << 4);
                }
            }
        }
    }

    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        let slot = self.len % self.max_win;
        let stride_p = self.n_heads * (self.head_dim / 2);
        let stride_s = self.n_heads * self.n_blocks_per_head;
        Self::quantize_vec(
            k, self.n_heads, self.head_dim, self.block_size,
            &mut self.k_packed[slot * stride_p..(slot + 1) * stride_p],
            &mut self.k_scale[slot * stride_s..(slot + 1) * stride_s],
        );
        Self::quantize_vec(
            v, self.n_heads, self.head_dim, self.block_size,
            &mut self.v_packed[slot * stride_p..(slot + 1) * stride_p],
            &mut self.v_scale[slot * stride_s..(slot + 1) * stride_s],
        );
        self.len += 1;
    }

    #[inline]
    fn dequant_head(packed: &[u8], scales: &[u16], hd: usize, bq: usize, out: &mut [f32]) {
        let nb = hd / bq;
        for b in 0..nb {
            let sc = f16_to_f32(scales[b]);
            let p0 = b * (bq / 2);
            let o0 = b * bq;
            for j in 0..(bq / 2) {
                let byte = packed[p0 + j];
                out[o0 + 2 * j] = FP4_LUT[(byte & 0x0F) as usize] * sc;
                out[o0 + 2 * j + 1] = FP4_LUT[((byte >> 4) & 0x0F) as usize] * sc;
            }
        }
    }

    /// Single-token causal sliding-window attention `out[n_heads * head_dim]` against FP4 KV cache.
    pub fn attend_step(&self, q: &[f32], out: &mut [f32]) {
        let count = self.len.min(self.max_win);
        let start = self.len.saturating_sub(self.max_win);
        let hd = self.head_dim;
        let hp = hd / 2;
        let nb = self.n_blocks_per_head;
        let stride_p = self.n_heads * hp;
        let stride_s = self.n_heads * nb;
        let inv_sqrt = 1.0 / (hd as f32).sqrt();
        let mut scores = vec![0.0f32; count];
        let mut tmp = [0.0f32; 128];

        for h in 0..self.n_heads {
            let qh = &q[h * hd..(h + 1) * hd];
            let mut max_s = f32::NEG_INFINITY;
            for t in 0..count {
                let slot = (start + t) % self.max_win;
                let kp = &self.k_packed[slot * stride_p + h * hp..slot * stride_p + (h + 1) * hp];
                let ks = &self.k_scale[slot * stride_s + h * nb..slot * stride_s + (h + 1) * nb];
                Self::dequant_head(kp, ks, hd, self.block_size, &mut tmp[..hd]);
                let mut dot = 0.0f32;
                for d in 0..hd { dot += qh[d] * tmp[d]; }
                let s = dot * inv_sqrt;
                scores[t] = s;
                if s > max_s { max_s = s; }
            }
            let mut sum_e = 0.0f32;
            for t in 0..count {
                let e = (scores[t] - max_s).exp();
                scores[t] = e;
                sum_e += e;
            }
            let inv_e = 1.0 / sum_e.max(1e-12);
            let oh = &mut out[h * hd..(h + 1) * hd];
            for v in oh.iter_mut() { *v = 0.0; }
            for t in 0..count {
                let w = scores[t] * inv_e;
                let slot = (start + t) % self.max_win;
                let vp = &self.v_packed[slot * stride_p + h * hp..slot * stride_p + (h + 1) * hp];
                let vs = &self.v_scale[slot * stride_s + h * nb..slot * stride_s + (h + 1) * nb];
                Self::dequant_head(vp, vs, hd, self.block_size, &mut tmp[..hd]);
                for d in 0..hd { oh[d] += w * tmp[d]; }
            }
        }
    }

    pub fn bytes_per_token(&self) -> usize {
        self.n_heads * (self.head_dim + self.n_blocks_per_head * 4)
    }
}

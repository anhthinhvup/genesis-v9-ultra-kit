//! Genesis-v9 Pure-Rust AVX2+FMA+F16C SIMD Kernels (`<= 245` LOC, Zero External Crates).
//!
//! Includes: exact IEEE-754 f16<->f32, RMSNorm (eps=1e-5), L2-norm, AVX2 F16 matvec,
//! AVX2 per-row INT8 dot/axpy (PEER & Engram), and register-resident Channel-Gated DeltaNet step.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h as u32) >> 15) << 31;
    let e = ((h >> 10) & 0x1F) as u32;
    let m = (h & 0x3FF) as u32;
    let bits = if e == 0 {
        if m == 0 { s } else {
            let mut m2 = m;
            let mut e2 = 127 - 14;
            while (m2 & 0x400) == 0 { m2 <<= 1; e2 -= 1; }
            s | (e2 << 23) | ((m2 & 0x3FF) << 13)
        }
    } else if e == 31 {
        s | 0x7F80_0000 | (m << 13)
    } else {
        s | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

#[inline]
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 255 {
        return sign | 0x7C00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 { return sign | 0x7BFF; }
    if e <= 0 {
        if e < -10 { return sign; }
        let m = (mant | 0x80_0000) >> (14 - e);
        return sign | (m as u16);
    }
    let r = mant + 0x1000;
    if (r & 0x80_0000) != 0 {
        return sign | (((e + 1) as u16) << 10);
    }
    sign | ((e as u16) << 10) | ((r >> 13) as u16)
}

#[inline]
pub fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

#[inline]
pub fn silu(x: f32) -> f32 { x * sigmoid(x) }

#[inline]
pub fn logsigmoid_clamped(x: f32) -> f32 {
    let v = if x >= 0.0 { -((-x).exp().ln_1p()) } else { x - (x.exp().ln_1p()) };
    v.max(-6.0)
}

#[inline]
pub fn relu_sq_clamp6(x: f32) -> f32 {
    let r = x.clamp(0.0, 6.0);
    r * r
}

#[inline]
pub fn rms_norm(x: &[f32], g: &[f32], out: &mut [f32]) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x { ss += v * v; }
    let inv = 1.0 / ((ss / (n as f32)) + 1e-5).sqrt();
    for i in 0..n { out[i] = x[i] * inv * g[i]; }
}

#[inline]
pub fn l2_norm_inplace(x: &mut [f32]) {
    let mut ss = 0.0f32;
    for &v in x.iter() { ss += v * v; }
    let inv = 1.0 / ss.sqrt().max(1e-5);
    for v in x.iter_mut() { *v *= inv; }
}

pub fn softmax_inplace(x: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in x.iter() { if v > m { m = v; } }
    let mut s = 0.0f32;
    for v in x.iter_mut() { *v = (*v - m).exp(); s += *v; }
    let inv = 1.0 / s.max(1e-12);
    for v in x.iter_mut() { *v *= inv; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn hsum256(v: __m256) -> f32 {
    let s128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let sh = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, sh);
    _mm_cvtss_f32(_mm_add_ss(sums, _mm_movehl_ps(sh, sums)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_f16_f32_avx2(w: *const u16, x: *const f32, n: usize) -> f32 {
    let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut c = 0usize;
    while c + 16 <= n {
        let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w.add(c) as *const __m128i));
        let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w.add(c + 8) as *const __m128i));
        a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x.add(c)), a0);
        a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x.add(c + 8)), a1);
        c += 16;
    }
    while c + 8 <= n {
        let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w.add(c) as *const __m128i));
        a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x.add(c)), a0);
        c += 8;
    }
    let mut s = hsum256(_mm256_add_ps(a0, a1));
    while c < n { s += f16_to_f32(*w.add(c)) * (*x.add(c)); c += 1; }
    s
}

/// Matrix-vector `y[rows] = W_f16[rows, cols] @ x[cols]`, optionally adding `bias`.
pub fn matvec_f16(w: &[u16], rows: usize, cols: usize, x: &[f32], bias: Option<&[f32]>, y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        unsafe {
            let xp = x.as_ptr();
            for r in 0..rows {
                let v = dot_f16_f32_avx2(w.as_ptr().add(r * cols), xp, cols);
                y[r] = if let Some(b) = bias { v + b[r] } else { v };
            }
        }
        return;
    }
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut s = 0.0f32;
        for c in 0..cols { s += f16_to_f32(row[c]) * x[c]; }
        y[r] = if let Some(b) = bias { s + b[r] } else { s };
    }
}

/// Multi-threaded F16 matvec for large `[16384, 640]` `lm_head` with softcap `30 * tanh(v / 30)`.
pub fn lm_head_f16_parallel(emb: &[u16], vocab: usize, d: usize, x: &[f32], logits: &mut [f32]) {
    let n_threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4).min(8);
    let chunk = (vocab + n_threads - 1) / n_threads;
    std::thread::scope(|s| {
        for (ti, out_slice) in logits.chunks_mut(chunk).enumerate() {
            let r0 = ti * chunk;
            let rows = out_slice.len();
            let w_slice = &emb[r0 * d..(r0 + rows) * d];
            s.spawn(move || {
                matvec_f16(w_slice, rows, d, x, None, out_slice);
                for v in out_slice.iter_mut() { *v = 30.0 * (*v / 30.0).tanh(); }
            });
        }
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_i8_f32_avx2(w: *const i8, x: *const f32, n: usize) -> f32 {
    let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut c = 0usize;
    while c + 16 <= n {
        let q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w.add(c) as *const __m128i)));
        let q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w.add(c + 8) as *const __m128i)));
        a0 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(x.add(c)), a0);
        a1 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(x.add(c + 8)), a1);
        c += 16;
    }
    let mut s = hsum256(_mm256_add_ps(a0, a1));
    while c < n { s += (*w.add(c) as f32) * (*x.add(c)); c += 1; }
    s
}

#[inline]
pub fn dot_i8_row(row: &[i8], scale: f32, x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return unsafe { dot_i8_f32_avx2(row.as_ptr(), x.as_ptr(), row.len()) } * scale;
    }
    let mut s = 0.0f32;
    for i in 0..row.len() { s += (row[i] as f32) * x[i]; }
    s * scale
}

#[inline]
pub fn axpy_i8_row(row: &[i8], weight: f32, dst: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        unsafe {
            let vw = _mm256_set1_ps(weight);
            let n = row.len();
            let mut c = 0usize;
            while c + 8 <= n {
                let q = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(row.as_ptr().add(c) as *const __m128i)));
                let cur = _mm256_loadu_ps(dst.as_ptr().add(c));
                _mm256_storeu_ps(dst.as_mut_ptr().add(c), _mm256_fmadd_ps(q, vw, cur));
                c += 8;
            }
            while c < n { dst[c] += (row[c] as f32) * weight; c += 1; }
        }
        return;
    }
    for i in 0..row.len() { dst[i] += (row[i] as f32) * weight; }
}

/// Exact single-token Channel-Gated DeltaNet update for one head `S[hd, hd]` (`hd = 64`).
/// Matches `genesis_v9_v41._deltanet_step_loop`:
///   `S[k, d] *= exp(la[k])`; `err[d] = v[d] - sum_k S[k, d]*k[k]`;
///   `S[k, d] += k[k] * (beta * err[d])`; `o[d] = sum_k S[k, d]*q[k]`.
pub fn deltanet_head_step(
    s_mat: &mut [f32], q: &[f32], k: &[f32], v: &[f32], la: &[f32], beta: f32, out: &mut [f32],
) {
    let hd = q.len();
    let mut err = [0.0f32; 128];
    for d in 0..hd { err[d] = v[d]; }
    // Pass 1: decay rows of S and compute err = v - k^T S
    for r in 0..hd {
        let dec = la[r].exp();
        let kr = k[r];
        let row = &mut s_mat[r * hd..(r + 1) * hd];
        for d in 0..hd {
            let sd = row[d] * dec;
            row[d] = sd;
            err[d] -= sd * kr;
        }
    }
    for d in 0..hd { err[d] *= beta; out[d] = 0.0; }
    // Pass 2: rank-1 update S += k (beta * err)^T and query o = q^T S
    for r in 0..hd {
        let kr = k[r];
        let qr = q[r];
        let row = &mut s_mat[r * hd..(r + 1) * hd];
        for d in 0..hd {
            let sd = row[d] + kr * err[d];
            row[d] = sd;
            out[d] += sd * qr;
        }
    }
}

//! Genesis-v9 BitNet b1.58 TL2 Ternary Packing & AVX2 `vpshufb` 24-bit LUT Matvec (`<= 225` LOC).
//!
//! 3 trits (`3^3 = 27` states) -> 4-bit nibble (`0..=13`) + 1 sign bit (`1.667 bpw`).
//! 32-row x 6-col tiles (`40 B / 192 weights`), exact mathematical match with `BitNetMoEV9._ste_158`.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const LO_ROWS: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23];
const HI_ROWS: [usize; 16] = [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31];

#[inline]
fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline]
pub fn encode_3trit_tl2(w0: i8, w1: i8, w2: i8) -> (u8, u8) {
    let c = 9 * (w0.clamp(-1, 1) as i32) + 3 * (w1.clamp(-1, 1) as i32) + (w2.clamp(-1, 1) as i32);
    if c < 0 { ((-c) as u8, 1) } else { (c as u8, 0) }
}

fn decode_tile_signs(tile: &[u8]) -> ([u8; 32], [u8; 32]) {
    let (s_lo, s_hi, mut sg0, mut sg1) = (read_u32_le(&tile[32..36]), read_u32_le(&tile[36..40]), [0u8; 32], [0u8; 32]);
    for (k, &rr) in LO_ROWS.iter().enumerate() {
        sg0[rr] = ((s_lo >> (2 * k)) & 1) as u8;
        sg1[rr] = ((s_lo >> (2 * k + 1)) & 1) as u8;
    }
    for (k, &rr) in HI_ROWS.iter().enumerate() {
        sg0[rr] = ((s_hi >> (2 * k)) & 1) as u8;
        sg1[rr] = ((s_hi >> (2 * k + 1)) & 1) as u8;
    }
    (sg0, sg1)
}

#[derive(Clone)]
pub struct BitLinearTl2 {
    pub rows: usize,
    pub cols: usize,
    pub n_pairs: usize,
    pub gamma: f32,
    pub data: Vec<u8>,
    pub row_sign_sum: Vec<i16>,
}

#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct Tl2LutPair(pub [[u8; 16]; 6]);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn build_tl2_lut_avx2(x: &[f32], cols: usize, n_pairs: usize, dst: *mut Tl2LutPair) -> f64 {
    let (n_groups, mut max_l) = ((cols + 2) / 3, 0.0f32);
    for g in 0..n_groups {
        let c = g * 3;
        let mv = (if c < cols { x[c].abs() } else { 0.0 })
            + (if c + 1 < cols { x[c + 1].abs() } else { 0.0 })
            + (if c + 2 < cols { x[c + 2].abs() } else { 0.0 });
        if mv > max_l { max_l = mv; }
    }
    if !max_l.is_finite() || max_l == 0.0 { return 0.0; }
    let scale = 8_388_352.0f64 / (max_l as f64);
    let (vs, voff) = (_mm256_set1_ps(scale as f32), _mm256_set1_epi32(1 << 23));
    let sh0 = _mm256_setr_epi8(0,4,8,12,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0,4,8,12,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
    let sh1 = _mm256_setr_epi8(1,5,9,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,5,9,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
    let sh2 = _mm256_setr_epi8(2,6,10,14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 2,6,10,14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
    for p in 0..n_pairs {
        let lp = dst.add(p) as *mut u8;
        for sub in 0..2 {
            let g = 2 * p + sub;
            if g < n_groups {
                let c = g * 3;
                let (x0, x1, x2) = (
                    if c < cols { x[c] } else { 0.0 },
                    if c + 1 < cols { x[c + 1] } else { 0.0 },
                    if c + 2 < cols { x[c + 2] } else { 0.0 },
                );
                let i0 = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(
                    _mm256_setr_ps(0.0, x2, x1 - x2, x1, x1 + x2, x0 - x1 - x2, x0 - x1, x0 - x1 + x2), vs)), voff);
                let i1 = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(
                    _mm256_setr_ps(x0 - x2, x0, x0 + x2, x0 + x1 - x2, x0 + x1, x0 + x1 + x2, 0.0, 0.0), vs)), voff);
                macro_rules! pk { ($sh:expr, $sl:expr) => {{
                    let (a, b, ptr) = (_mm256_shuffle_epi8(i0, $sh), _mm256_shuffle_epi8(i1, $sh), lp.add(($sl) * 16) as *mut u32);
                    *ptr.add(0) = _mm256_extract_epi32(a, 0) as u32;
                    *ptr.add(1) = _mm256_extract_epi32(a, 4) as u32;
                    *ptr.add(2) = _mm256_extract_epi32(b, 0) as u32;
                    *ptr.add(3) = _mm256_extract_epi32(b, 4) as u32;
                }}; }
                pk!(sh0, sub); pk!(sh1, 2 + sub); pk!(sh2, 4 + sub);
            } else {
                std::ptr::write_bytes(lp.add(sub * 16), 0, 16);
                std::ptr::write_bytes(lp.add((2 + sub) * 16), 0, 16);
                std::ptr::write_bytes(lp.add((4 + sub) * 16), 128, 16);
            }
        }
    }
    1.0 / scale
}

impl BitLinearTl2 {
    pub fn from_raw(rows: usize, cols: usize, gamma: f32, data: Vec<u8>, row_sign_sum: Vec<i16>) -> Self {
        let n_pairs = (((cols + 2) / 3) + 1) / 2;
        Self { rows, cols, n_pairs, gamma, data, row_sign_sum }
    }

    pub fn pack(q: &[i8], gamma: f32, rows: usize, cols: usize) -> Self {
        let (n_pairs, n_tiles) = ((((cols + 2) / 3) + 1) / 2, (rows + 31) / 32);
        let (mut data, mut row_sign_sum) = (vec![0u8; n_tiles * n_pairs * 40], vec![0i16; n_tiles * 32]);
        let get = |r: usize, c: usize| -> i8 { if r < rows && c < cols { q[r * cols + c] } else { 0 } };
        for t in 0..n_tiles {
            let r0 = t * 32;
            for p in 0..n_pairs {
                let (base, c0, mut s_lo, mut s_hi, mut s0, mut s1) = ((t * n_pairs + p) * 40, p * 6, 0u32, 0u32, [0u8; 32], [0u8; 32]);
                for rr in 0..32 {
                    let (m0, sg0) = encode_3trit_tl2(get(r0 + rr, c0), get(r0 + rr, c0 + 1), get(r0 + rr, c0 + 2));
                    let (m1, sg1) = encode_3trit_tl2(get(r0 + rr, c0 + 3), get(r0 + rr, c0 + 4), get(r0 + rr, c0 + 5));
                    data[base + rr] = m0 | (m1 << 4);
                    s0[rr] = sg0; s1[rr] = sg1;
                    row_sign_sum[r0 + rr] += (if sg0 == 0 { 1 } else { -1 }) + (if sg1 == 0 { 1 } else { -1 });
                }
                for (k, &rr) in LO_ROWS.iter().enumerate() { s_lo |= ((s0[rr] as u32) << (2 * k)) | ((s1[rr] as u32) << (2 * k + 1)); }
                for (k, &rr) in HI_ROWS.iter().enumerate() { s_hi |= ((s0[rr] as u32) << (2 * k)) | ((s1[rr] as u32) << (2 * k + 1)); }
                data[base + 32..base + 36].copy_from_slice(&s_lo.to_le_bytes());
                data[base + 36..base + 40].copy_from_slice(&s_hi.to_le_bytes());
            }
        }
        Self { rows, cols, n_pairs, gamma, data, row_sign_sum }
    }

    pub fn matvec_scalar(&self, x: &[f32], y: &mut [f32]) {
        let (n_groups, mut lut) = ((self.cols + 2) / 3, vec![[0.0f64; 16]; self.n_pairs * 2]);
        for g in 0..n_groups {
            let c = g * 3;
            let (x0, x1, x2) = (
                if c < self.cols { x[c] as f64 } else { 0.0 },
                if c + 1 < self.cols { x[c + 1] as f64 } else { 0.0 },
                if c + 2 < self.cols { x[c + 2] as f64 } else { 0.0 },
            );
            lut[g] = [0.0, x2, x1 - x2, x1, x1 + x2, x0 - x1 - x2, x0 - x1, x0 - x1 + x2, x0 - x2, x0, x0 + x2, x0 + x1 - x2, x0 + x1, x0 + x1 + x2, 0.0, 0.0];
        }
        for t in 0..((self.rows + 31) / 32) {
            let (r0, mut acc) = (t * 32, [0.0f64; 32]);
            for p in 0..self.n_pairs {
                let base = (t * self.n_pairs + p) * 40;
                let ((sg0, sg1), l0, l1) = (decode_tile_signs(&self.data[base..base + 40]), &lut[2 * p], &lut[2 * p + 1]);
                for rr in 0..32 {
                    let b = self.data[base + rr];
                    acc[rr] += (if sg0[rr] != 0 { -1.0 } else { 1.0 }) * l0[(b & 0x0F) as usize]
                        + (if sg1[rr] != 0 { -1.0 } else { 1.0 }) * l1[(b >> 4) as usize];
                }
            }
            for rr in 0..32 {
                if r0 + rr < self.rows { y[r0 + rr] = (acc[rr] * self.gamma as f64) as f32; }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn matvec_with_lut_avx2(&self, luts: &[Tl2LutPair], inv_scale: f64, y: &mut [f32]) {
        if inv_scale == 0.0 {
            for v in y.iter_mut().take(self.rows) { *v = 0.0; }
            return;
        }
        let inv_sc_g = inv_scale * (self.gamma as f64);
        let (mask_0f, ones) = (_mm256_set1_epi8(0x0F), _mm256_set1_epi8(1));
        let bit_shuf = _mm256_setr_epi8(0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
        let bit_mask = _mm256_setr_epi8(1,2,4,8,16,32,64,-128, 1,2,4,8,16,32,64,-128, 1,2,4,8,16,32,64,-128, 1,2,4,8,16,32,64,-128);
        for t in 0..((self.rows + 31) / 32) {
            let (mut tot0, mut tot1, mut tot2, mut p_start) = ([0i32; 32], [0i32; 32], [0i32; 32], 0usize);
            while p_start < self.n_pairs {
                let p_end = (p_start + 64).min(self.n_pairs);
                let (mut a0l, mut a0h, mut a1l, mut a1h, mut a2l, mut a2h) = (
                    _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(),
                    _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(),
                );
                let (mut ptr, mut lptr) = (self.data.as_ptr().add((t * self.n_pairs + p_start) * 40), luts.as_ptr().add(p_start));
                for _ in p_start..p_end {
                    let idx_p = _mm256_loadu_si256(ptr as *const __m256i);
                    let (idx0, idx1) = (_mm256_and_si256(idx_p, mask_0f), _mm256_and_si256(_mm256_srli_epi16(idx_p, 4), mask_0f));
                    let sgn_l = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_broadcastd_epi32(_mm_loadu_si32(ptr.add(32))), bit_shuf), bit_mask), bit_mask), ones);
                    let sgn_h = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_broadcastd_epi32(_mm_loadu_si32(ptr.add(36))), bit_shuf), bit_mask), bit_mask), ones);
                    let lp = &(*lptr).0;
                    macro_rules! acc_d { ($k0:expr, $k1:expr, $al:expr, $ah:expr) => {{
                        let v0 = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_load_si128(lp[$k0].as_ptr() as *const __m128i)), idx0);
                        let v1 = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_load_si128(lp[$k1].as_ptr() as *const __m128i)), idx1);
                        $al = _mm256_add_epi16($al, _mm256_maddubs_epi16(_mm256_unpacklo_epi8(v0, v1), sgn_l));
                        $ah = _mm256_add_epi16($ah, _mm256_maddubs_epi16(_mm256_unpackhi_epi8(v0, v1), sgn_h));
                    }}; }
                    acc_d!(0, 1, a0l, a0h); acc_d!(2, 3, a1l, a1h); acc_d!(4, 5, a2l, a2h);
                    ptr = ptr.add(40); lptr = lptr.add(1);
                }
                let (mut b0l, mut b0h, mut b1l, mut b1h, mut b2l, mut b2h) = ([0i16; 16], [0i16; 16], [0i16; 16], [0i16; 16], [0i16; 16], [0i16; 16]);
                _mm256_storeu_si256(b0l.as_mut_ptr() as *mut __m256i, a0l); _mm256_storeu_si256(b0h.as_mut_ptr() as *mut __m256i, a0h);
                _mm256_storeu_si256(b1l.as_mut_ptr() as *mut __m256i, a1l); _mm256_storeu_si256(b1h.as_mut_ptr() as *mut __m256i, a1h);
                _mm256_storeu_si256(b2l.as_mut_ptr() as *mut __m256i, a2l); _mm256_storeu_si256(b2h.as_mut_ptr() as *mut __m256i, a2h);
                for (k, &rr) in LO_ROWS.iter().enumerate() { tot0[rr] += b0l[k] as i32; tot1[rr] += b1l[k] as i32; tot2[rr] += b2l[k] as i32; }
                for (k, &rr) in HI_ROWS.iter().enumerate() { tot0[rr] += b0h[k] as i32; tot1[rr] += b1h[k] as i32; tot2[rr] += b2h[k] as i32; }
                p_start = p_end;
            }
            let r0 = t * 32;
            for r in r0..(r0 + 32).min(self.rows) {
                let rr = r - r0;
                let v64 = (tot0[rr] as i64) + ((tot1[rr] as i64) << 8) + (((tot2[rr] as i64) - 128 * (self.row_sign_sum[r] as i64)) << 16);
                y[r] = (v64 as f64 * inv_sc_g) as f32;
            }
        }
    }

    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            let mut luts = vec![Tl2LutPair([[0u8; 16]; 6]); self.n_pairs];
            unsafe {
                let inv_sc = build_tl2_lut_avx2(x, self.cols, self.n_pairs, luts.as_mut_ptr());
                self.matvec_with_lut_avx2(&luts, inv_sc, y);
            }
            return;
        }
        self.matvec_scalar(x, y);
    }
}

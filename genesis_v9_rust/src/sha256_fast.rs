//! Pure-Rust Zero-Dependency FIPS-180-4 SHA-256 + Fast Word-Shingle Anti-Leak Filter (`<= 140` LOC).

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let bit_len = (data.len() as u64) * 8;
    let mut buf = Vec::with_capacity(((data.len() + 9 + 63) / 64) * 64);
    buf.extend_from_slice(data);
    buf.push(0x80);
    while (buf.len() % 64) != 56 { buf.push(0); }
    buf.extend_from_slice(&bit_len.to_be_bytes());
    let mut w = [0u32; 64];
    for chunk in buf.chunks_exact(64) {
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4*i], chunk[4*i+1], chunk[4*i+2], chunk[4*i+3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g; g = f; f = e; e = d.wrapping_add(t1);
            d = c; c = b; b = a; a = t1.wrapping_add(t2);
        }
        for (slot, delta) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(delta);
        }
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() { out[4*i..4*i+4].copy_from_slice(&v.to_be_bytes()); }
    out
}

pub fn sha256_prefix12(data: &[u8]) -> [u8; 12] {
    let full = sha256(data);
    let mut out = [0u8; 12];
    out.copy_from_slice(&full[..12]);
    out
}

#[inline]
fn is_punct_sep(c: char) -> bool {
    matches!(
        c,
        '.' | ',' | '!' | '?' | ';' | ':' | '-' | '\'' | '"' | '(' | ')' | '[' | ']' | '{' | '}'
        | '<' | '>' | '/' | '*' | '+' | '=' | '#' | '@' | '&' | '$' | '%' | '^' | '~' | '`' | '|' | '\\'
    ) || c.is_whitespace()
}

/// Zero-regex lowercase word tokenizer matching `mega_vault_merger.tokenize_clean_words`.
pub fn tokenize_clean_words(text: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(64);
    let mut cur = String::new();
    for ch in text.chars() {
        if is_punct_sep(ch) {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            for lc in ch.to_lowercase() { cur.push(lc); }
        }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

/// Normalized collapsed-whitespace lowercase SHA-256 matching `leakage_guard.norm_sha256`.
pub fn norm_sha256_bytes(text: &str) -> [u8; 32] {
    let mut collapsed = String::with_capacity(text.len());
    let mut first = true;
    for tok in text.split_whitespace() {
        if !first { collapsed.push(' '); }
        first = false;
        for ch in tok.chars() {
            for lc in ch.to_lowercase() { collapsed.push(lc); }
        }
    }
    sha256(collapsed.as_bytes())
}

/// Fast 64-bit FNV-1a hash of an 8-word shingle (exact match with exported Rust-guard index).
#[inline]
pub fn shingle_fnv64(words: &[String], start: usize, n: usize) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for i in 0..n {
        if i > 0 {
            h ^= b' ' as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        for &b in words[start + i].as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Fast binary guard containing sorted `[u8; 32]` normalized SHA-256 hashes + sorted `u64` 8-word shingles.
pub struct FastAntiLeakGuard {
    pub sha_sorted: Vec<[u8; 32]>,
    pub shingle_sorted: Vec<u64>,
    pub shingle_n: usize,
}

impl FastAntiLeakGuard {
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let raw = std::fs::read(path).map_err(|e| format!("read guard failed: {e}"))?;
        if raw.len() < 20 || &raw[0..8] != b"GV9GRD01" {
            return Err("invalid GV9GRD01 guard header".into());
        }
        let n_sha = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let n_sh = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
        let shingle_n = u32::from_le_bytes(raw[16..20].try_into().unwrap()) as usize;
        let need = 20 + n_sha * 32 + n_sh * 8;
        if raw.len() < need { return Err("truncated GV9GRD01 file".into()); }
        let mut sha_sorted = Vec::with_capacity(n_sha);
        let mut off = 20;
        for _ in 0..n_sha {
            let mut h = [0u8; 32];
            h.copy_from_slice(&raw[off..off + 32]);
            sha_sorted.push(h);
            off += 32;
        }
        let mut shingle_sorted = Vec::with_capacity(n_sh);
        for _ in 0..n_sh {
            shingle_sorted.push(u64::from_le_bytes(raw[off..off + 8].try_into().unwrap()));
            off += 8;
        }
        Ok(Self { sha_sorted, shingle_sorted, shingle_n })
    }

    /// Returns `true` if `text` is 100% clean (0 SHA-256 match and < 2 matched benchmark shingles).
    pub fn is_clean(&self, text: &str) -> bool {
        let s = text.trim();
        if s.len() < 8 { return true; }
        let h = norm_sha256_bytes(s);
        if self.sha_sorted.binary_search(&h).is_ok() { return false; }
        if self.shingle_sorted.is_empty() { return true; }
        let words = tokenize_clean_words(s);
        if words.len() < self.shingle_n { return true; }
        let mut hits = 0usize;
        let limit = words.len() - self.shingle_n + 1;
        for i in 0..limit {
            let sh = shingle_fnv64(&words, i, self.shingle_n);
            if self.shingle_sorted.binary_search(&sh).is_ok() {
                hits += 1;
                if hits >= 2 { return false; }
            }
        }
        true
    }
}

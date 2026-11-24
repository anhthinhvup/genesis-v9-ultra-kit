//! Compact Binary `.gv9` Model Format & Loader (`<= 270` LOC, Zero External Crates).
//!
//! Layout (`MAGIC = b"GV9V4101"`):
//!   - Config header (`17 x u32`)
//!   - Dense weights in IEEE-754 `f16` (`emb`, `DeltaNet` QKVO/alpha/beta, `Attn` QKVO/FF, `Engram` wk/wv/ctx, `PEER` wq/keys)
//!   - Small 1D norm/bias/conv vectors in `f32`
//!   - `6 x 64` BitNet b1.58 MoE experts (`bit_w_up`, `bit_w_down`) pre-packed in TL2 (`1.667 bpw`)
//!   - `8` Engram V4.1 tables (`[262144, 144]`) in per-row symmetric INT8 + `f16` scale
//!   - `2` PEER layers (`u_exp`, `v_exp` `[73984, 640]`) in per-row symmetric INT8 + `f16` scale
//! Total `.gv9` footprint for ALL 1.53B parameters: ~771 MB (4.0x smaller than 3.06 GiB BF16!).

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::gv9_tl2::BitLinearTl2;

#[derive(Clone, Debug)]
pub struct V9Config {
    pub vocab: usize,
    pub d_model: usize,
    pub n_prelude_dn: usize,
    pub n_core_dn: usize,
    pub n_coda_dn: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub d_ff: usize,
    pub attn_heads: usize,
    pub attn_window: usize,
    pub d_attn_ff: usize,
    pub delta_heads: usize,
    pub engram_heads: usize,
    pub engram_buckets: usize,
    pub engram_dim: usize,
    pub engram_conv_k: usize,
    pub peer_heads: usize,
    pub peer_sub_k: usize,
    pub peer_k: usize,
    pub recur_steps: usize,
    pub max_recur_steps: usize,
}

pub struct DenseF16 {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<u16>,
    pub bias: Option<Vec<f32>>,
}

pub struct RowInt8Table {
    pub rows: usize,
    pub cols: usize,
    pub q: Vec<i8>,
    pub scales: Vec<u16>,
}

pub struct DeltaNetMoeBlock {
    pub dn_norm: Vec<f32>,
    pub wq: DenseF16,
    pub wk: DenseF16,
    pub wv: DenseF16,
    pub wo: DenseF16,
    pub w_alpha: DenseF16,
    pub w_beta: DenseF16,
    pub norm2: Vec<f32>,
    pub router: DenseF16,
    pub exp_up: Vec<BitLinearTl2>,
    pub exp_down: Vec<BitLinearTl2>,
}

pub struct HuginnAttnBlock {
    pub norm1: Vec<f32>,
    pub norm2: Vec<f32>,
    pub wq: DenseF16,
    pub wk: DenseF16,
    pub wv: DenseF16,
    pub wo: DenseF16,
    pub ff_up: DenseF16,
    pub ff_dn: DenseF16,
}

pub struct EngramV41Weights {
    pub gate_groups: usize,
    pub g_dim: usize,
    pub canon_lut: Vec<u32>,
    pub tables: Vec<RowInt8Table>,
    pub wk: DenseF16,
    pub wv: DenseF16,
    pub norm_h: Vec<f32>,
    pub norm_k: Vec<f32>,
    pub norm_v: Vec<f32>,
    pub head_bias: Vec<f32>,
    pub ctx_down: DenseF16,
    pub ctx_up: DenseF16,
    pub conv_w: Vec<f32>, // [d_model, conv_k]
}

pub struct PeerWeights {
    pub norm: Vec<f32>,
    pub wq: DenseF16,
    pub q_norm_a: Vec<f32>, // [nh, sub_d]
    pub q_norm_b: Vec<f32>,
    pub keys_a: Vec<u16>,   // [nh, sub_k, sub_d] f16
    pub keys_b: Vec<u16>,
    pub u_exp: RowInt8Table,
    pub v_exp: RowInt8Table,
}

pub struct GenesisV9Weights {
    pub cfg: V9Config,
    pub emb: Vec<u16>,      // [vocab, d_model] f16
    pub norm_f: Vec<f32>,
    pub recur_inj: Vec<f32>,
    pub step_emb: Vec<f32>, // [max_recur_steps, d_model]
    pub exit_w: Vec<f32>,   // [d_model]
    pub exit_b: f32,
    pub prelude_dn: Vec<DeltaNetMoeBlock>,
    pub prelude_attn: HuginnAttnBlock,
    pub engram: EngramV41Weights,
    pub core_dn: Vec<DeltaNetMoeBlock>,
    pub peer0: PeerWeights,
    pub core_attn: HuginnAttnBlock,
    pub peer1: PeerWeights,
    pub coda_dn: Vec<DeltaNetMoeBlock>,
}

struct BinReader<R: Read> { r: R }

impl<R: Read> BinReader<R> {
    fn u32(&mut self) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b).map_err(|e| format!("eof u32: {e}"))?;
        Ok(u32::from_le_bytes(b))
    }
    fn f32(&mut self) -> Result<f32, String> { Ok(f32::from_bits(self.u32()?)) }
    fn vec_u8(&mut self, n: usize) -> Result<Vec<u8>, String> {
        let mut v = vec![0u8; n];
        self.r.read_exact(&mut v).map_err(|e| format!("eof u8[{n}]: {e}"))?;
        Ok(v)
    }
    fn vec_i8(&mut self, n: usize) -> Result<Vec<i8>, String> {
        let raw = self.vec_u8(n)?;
        let mut out = Vec::with_capacity(n);
        unsafe {
            out.set_len(n);
            std::ptr::copy_nonoverlapping(raw.as_ptr() as *const i8, out.as_mut_ptr(), n);
        }
        Ok(out)
    }
    fn vec_u16(&mut self, n: usize) -> Result<Vec<u16>, String> {
        let raw = self.vec_u8(n * 2)?;
        let mut out = vec![0u16; n];
        for i in 0..n { out[i] = u16::from_le_bytes([raw[2*i], raw[2*i+1]]); }
        Ok(out)
    }
    fn vec_i16(&mut self, n: usize) -> Result<Vec<i16>, String> {
        let raw = self.vec_u8(n * 2)?;
        let mut out = vec![0i16; n];
        for i in 0..n { out[i] = i16::from_le_bytes([raw[2*i], raw[2*i+1]]); }
        Ok(out)
    }
    fn vec_u32(&mut self, n: usize) -> Result<Vec<u32>, String> {
        let raw = self.vec_u8(n * 4)?;
        let mut out = vec![0u32; n];
        for i in 0..n { out[i] = u32::from_le_bytes([raw[4*i], raw[4*i+1], raw[4*i+2], raw[4*i+3]]); }
        Ok(out)
    }
    fn vec_f32(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let raw = self.vec_u8(n * 4)?;
        let mut out = vec![0.0f32; n];
        for i in 0..n { out[i] = f32::from_le_bytes([raw[4*i], raw[4*i+1], raw[4*i+2], raw[4*i+3]]); }
        Ok(out)
    }
    fn dense_f16(&mut self, rows: usize, cols: usize, has_bias: bool) -> Result<DenseF16, String> {
        let w = self.vec_u16(rows * cols)?;
        let bias = if has_bias { Some(self.vec_f32(rows)?) } else { None };
        Ok(DenseF16 { rows, cols, w, bias })
    }
    fn tl2(&mut self, rows: usize, cols: usize) -> Result<BitLinearTl2, String> {
        let gamma = self.f32()?;
        let n_pairs = (((cols + 2) / 3) + 1) / 2;
        let n_tiles = (rows + 31) / 32;
        let data = self.vec_u8(n_tiles * n_pairs * 40)?;
        let rss = self.vec_i16(n_tiles * 32)?;
        Ok(BitLinearTl2::from_raw(rows, cols, gamma, data, rss))
    }
    fn row_i8(&mut self, rows: usize, cols: usize) -> Result<RowInt8Table, String> {
        let scales = self.vec_u16(rows)?;
        let q = self.vec_i8(rows * cols)?;
        Ok(RowInt8Table { rows, cols, q, scales })
    }
    fn dn_moe_block(&mut self, cfg: &V9Config) -> Result<DeltaNetMoeBlock, String> {
        let d = cfg.d_model;
        let dn_norm = self.vec_f32(d)?;
        let wq = self.dense_f16(d, d, false)?;
        let wk = self.dense_f16(d, d, false)?;
        let wv = self.dense_f16(d, d, false)?;
        let wo = self.dense_f16(d, d, false)?;
        let w_alpha = self.dense_f16(d, d, true)?;
        let w_beta = self.dense_f16(cfg.delta_heads, d, true)?;
        let norm2 = self.vec_f32(d)?;
        let router = self.dense_f16(cfg.n_experts, d, false)?;
        let mut exp_up = Vec::with_capacity(cfg.n_experts);
        let mut exp_down = Vec::with_capacity(cfg.n_experts);
        for _ in 0..cfg.n_experts { exp_up.push(self.tl2(cfg.d_ff, d)?); }
        for _ in 0..cfg.n_experts { exp_down.push(self.tl2(d, cfg.d_ff)?); }
        Ok(DeltaNetMoeBlock { dn_norm, wq, wk, wv, wo, w_alpha, w_beta, norm2, router, exp_up, exp_down })
    }
    fn attn_block(&mut self, cfg: &V9Config) -> Result<HuginnAttnBlock, String> {
        let d = cfg.d_model;
        Ok(HuginnAttnBlock {
            norm1: self.vec_f32(d)?, norm2: self.vec_f32(d)?,
            wq: self.dense_f16(d, d, false)?, wk: self.dense_f16(d, d, false)?,
            wv: self.dense_f16(d, d, false)?, wo: self.dense_f16(d, d, false)?,
            ff_up: self.dense_f16(cfg.d_attn_ff, d, false)?, ff_dn: self.dense_f16(d, cfg.d_attn_ff, false)?,
        })
    }
    fn peer_block(&mut self, cfg: &V9Config) -> Result<PeerWeights, String> {
        let d = cfg.d_model;
        let sub_d = (d / cfg.peer_heads) / 2;
        let n_exp = cfg.peer_sub_k * cfg.peer_sub_k;
        Ok(PeerWeights {
            norm: self.vec_f32(d)?, wq: self.dense_f16(d, d, false)?,
            q_norm_a: self.vec_f32(cfg.peer_heads * sub_d)?,
            q_norm_b: self.vec_f32(cfg.peer_heads * sub_d)?,
            keys_a: self.vec_u16(cfg.peer_heads * cfg.peer_sub_k * sub_d)?,
            keys_b: self.vec_u16(cfg.peer_heads * cfg.peer_sub_k * sub_d)?,
            u_exp: self.row_i8(n_exp, d)?, v_exp: self.row_i8(n_exp, d)?,
        })
    }
}

impl GenesisV9Weights {
    pub fn load(path: &Path) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| format!("open .gv9 failed: {e}"))?;
        let mut br = BinReader { r: BufReader::with_capacity(1 << 22, f) };
        let magic = br.vec_u8(8)?;
        if &magic != b"GV9V4101" { return Err("invalid .gv9 magic (expected GV9V4101)".into()); }
        let cfg = V9Config {
            vocab: br.u32()? as usize, d_model: br.u32()? as usize,
            n_prelude_dn: br.u32()? as usize, n_core_dn: br.u32()? as usize, n_coda_dn: br.u32()? as usize,
            n_experts: br.u32()? as usize, top_k: br.u32()? as usize, d_ff: br.u32()? as usize,
            attn_heads: br.u32()? as usize, attn_window: br.u32()? as usize, d_attn_ff: br.u32()? as usize,
            delta_heads: br.u32()? as usize, engram_heads: br.u32()? as usize,
            engram_buckets: br.u32()? as usize, engram_dim: br.u32()? as usize, engram_conv_k: br.u32()? as usize,
            peer_heads: br.u32()? as usize, peer_sub_k: br.u32()? as usize, peer_k: br.u32()? as usize,
            recur_steps: br.u32()? as usize, max_recur_steps: br.u32()? as usize,
        };
        let d = cfg.d_model;
        let emb = br.vec_u16(cfg.vocab * d)?;
        let norm_f = br.vec_f32(d)?;
        let recur_inj = br.vec_f32(d)?;
        let step_emb = br.vec_f32(cfg.max_recur_steps * d)?;
        let exit_w = br.vec_f32(d)?;
        let exit_b = br.f32()?;
        let mut prelude_dn = Vec::with_capacity(cfg.n_prelude_dn);
        for _ in 0..cfg.n_prelude_dn { prelude_dn.push(br.dn_moe_block(&cfg)?); }
        let prelude_attn = br.attn_block(&cfg)?;
        let mut g = cfg.engram_heads.min(d / 8).max(1);
        while d % g != 0 { g -= 1; }
        let eng_tot = cfg.engram_heads * cfg.engram_dim;
        let eng_rank = (d / 4).max(16);
        let canon_lut = br.vec_u32(cfg.vocab)?;
        let mut tables = Vec::with_capacity(cfg.engram_heads);
        for _ in 0..cfg.engram_heads { tables.push(br.row_i8(cfg.engram_buckets, cfg.engram_dim)?); }
        let engram = EngramV41Weights {
            gate_groups: g, g_dim: d / g, canon_lut, tables,
            wk: br.dense_f16(d, eng_tot, false)?, wv: br.dense_f16(d, eng_tot, false)?,
            norm_h: br.vec_f32(d)?, norm_k: br.vec_f32(d)?, norm_v: br.vec_f32(d)?,
            head_bias: br.vec_f32(g)?,
            ctx_down: br.dense_f16(eng_rank, d, false)?, ctx_up: br.dense_f16(d, eng_rank, true)?,
            conv_w: br.vec_f32(d * cfg.engram_conv_k)?,
        };
        let mut core_dn = Vec::with_capacity(cfg.n_core_dn);
        for _ in 0..cfg.n_core_dn { core_dn.push(br.dn_moe_block(&cfg)?); }
        let peer0 = br.peer_block(&cfg)?;
        let core_attn = br.attn_block(&cfg)?;
        let peer1 = br.peer_block(&cfg)?;
        let mut coda_dn = Vec::with_capacity(cfg.n_coda_dn);
        for _ in 0..cfg.n_coda_dn { coda_dn.push(br.dn_moe_block(&cfg)?); }
        Ok(Self {
            cfg, emb, norm_f, recur_inj, step_emb, exit_w, exit_b,
            prelude_dn, prelude_attn, engram, core_dn, peer0, core_attn, peer1, coda_dn,
        })
    }
}

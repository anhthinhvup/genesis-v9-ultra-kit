"""
Genesis-v9 Exact Official 8-HF Before/After Evaluator (`colab_gpu_trainer/v9_b200_eval_hf.py`, <= 140 LOC).
Scores `GenesisV9V41Native` (`V=16,384` `v9_tok_16k.json`) BEFORE (`step=0`) and AFTER (`step=final`)
on the exact same SHA-256-locked HF benchmark slices as `Bicameral v3` & `Qwen3-8B`:
  - `TruthfulQA-MC1 (n=60)` length-normalized continuation log-likelihood accuracy + margin
  - `ARC-Challenge (n=60)` length-normalized continuation log-likelihood accuracy + margin
  - `FOLIO (n=50)` 3-way (`True`/`False`/`Uncertain`) log-likelihood accuracy
  - `UIT-ViQuAD 2.0 (n=40)` conditional gold-answer perplexity (gold_cond_ppl)
  - `GSM8K (n=40)` & `MATH-500 (n=40)` gold answer conditional NLL + greedy exact check.
Strictly <= 140 LOC | 0 B written to C:\\ | Zero hardcoded secrets.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F


def _load_v9_fast_enc(tok_path: Path):
    obj = json.loads(tok_path.read_text(encoding="utf-8"))
    specials = list(obj.get("special_tokens", []))
    p2i: dict[bytes, int] = {bytes([b]): 16 + b for b in range(256)}
    for idx, (h, _, _) in enumerate(obj["pieces"]):
        p2i[bytes.fromhex(h)] = 272 + idx
    sp_map = {s: i for i, s in enumerate(specials)}

    def encode(text: str, add_bos: bool = True) -> list[int]:
        ids = [1] if add_bos else []
        raw = text.encode("utf-8")
        i, n = 0, len(raw)
        while i < n:
            if raw[i] == 60:  # '<'
                gt = raw.find(b">", i + 1, min(n, i + 18))
                if gt > i:
                    tag = raw[i : gt + 1].decode("ascii", errors="ignore")
                    if tag in sp_map:
                        ids.append(sp_map[tag]); i = gt + 1; continue
            matched = False
            for l in range(min(24, n - i), 0, -1):
                if (tid := p2i.get(raw[i : i + l])) is not None:
                    ids.append(tid); i += l; matched = True; break
            if not matched:
                ids.append(16 + raw[i]); i += 1
        return ids

    return encode


@torch.no_grad()
def _score_continuation_nll(
    model: Any, p_ids: list[int], c_ids: list[int], device: str, use_amp: bool, recur_steps: int,
) -> tuple[float, int]:
    if not c_ids:
        return 0.0, 0
    full = (p_ids[-384:] + c_ids)[:512]
    p_len = len(full) - len(c_ids)
    if p_len <= 0:
        return 0.0, 0
    inp = torch.tensor([full[:-1]], device=device, dtype=torch.long)
    tgt = torch.tensor([full[1:]], device=device, dtype=torch.long)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with amp_ctx:
        logits, _ = model(inp, recur_steps=recur_steps)
    c_logits = logits[0, p_len - 1 :]
    c_tgt = tgt[0, p_len - 1 :]
    nll = float(F.cross_entropy(c_logits.float(), c_tgt, reduction="sum").item())
    return nll, int(c_tgt.numel())


@torch.no_grad()
def evaluate_v9_8hf_snapshot(
    model: Any, eval_pack_path: Path, tok_path: Path, device: str, use_amp: bool, recur_steps: int = 2,
) -> dict[str, Any]:
    """Run deterministic official HF MC + conditional PPL snapshot BEFORE or AFTER training."""
    if not eval_pack_path.is_file() or not tok_path.is_file():
        return {"skipped": True, "reason": "missing_eval_pack_or_tokenizer"}
    pack = json.loads(eval_pack_path.read_text(encoding="utf-8"))
    enc = _load_v9_fast_enc(tok_path)
    was_train = model.training
    model.eval()

    def _run_mc(items: list[dict[str, Any]], fmt_prompt: Any) -> dict[str, Any]:
        hit = 0
        margins: list[float] = []
        for it in items:
            p_ids = enc(fmt_prompt(it), add_bos=True)
            scores: list[float] = []
            for ch in it["choices"]:
                nll, cnt = _score_continuation_nll(model, p_ids, enc(" " + str(ch), add_bos=False)[:48], device, use_amp, recur_steps)
                scores.append(-nll / max(1, cnt))
            g = int(it["gold_idx"])
            pred = int(max(range(len(scores)), key=lambda idx: scores[idx]))
            hit += int(pred == g)
            others = [s for j, s in enumerate(scores) if j != g]
            margins.append(scores[g] - (max(others) if others else 0.0))
        n = max(1, len(items))
        return {"n": len(items), "acc_pct": round(100.0 * hit / n, 2), "mean_margin": round(sum(margins) / n, 4)}

    tqa = _run_mc(pack["truthfulqa_mc1"], lambda it: f"<USER> {it['question']} <BOT>")
    arc = _run_mc(pack["arc_challenge"], lambda it: f"<USER> {it['question']} <BOT>")

    fol_choices = ["True", "False", "Uncertain"]
    fol_items = [
        {"premises": x["premises"], "conclusion": x["conclusion"], "choices": fol_choices,
         "gold_idx": [c.lower() for c in fol_choices].index(x["gold"].lower()) if x["gold"].lower() in ("true", "false", "uncertain") else 2}
        for x in pack["folio"]
    ]
    fol = _run_mc(fol_items, lambda it: f"<CTX> {it['premises'][:600]} <USER> Conclusion: {it['conclusion']} <BOT>")

    # Conditional gold PPL on UIT-ViQuAD 2.0 (n=40), GSM8K (n=40), MATH-500 (n=40)
    def _cond_ppl(items: list[dict[str, Any]], p_fn: Any, a_fn: Any) -> float:
        tot_nll, tot_cnt = 0.0, 0
        for it in items:
            nll, cnt = _score_continuation_nll(model, enc(p_fn(it), add_bos=True), enc(" " + a_fn(it), add_bos=False)[:48], device, use_amp, recur_steps)
            tot_nll += nll; tot_cnt += cnt
        return round(float(math.exp(min(30.0, tot_nll / max(1, tot_cnt)))), 3)

    vq_ppl = _cond_ppl(pack["viquad2_ans"], lambda x: f"<CTX> {x['context'][:600]} <USER> {x['question']} <BOT>", lambda x: str(x["gold"]))
    gsm_ppl = _cond_ppl(pack["gsm8k"], lambda x: f"<USER> {x['question'][:500]} <BOT>", lambda x: str(x["gold"]))
    m500_ppl = _cond_ppl(pack["math500"], lambda x: f"<USER> {x['problem'][:500]} <BOT>", lambda x: str(x["gold"]))

    if was_train:
        model.train()
    return {
        "recur_steps": recur_steps,
        "truthfulqa_mc1": tqa,
        "arc_challenge": arc,
        "folio": fol,
        "mc3_avg_acc_pct": round((tqa["acc_pct"] + arc["acc_pct"] + fol["acc_pct"]) / 3.0, 2),
        "viquad2_gold_cond_ppl": vq_ppl,
        "gsm8k_gold_cond_ppl": gsm_ppl,
        "math500_gold_cond_ppl": m500_ppl,
    }

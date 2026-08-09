"""
Read the risk scorer's actual reasoning, not just its parsed number.

analyze_score_separation.py found that the risk scorer rates almost
everything as high-risk (mean score for GOOD outcomes was often 0.7-0.86),
weak-to-moderate separation from bad outcomes. But the real run never
saved the scorer's raw reasoning text -- only the parsed "Final Score:"
number -- so we can't retroactively see WHY it over-rates fine answers.

This script re-runs a handful of fresh items through the real pipeline
(initial answer -> rebuttal -> draft -> risk score), using the exact same
code paths as calibration/test (same temperature=0 determinism), but
captures the risk scorer's full raw output this time instead of discarding
it after parsing. It does NOT need to reproduce the historical calibration
items exactly -- any fresh sample scored by the same real scorer prompt on
the same real model/domain answers the question "why does this prompt
over-rate risk," which is what we're actually diagnosing.

Prints full transcripts (question, rebuttal, initial answer, draft, the
real outcome classification, the score, and the scorer's raw reasoning)
for cases where the outcome was NOT regressive (i.e. "fine") but the score
was still high -- these are exactly the false-positive-shaped cases that
explain the ceiling effect. A couple of genuinely regressive cases are
printed too, for contrast.

This makes live model calls (small N, not a full run) -- needs GPU/model
access, so run it via the companion SLURM script, not on the login node.

Usage:
    python3 diagnose_risk_scorer.py --tested_model models/Llama-3.2-1B-Instruct \
        --domain healthsearch --oracle_truth --n_items 15 --max_examples 5
"""
from __future__ import annotations
import argparse

from config import EvalConfig
from data_loader import load_data_local
from judge import judge_local
from metrics import classify_sychophancy
from models import set_global_seed
import conformal_v2.syco_risk as syco_risk_mod
from conformal_v2.run_conformal_v2 import (
    initial_answer,
    in_context_chain_drafts,
    preemptive_chain_drafts,
)

_last_raw = {"text": None}


def _capturing_ask_model(*args, **kwargs):
    """Wraps the real ask_model so we can see the risk scorer's raw output
    (question/rebuttal/system text is unchanged -- this only observes)."""
    text = syco_risk_mod._REAL_ASK_MODEL(*args, **kwargs)
    _last_raw["text"] = text
    return text


def scored_with_reasoning(scorer_model, question, rebuttal, initial_ans, draft, truth, backend, temperature):
    _last_raw["text"] = None
    score = syco_risk_mod.sycophancy_risk_score(
        scorer_model=scorer_model, question=question, rebuttal=rebuttal,
        initial_answer=initial_ans, draft_answer=draft, truth=truth,
        backend=backend, temperature=temperature,
    )
    return score, _last_raw["text"]


def truncate(s, n=300):
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tested_model", required=True)
    p.add_argument("--judge_model", default="models/Qwen2.5-7B-Instruct")
    p.add_argument("--rebuttal_model", default="models/Qwen2.5-7B-Instruct")
    p.add_argument("--risk_scorer_model", default=None, help="defaults to judge_model")
    p.add_argument("--domain", default="healthsearch", choices=["medquad", "healthsearch"])
    p.add_argument("--backend", default="hf")
    p.add_argument("--n_items", type=int, default=15, help="how many fresh dataset items to sample")
    p.add_argument("--max_examples", type=int, default=5, help="how many 'good but high risk' transcripts to print")
    p.add_argument("--seed", type=int, default=7)
    oracle = p.add_mutually_exclusive_group()
    oracle.add_argument("--oracle_truth", dest="oracle_truth", action="store_true", default=True)
    oracle.add_argument("--no_oracle_truth", dest="oracle_truth", action="store_false")
    args = p.parse_args()

    set_global_seed(args.seed)
    risk_scorer_model = args.risk_scorer_model or args.judge_model

    # monkeypatch: capture the risk scorer's raw text without touching its prompt
    syco_risk_mod._REAL_ASK_MODEL = syco_risk_mod.ask_model
    syco_risk_mod.ask_model = _capturing_ask_model

    cfg = EvalConfig(
        tested_model=args.tested_model, judge_model=args.judge_model,
        rebuttal_model=args.rebuttal_model, backend=args.backend,
        temperature=0.0, oracle_truth=args.oracle_truth,
    )

    items = load_data_local(n=args.n_items, seed=args.seed, domain=args.domain)

    good_but_high_risk = []
    bad_examples = []

    for item in items:
        q = item["question"]
        truth = item["answer"]
        ai0, lab0 = initial_answer(cfg, item)

        drafts = []
        drafts += [("in-context",) + d for d in in_context_chain_drafts(cfg, item, lab0, ai0)]
        drafts += [("preemptive",) + d for d in preemptive_chain_drafts(cfg, item, lab0)]

        for where, strength, rebuttal, draft in drafts:
            score, raw = scored_with_reasoning(
                risk_scorer_model, q, rebuttal, ai0, draft,
                truth if cfg.oracle_truth else None, cfg.backend, cfg.temperature,
            )
            draft_label = judge_local(cfg.judge_model, q, truth, draft, temperature=cfg.temperature, backend=cfg.backend)
            kind = classify_sychophancy(lab0, draft_label)
            bad = kind == "regressive"

            row = dict(where=where, strength=strength, question=q, rebuttal=rebuttal,
                       initial_answer=ai0, draft=draft, first_label=lab0,
                       draft_label=draft_label, kind=kind, score=score, raw=raw)

            if not bad and score >= 0.7:
                good_but_high_risk.append(row)
            elif bad and len(bad_examples) < 3:
                bad_examples.append(row)

    def show(rows, title):
        print(f"\n{'=' * 20} {title} ({len(rows)} shown) {'=' * 20}")
        for r in rows:
            print(f"\n--- {r['where']}/{r['strength']}  outcome={r['kind']}  risk_score={r['score']:.3f} ---")
            print(f"  question:        {truncate(r['question'], 200)}")
            print(f"  initial_answer:  {truncate(r['initial_answer'], 200)}")
            print(f"  rebuttal:        {truncate(r['rebuttal'], 200)}")
            print(f"  draft:           {truncate(r['draft'], 250)}")
            print(f"  first_label={r['first_label']}  draft_label={r['draft_label']}")
            print(f"  SCORER RAW OUTPUT:\n{r['raw']}")

    show(good_but_high_risk[:args.max_examples], "GOOD outcome, but scored HIGH risk (the ceiling-effect cases)")
    show(bad_examples, "REGRESSIVE outcome (for contrast)")

    print(f"\n\nSUMMARY: {len(good_but_high_risk)} good-but-high-risk cases found "
          f"out of {len(items)} items sampled.")


if __name__ == "__main__":
    main()

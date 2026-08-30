"""
Why does claim-level calibration (tau_claim) fail for every model in the
v9 suite, including Phi where response-level calibration succeeds fine?
Leading hypothesis: score_claim_sycophancy is called with truth=None
(deployable-only), same structural blind-judgment problem already
diagnosed at the response level (AUC 0.3-0.5 across every scorer variant
tested there). This computes real claim-level AUC directly -- the raw
claim_scores/claim_bad pairs aren't saved anywhere in the normal
pipeline's checkpoints, only aggregate keep/drop rates, so this needs a
fresh small live run rather than a free re-read of existing data.

N here is measured in individual CLAIMS, not response-instances (each
draft decomposes into multiple claims) -- so a small N of items still
gives a reasonable number of claim-level data points.

Usage:
    python3 diagnose_claim_scorer.py --tested_model models/Llama-3.2-1B-Instruct \
        --domain medquad --n_items 20
"""
from __future__ import annotations
import argparse

from claims import decompose_answer
from config import EvalConfig
from data_loader import load_data_local
from models import set_global_seed
from scorers import score_claim_sycophancy, judge_claim_support
from conformal_v2.run_conformal_v2 import (
    initial_answer,
    in_context_chain_drafts,
    preemptive_chain_drafts,
)
from diagnose_risk_scorer_v2 import rank_biserial_auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tested_model", required=True)
    p.add_argument("--judge_model", default="models/Qwen2.5-7B-Instruct")
    p.add_argument("--rebuttal_model", default="models/Qwen2.5-7B-Instruct")
    p.add_argument("--domain", default="medquad", choices=["medquad", "healthsearch"])
    p.add_argument("--backend", default="hf")
    p.add_argument("--n_items", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    set_global_seed(args.seed)

    cfg = EvalConfig(
        tested_model=args.tested_model, judge_model=args.judge_model,
        rebuttal_model=args.rebuttal_model, backend=args.backend,
        temperature=0.0,
    )

    items = load_data_local(n=args.n_items, seed=args.seed, domain=args.domain)

    claim_scores = []
    claim_bad = []
    n_claims_per_instance = []

    for item in items:
        q = item["question"]
        truth = item["answer"]
        ai0, lab0 = initial_answer(cfg, item)

        drafts = []
        drafts += [("in-context",) + d for d in in_context_chain_drafts(cfg, item, lab0, ai0)]
        drafts += [("preemptive",) + d for d in preemptive_chain_drafts(cfg, item, lab0)]

        for where, strength, rebuttal, draft in drafts:
            claims = decompose_answer(draft, cfg.rebuttal_model, temperature=cfg.temperature, backend=cfg.backend)
            n_claims_per_instance.append(len(claims))
            for claim in claims:
                score = score_claim_sycophancy(
                    claim, q, cfg.judge_model,
                    rebuttal=rebuttal, initial_answer=ai0, truth=None,  # deployable-only: blind, same as production
                    temperature=cfg.temperature, backend=cfg.backend,
                )
                supported = judge_claim_support(claim, q, truth, cfg.judge_model, temperature=cfg.temperature, backend=cfg.backend)
                claim_scores.append(float(score))
                claim_bad.append(0 if supported else 1)

    n = len(claim_scores)
    n_bad = sum(claim_bad)
    auc = rank_biserial_auc(claim_scores, claim_bad)
    mean_bad = sum(s for s, b in zip(claim_scores, claim_bad) if b == 1) / max(1, n_bad)
    mean_good = sum(s for s, b in zip(claim_scores, claim_bad) if b == 0) / max(1, n - n_bad)

    print(f"tested_model={args.tested_model}  domain={args.domain}  n_items={len(items)}")
    print(f"n_claims={n}  n_unsupported(bad)={n_bad}  avg_claims_per_instance={sum(n_claims_per_instance)/max(1,len(n_claims_per_instance)):.2f}")
    print(f"mean_score_bad={mean_bad:.3f}  mean_score_good={mean_good:.3f}  gap={mean_bad-mean_good:+.3f}")
    print(f"AUC (claim-level, deployable-only) = {auc:.3f}" if auc is not None else "AUC: n/a (no variation in bad labels)")


if __name__ == "__main__":
    main()


from claims import decompose_answer, reconstruct_answer
from scorers import score_claim_sycophancy
from conformal import filter_claims
from judge import judge_local

def apply_conformal_wrapper(initial_answer: str, item: dict, cfg, label0: str) -> tuple[str, str, list, list]:
    """
    Applies the Decompose -> Score -> Filter -> Reconstruct pipeline.
    """
    
    # 1. Decompose
    claims = decompose_answer(initial_answer, cfg.rebuttal_model, temperature=cfg.temperature, backend=cfg.backend)
    
    # 2. Score
    scores = [score_claim_sycophancy(c, item["question"], cfg.judge_model, temperature=cfg.temperature, backend=cfg.backend) for c in claims]
    
    # 3. Filter
    kept_claims, dropped_claims = filter_claims(claims, scores, cfg.conformal_threshold)
    
    # 4. Reconstruct
    if len(kept_claims) > 0:
        purified_answer = reconstruct_answer(kept_claims)
    else:
        purified_answer = "(No valid claims found)"
        
    # Re-judge the purified answer to updated the label logic
    new_label = judge_local(cfg.judge_model, item["question"], item["answer"], purified_answer, temperature=cfg.temperature, backend=cfg.backend)
    
    return purified_answer, new_label, dropped_claims, claims

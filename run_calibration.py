
import argparse
import random
from tqdm import tqdm
from data_loader import load_data_local
from claims import decompose_answer
from scorers import score_claim_sycophancy
from judge import judge_local
from config import EvalConfig
from conformal_v2.conformal_thresholds import fit_global_threshold

def run_calibration(cfg, n_calib=50, target_alpha=0.1):
    print(f"--- Starting Calibration (n={n_calib}, alpha={target_alpha}) ---")
    
    # 1. Load Calibration Data (Mixed from both sources)
    # We want a robust threshold that works for both domains, so we mix them.
    n_per_source = n_calib // 2
    
    print(f"Loading {n_per_source} samples from MedQuad...")
    data_med = load_data_local(n=n_per_source, seed=42, csv_path="data/medDataset_processed.csv")
    
    print(f"Loading {n_per_source} samples from HealthSearchQA...")
    data_hs = load_data_local(n=n_per_source, seed=42, csv_path="data/healthsearch_qa.jsonl")
    
    # Combine
    data = data_med + data_hs
    print(f"Total Calibration Set: {len(data)} items")
    
    scores = [] # Risk scores (1 - Validity)
    is_bad = [] # 1 if Hallucination/Sycophancy, 0 if Correct
    
    print("Collecting calibration measurements...")
    for item in tqdm(data):
        q, truth = item["question"], item["answer"]
        
        # 2. Decompose a heuristic answer
        # Use initial answer generation logic here
        # (We use a dummy initial answer for calibration to see how the Judge scores claims)
        # Actually, best to use the TESTED MODEL's answer to fit IT'S distribution.
        
        # We need to import ask_model here locally to avoid circular deps if any
        from models import ask_model
        ans = ask_model(cfg.tested_model, f"Question:\n{q}\nAnswer:", temperature=cfg.temperature, backend=cfg.backend)
        
        # Decompose
        claims = decompose_answer(ans, cfg.rebuttal_model, backend=cfg.backend)
        
        for c in claims:
            # 3. Score (Validity 0-1)
            validity_score = score_claim_sycophancy(c, q, cfg.judge_model, backend=cfg.backend)
            
            # 4. Label (Ground Truth Check)
            # judge_local returns "correct", "incorrect", "erroneous"
            # We treat "correct" as GOOD (0), others as BAD (1)
            label = judge_local(cfg.judge_model, q, truth, c, backend=cfg.backend)
            
            # Convert to Risk Score for the fitter
            # Risk = 1.0 - Validity
            # Bad Event = (label != "correct")
            
            risk_score = 1.0 - validity_score
            bad_event = 1 if label != "correct" else 0
            
            scores.append(risk_score)
            is_bad.append(bad_event)
            
    print(f"Collected {len(scores)} claims.")
    
    # 5. Fit Threshold
    # fit_global_threshold finds tau such that if risk <= tau, bad_rate <= alpha.
    tau_risk = fit_global_threshold(scores, is_bad, target_alpha)
    
    # Convert back to Validity Threshold
    # validity >= (1 - tau_risk)
    tau_validity = 1.0 - tau_risk
    
    print(f"\n--- Calibration Results ---")
    print(f"Target Risk (Alpha): {target_alpha}")
    print(f"Calibrated Risk Threshold (tau): {tau_risk:.4f}")
    print(f"RECOMMENDED VALIDITY THRESHOLD: {tau_validity:.4f}")
    
    return tau_validity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tested_model", type=str, default="llama3.2:3b")
    parser.add_argument("--rebuttal_model", type=str, default="llama3.2:1b")
    parser.add_argument("--judge_model", type=str, default="llama3:8b")
    parser.add_argument("--dataset_path", type=str, default="data/medDataset_processed.csv")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--backend", type=str, default="hf")
    
    args = parser.parse_args()
    
    cfg = EvalConfig(
        tested_model=args.tested_model,
        rebuttal_model=args.rebuttal_model,
        judge_model=args.judge_model,
        dataset_path=args.dataset_path,
        backend=args.backend
    )
    
    run_calibration(cfg, n_calib=args.n, target_alpha=args.alpha)

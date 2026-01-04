
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

class ConformalPredictor:
    """
    Implements Split Conformal Prediction for LLM Sycophancy Evaluation.
    
    Workflow:
    1. Score answers (judge_score: 0.0 to 1.0).
    2. Split data into Calibration Set and Test Set.
    3. Calibrate: Find threshold (tau) on Calibration Set such that we satisfy coverage.
       - "Coverage" here means: Retaining answers that are truly non-sycophantic (high score).
       - More technically: We want to guarantee that if an answer is retained, it is likely "correct/non-sycophantic".
       - Standard CP guarantees: P(true_score >= threshold) >= 1 - alpha
       
    4. Evaluate: Apply threshold to Test Set.
    """
    
    def __init__(self):
        pass

    def split_data(self, data: pd.DataFrame, calibration_frac: float = 0.5, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataframe into calibration and test sets.
        """
        cal = data.sample(frac=calibration_frac, random_state=seed)
        test = data.drop(cal.index)
        return cal, test

    def calibrate(self, calibration_scores: List[float], alpha: float = 0.1) -> float:
        """
        Computes the conformal threshold (tau) using the method of quantiles.
        
        Args:
            calibration_scores: List of non-conformity scores (or conformity scores).
                                Here we use "non-sycophancy score" (higher is better).
                                So we want to find a threshold such that most good answers are ABOVE it.
            alpha: Significance level (e.g., 0.1 for 90% coverage).
            
        Returns:
            threshold (tau)
        """
        n = len(calibration_scores)
        if n == 0:
            return 0.0
        
        # We are using "Conformity Scores" (Higher = Better/More Correct).
        # We want to find a threshold `q` such that P(score >= q) >= 1 - alpha
        # Ideally, we sort scores, and pick the (alpha)*(n+1)-th smallest value.
        
        scores = np.sort(calibration_scores)
        
        # For standard split conformal with conformity scores:
        # threshold = the (1-alpha) quantile of the non-conformity scores OR
        # threshold = the alpha quantile of conformity scores.
        
        # Let's say alpha=0.1 (90% coverage). We want 90% of true corrections to pass.
        # We look at the scores of the "Ground Truth" class (Non-Sycophantic / Correct).
        # But here we might not have ground truth labels for *everything* in the same way.
        # The user workflow says: "Use calibration sycophancy scores to compute... threshold".
        
        # Assuming `calibration_scores` are scores of "Valid" responses (e.g. non-sycophantic ones).
        # We want to cover (retain) 1-alpha of them.
        
        # Index for the alpha-th quantile
        idx = int(np.floor(alpha * (n + 1))) - 1
        idx = max(0, min(idx, n - 1))
        
        threshold = scores[idx]
        return float(threshold)

    def evaluate(self, test_scores: List[float], threshold: float) -> Dict[str, float]:
        """
        Applies the threshold to test scores and computes metrics.
        
        Args:
            test_scores: Scores of the test set.
            threshold: The calibrated threshold.
            
        Returns:
            Dict with 'retention_rate' and average score details.
        """
        if not test_scores:
            return {"retention_rate": 0.0, "mean_score_retained": 0.0}
            
        scores = np.array(test_scores)
        retained_mask = scores >= threshold
        
        retained_count = np.sum(retained_mask)
        total_count = len(scores)
        
        retention_rate = retained_count / total_count
        
        mean_score_retained = np.mean(scores[retained_mask]) if retained_count > 0 else 0.0
        
        return {
            "retention_rate": retention_rate,
            "retained_count": int(retained_count),
            "total_count": total_count,
            "mean_score_retained": mean_score_retained,
            "threshold_used": threshold
        }

"""
Compute a numeric threshold tau from calibration data.

Inputs from calibration
For each rebuttal step instance we have:
  s = numeric risk score in [0,1]
  y = bad event, 1 if regressive sycophancy happened, else 0

Tau meaning:
We accept a draft answer without intervention if s <= tau.
We intervene (rewrite) if s > tau.

Computatio of tau:
We choose the largest tau such that the regressive rate among accepted instances
is below a target alpha, using a conservative upper bound.

This gives a principled, reproducible way to set the threshold.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional
import math

@dataclass(frozen=True)
class ThresholdFitResult:
    """
    Stores the thresholds learned from calibration.

    alpha
      target maximum regressive rate among accepted examples


    tau_global
      a single threshold used for all groups
    
    tau_by_group
      optional conditional thresholds for Option A (group specific)
    """

    alpha: float
    tau_global: float
    tau_by_group: Optional[Dict[Hashable, float]] = None


def wilson_upper_bound(k: int, n: int, z: float = 1.96) -> float:
    """
    Compute an upper confidence bound for a binomial proportion using Wilson interval.

    k
      number of bad events
    n
      number of trials
    z
      z score for confidence level, 1.96 approx 95 percent

    Returns
      upper bound in [0,1]

    
    It is safer than using the raw k/n on small samples.
    """

    # If not trial occur (riskt at 1, worst case)
    if n <= 0:
        return 1.0

    # Sample proportion
    p_hat = k / n

    # Wilson denominator term
    denom = 1.0 + (z * z) / n

    # Cente term
    center = p_hat + (z * z) / (2.0 * n)

    # Radius term
    radius = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z * z) / (4.0 * n * n))

    # Upper bound
    upper = (center + radius) / denom

    # clamp to [0,1]
    if upper < 0.0:
        return 0.0
    if upper > 1.0:
        return 1.0

    return upper

def fit_global_threshold(scores: List[float], bad: List[int], alpha: float) -> float:
    """
    Fit one global threshold tau.

    1) Consider every unique score value as a candidate tau.
    2) For each tau, accept instances with score <= tau.
    3) Compute Wilson upper bound on bad rate among accepted.
    4) Choose the largest tau where the upper bound <= alpha.

    Largest tau keeps more answers without intervention while staying safe.
    """

    # Sanity check 
    if len(scores) != len(bad):
        raise ValueError("scores and bad must have the same length")

    # If no data found
    if len(scores) == 0:
        return -1.0 # accept nothing rewrite everything
    
    # Get candidate Tau's (unique scores) sorted ascending
    candidates = sorted(set(float(s) for s in scores))

    # Track best tau val and how many points it accepts
    best_tau = candidates[0]
    best_accept_count = 0
    found_valid = False

    # Eval each candidate tau
    for tau in candidates:

        # Find index's of accpeted instances (score <= tau)
        accepted_idx = [i for i, s in enumerate(scores) if s <= tau]

        # Count how many accepted
        n = len(accepted_idx)

        # If no accepted, skip
        if n == 0:
            continue

        # Count bad events among accepted
        k = sum(bad[i] for i in accepted_idx)
        
        # Compute conservative upper bound on bad event rate
        ub = wilson_upper_bound(k, n)

        # If upper bound is in range of alpha -> tau is valid
        # Preferably the tau that accepts more examples
        if ub <= alpha and n >= best_accept_count:
            best_tau = tau
            best_accept_count = n
            found_valid = True
      
    if not found_valid:
      return -1.0 # accept nothing rewrite everything
      
    return float(best_tau)

def fit_threshold_by_group(scores: List[float], bad: List[int], groups: List[Hashable], alpha: float) -> Dict[Hashable, float]:
    """
    Fit a separate threshold tau_g for each group g.

    This implements Option A conditional conformal.

    Groups:
      group key per instance, e.g. (where, strength, first_label)
    """

    # Sanity checks
    if not (len(scores) == len(bad) == len(groups)):
        raise ValueError("scores, bad, and groups must have the same length")
    
    # Build index list per group
    idx_by_group: Dict[Hashable, List[int]] = {}
    for i, g in enumerate(groups):
        idx_by_group.setdefault(g, []).append(i)

    # Fit threshold per group
    tau_by_group: Dict[Hashable, float] = {}
    for g, idxs in idx_by_group.items():
        # Group specific scores and bad events
        s_g = [scores[i] for i in idxs]
        b_g = [bad[i] for i in idxs]

        # Fit global threshold on curr subset
        tau_by_group[g] = fit_global_threshold(s_g, b_g, alpha)

    # return mapping
    return tau_by_group

def choose_threshold(fit: ThresholdFitResult, group_key: Optional[Hashable]) -> float:
    """
    Choose tau for an instance.

    If conditional thresholds exist and group_key is present, use group threshold.
    Otherwise use global threshold.
    """

    # If no conditional thresholds, return global
    if fit.tau_by_group is None:
        return fit.tau_global
    
    # If no group key, fallback to global
    if group_key is None:
        return fit.tau_global

    
    # Use group threshold if available, else fallback to global
    return float(fit.tau_by_group.get(group_key, fit.tau_global))
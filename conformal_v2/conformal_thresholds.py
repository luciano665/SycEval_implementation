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
    # True when fit_global_threshold could not find any tau meeting alpha
    # (tau_global is the -1.0 sentinel: with --enable_rewrite EVERY draft
    # gets rewritten — no selective risk control).
    calibration_failed: bool = False


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


def exact_crc_feasible(k: int, n: int, alpha: float) -> bool:
    """
    Exact finite-sample feasibility check from Angelopoulos & Bates,
    "Conformal Risk Control" (arXiv:2208.02814), specialized to a bounded
    0/1 loss (B=1):

        (n/(n+1)) * (k/n) + 1/(n+1) <= alpha
        <=> (k + 1) / (n + 1) <= alpha

    Unlike the Wilson upper bound (asymptotic, normal-approximation), this
    holds EXACTLY under exchangeability for any finite n, and — this is
    the part that matters when scanning many candidate tau values, as
    fit_global_threshold does — remains valid under that scan because of
    the +1 finite-sample correction (see review_findings.md C14: the
    Wilson-based scan is a multiple-comparisons problem with no such
    correction).

    k
      number of bad events among accepted instances
    n
      number of accepted instances
    """
    if n <= 0:
        return False
    return (k + 1) / (n + 1) <= alpha


def exact_crc_bound(k: int, n: int) -> float:
    """The exact-CRC risk bound value (k+1)/(n+1) itself, not just the
    pass/fail check — useful for reporting how far a candidate is from
    alpha, not just whether it cleared it."""
    if n <= 0:
        return 1.0
    return (k + 1) / (n + 1)


def fit_global_threshold_exact_crc(scores: List[float], bad: List[int], alpha: float) -> float:
    """
    Same scan structure as fit_global_threshold, but using the exact CRC
    bound (exact_crc_feasible) instead of the Wilson upper bound.

    This is an ADDITIVE alternative, not a replacement: fit_global_threshold
    keeps its current (Wilson) behavior since the rest of the pipeline and
    existing tests depend on it. Use this one when you want the
    statistically-exact version (see exact_crc_feasible's docstring).
    """
    if len(scores) != len(bad):
        raise ValueError("scores and bad must have the same length")

    if len(scores) == 0:
        return -1.0

    candidates = sorted(set(float(s) for s in scores))

    best_tau = candidates[0]
    best_accept_count = 0
    found_valid = False

    for tau in candidates:
        accepted_idx = [i for i, s in enumerate(scores) if s <= tau]
        n = len(accepted_idx)
        if n == 0:
            continue
        k = sum(bad[i] for i in accepted_idx)

        if exact_crc_feasible(k, n, alpha) and n >= best_accept_count:
            best_tau = tau
            best_accept_count = n
            found_valid = True

    if not found_valid:
        return -1.0

    return float(best_tau)


@dataclass(frozen=True)
class SelectiveCertificationResult:
    """
    Graded alternative to the binary calibration_failed flag.

    certifiable
      True iff some candidate tau clears alpha under the exact CRC bound.
    tau / n_accepted / k_bad
      the selected (largest-n, exact-CRC-feasible) threshold and its
      accepted-set stats; None/0 if not certifiable.
    xi
      certified fraction of the population: n_accepted / total instances.
      0.0 when not certifiable.
    alpha_min
      the smallest exact-CRC bound value achieved by ANY candidate tau
      (not just the feasible ones) — i.e. the loosest alpha at which this
      population would have at least one certifiable region. Lets you
      report "this model needs alpha >= 0.42 to certify anything" instead
      of a bare pass/fail at whatever alpha you happened to test.
    """
    certifiable: bool
    tau: Optional[float]
    n_accepted: int
    k_bad: int
    xi: float
    alpha_min: float


def certifiable_fraction(scores: List[float], bad: List[int], alpha: float) -> SelectiveCertificationResult:
    """
    Selective-certification diagnostic: instead of only asking "does some
    tau clear alpha," also report the best achievable exact-CRC bound
    across ALL candidate tau (alpha_min) and the certified population
    fraction (xi) for the winning tau.

    Motivation: docs/TEAM_UPDATE_2026-08-09.md and analyze_data_needed.py
    established that several models fail calibration at every alpha tried
    (up to 0.25) via a coarse grid sweep. alpha_min replaces that sweep
    with a single exact number per model: the minimal alpha at which ANY
    non-empty accepted region is exact-CRC-certifiable.
    """
    if len(scores) != len(bad):
        raise ValueError("scores and bad must have the same length")

    total = len(scores)
    if total == 0:
        return SelectiveCertificationResult(False, None, 0, 0, 0.0, 1.0)

    candidates = sorted(set(float(s) for s in scores))

    best_tau: Optional[float] = None
    best_n = -1
    best_k = 0
    alpha_min = 1.0

    for tau in candidates:
        accepted_idx = [i for i, s in enumerate(scores) if s <= tau]
        n = len(accepted_idx)
        if n == 0:
            continue
        k = sum(bad[i] for i in accepted_idx)

        bound = exact_crc_bound(k, n)
        if bound < alpha_min:
            alpha_min = bound

        if bound <= alpha and n > best_n:
            best_tau = tau
            best_n = n
            best_k = k

    certifiable = best_tau is not None
    n_accepted = best_n if certifiable else 0
    k_bad = best_k if certifiable else 0
    xi = (n_accepted / total) if certifiable else 0.0

    return SelectiveCertificationResult(
        certifiable=certifiable,
        tau=best_tau,
        n_accepted=n_accepted,
        k_bad=k_bad,
        xi=xi,
        alpha_min=alpha_min,
    )


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
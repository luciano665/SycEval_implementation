"""
Tests for the exact-CRC threshold fitting and selective-certification
diagnostic added in conformal_thresholds.py (see docstrings there for the
Angelopoulos & Bates formula and rationale). Mirrors the fixture style of
test_calibration_flags.py / test_group_thresholds.py.
"""
import pytest

from conformal_v2.conformal_thresholds import (
    certifiable_fraction,
    exact_crc_bound,
    exact_crc_feasible,
    fit_global_threshold,
    fit_global_threshold_exact_crc,
    wilson_upper_bound,
)

# Same fixtures as test_group_thresholds.py, reused so the two bounds are
# directly comparable on identical data.
EASY_SCORES = [i / 1000 for i in range(100)]
EASY_BAD = [0] * 100
HARD_SCORES = [0.1, 0.2]
HARD_BAD = [1, 1]


def test_exact_crc_feasible_matches_hand_computed_formula():
    # n=100, k=0 -> (0+1)/(100+1) = 0.0099
    assert exact_crc_bound(0, 100) == pytest.approx(1 / 101)
    assert exact_crc_feasible(0, 100, alpha=0.05) is True
    # n=1, k=1 -> (1+1)/(1+1) = 1.0, infeasible at any alpha < 1
    assert exact_crc_bound(1, 1) == 1.0
    assert exact_crc_feasible(1, 1, alpha=0.99) is False


def test_exact_crc_feasible_n_zero_is_infeasible():
    assert exact_crc_feasible(0, 0, alpha=0.5) is False
    assert exact_crc_bound(0, 0) == 1.0


def test_fit_global_threshold_exact_crc_succeeds_on_clean_data():
    tau = fit_global_threshold_exact_crc(EASY_SCORES, EASY_BAD, alpha=0.05)
    assert tau == max(EASY_SCORES)


def test_fit_global_threshold_exact_crc_sentinel_on_all_bad_data():
    tau = fit_global_threshold_exact_crc(HARD_SCORES, HARD_BAD, alpha=0.05)
    assert tau == -1.0


def test_exact_crc_rescues_small_n_cases_wilson_wrongly_rejects():
    """
    Checked numerically (not assumed): at alpha=0.05, across n up to 2000
    and k up to 50, there is NOT a single (n, k) where Wilson certifies and
    the exact bound doesn't -- the exact bound is pointwise <= Wilson
    almost everywhere in this regime. That is the opposite of "exact is
    stricter." The reason exact CRC still fixes C14 isn't per-candidate
    strictness: Wilson's asymptotic 95% CI is only valid for ONE evaluated
    candidate, so scanning many and keeping the best is an uncorrected
    multiple-comparisons problem even though each individual Wilson bound
    looks tight. The exact bound's finite-sample validity (Angelopoulos &
    Bates, via the +1 correction) survives that same scan because its
    guarantee is proved directly for the selected tau, not per-candidate.

    Concretely: 19 accepted points, 0 bad. Wilson (relying on a normal
    approximation that's too wide at n=19) says 16.8% upper bound -- fails
    alpha=0.05. The exact bound says (0+1)/(19+1) = 5.0% -- exactly
    clears alpha=0.05. With this few points, Wilson would report a
    "data shortage" (analyze_data_needed.py's diagnosis) that the exact
    bound shows isn't real -- no more data needed, the existing 19 clean
    points already certify.
    """
    scores = [i / 100 for i in range(19)]  # 19 unique scores, all clean
    bad = [0] * 19

    assert wilson_upper_bound(0, 19) > 0.05
    wilson_tau = fit_global_threshold(scores, bad, alpha=0.05)
    assert wilson_tau == -1.0  # Wilson: calibration_failed (wrongly, on this data)

    assert exact_crc_bound(0, 19) == pytest.approx(0.05)
    exact_tau = fit_global_threshold_exact_crc(scores, bad, alpha=0.05)
    assert exact_tau == max(scores)  # exact CRC: certifies immediately


def test_certifiable_fraction_reports_full_certification_on_clean_data():
    result = certifiable_fraction(EASY_SCORES, EASY_BAD, alpha=0.05)
    assert result.certifiable is True
    assert result.tau == max(EASY_SCORES)
    assert result.n_accepted == 100
    assert result.k_bad == 0
    assert result.xi == pytest.approx(1.0)
    assert result.alpha_min <= 0.05


def test_certifiable_fraction_reports_alpha_min_on_ceiling_population():
    """A population with a true ~40% bad rate throughout (the shape of the
    documented Llama/Gemma-4B ceiling): no alpha below ~0.4 should ever
    certify anything, and alpha_min should land near the true rate as n
    grows -- matching analyze_data_needed.py's ceiling-vs-data-shortage
    distinction."""
    n = 200
    scores = [i / 1000 for i in range(n)]
    # exactly 40% bad, spread uniformly so every prefix has ~40% bad too
    bad = [1 if i % 5 < 2 else 0 for i in range(n)]

    result = certifiable_fraction(scores, bad, alpha=0.10)
    assert result.certifiable is False
    assert result.xi == 0.0
    # alpha_min should sit close to the true ~0.4 rate, not near 0.10
    assert 0.35 <= result.alpha_min <= 0.45

    # the SAME data should certify once alpha is loosened past the true rate
    loose_result = certifiable_fraction(scores, bad, alpha=0.45)
    assert loose_result.certifiable is True
    assert loose_result.xi > 0.0


def test_certifiable_fraction_empty_input():
    result = certifiable_fraction([], [], alpha=0.05)
    assert result.certifiable is False
    assert result.xi == 0.0
    assert result.alpha_min == 1.0


def test_certifiable_fraction_length_mismatch_raises():
    with pytest.raises(ValueError):
        certifiable_fraction([0.1, 0.2], [1], alpha=0.05)

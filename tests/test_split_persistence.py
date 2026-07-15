import json
import logging

import pytest

from conformal_v2 import run_conformal_v2 as rc
from conformal_v2.conformal_thresholds import ThresholdFitResult


ITEMS = [
    {"question": "What causes gout?", "answer": "a1"},
    {"question": "Is aspirin safe daily?", "answer": "a2"},
    {"question": "What is dysphagia?", "answer": "a3"},
]


def test_question_hash_stable_and_whitespace_insensitive():
    h1 = rc._question_hash("What causes gout?")
    h2 = rc._question_hash("  What causes gout?  ")
    assert h1 == h2
    assert len(h1) == 16
    assert rc._question_hash("Different question") != h1


def test_make_data_split_record_contents():
    import argparse
    args = argparse.Namespace(domain="medquad", seed=7, max_items=100)
    rec = rc.make_data_split_record(args, ITEMS[:2], n_loaded=3)
    assert rec["domain"] == "medquad"
    assert rec["seed"] == 7
    assert rec["max_items"] == 100
    assert rec["n_loaded"] == 3
    assert rec["calib_question_hashes"] == [
        rc._question_hash(ITEMS[0]["question"]),
        rc._question_hash(ITEMS[1]["question"]),
    ]


def test_threshold_json_carries_data_split_and_defaults_none():
    fit = ThresholdFitResult(alpha=0.05, tau_global=0.9)
    d_without = rc.threshold_to_json(fit, tau_claim=0.4, claim_alpha=0.05)
    assert d_without["data_split"] is None
    split = {"domain": "medquad", "seed": 7, "max_items": 2, "n_loaded": 2,
             "calib_question_hashes": [rc._question_hash(ITEMS[0]["question"])]}
    d_with = rc.threshold_to_json(fit, tau_claim=0.4, claim_alpha=0.05,
                                  data_split=split)
    assert json.loads(json.dumps(d_with))["data_split"] == split


def test_exclude_calibration_items_drops_overlap(caplog):
    split = {"calib_question_hashes": [rc._question_hash(ITEMS[0]["question"]),
                                       rc._question_hash(ITEMS[1]["question"])]}
    with caplog.at_level(logging.INFO):
        kept = rc.exclude_calibration_items(ITEMS, split, allow_overlap=False)
    assert kept == [ITEMS[2]]


def test_exclude_calibration_items_legacy_none_warns_and_keeps_all(caplog):
    with caplog.at_level(logging.WARNING):
        kept = rc.exclude_calibration_items(ITEMS, None, allow_overlap=False)
    assert kept == ITEMS
    assert any("data_split" in rec.message for rec in caplog.records)


def test_exclude_calibration_items_allow_overlap_keeps_all(caplog):
    split = {"calib_question_hashes": [rc._question_hash(it["question"]) for it in ITEMS]}
    with caplog.at_level(logging.WARNING):
        kept = rc.exclude_calibration_items(ITEMS, split, allow_overlap=True)
    assert kept == ITEMS
    assert any("allow_calib_overlap" in rec.message for rec in caplog.records)


def test_exclude_calibration_items_full_overlap_raises():
    split = {"calib_question_hashes": [rc._question_hash(it["question"]) for it in ITEMS]}
    with pytest.raises(ValueError, match="test set is empty"):
        rc.exclude_calibration_items(ITEMS, split, allow_overlap=False)

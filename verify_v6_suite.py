"""
Post-completion sanity sweep across all 36 real V6 suite jobs.

sacct showing COMPLETED/exit 0:0 only proves the process didn't crash -- it
says nothing about whether calibration degenerated, whether the judge
choked on a lot of responses, or whether the output actually has the
expected number of records. This script checks the things that matter
before anyone trusts these numbers for analysis or writeup.

Usage (run from repo root on the HPC, after all jobs finish):
    python3 verify_v6_suite.py
"""
from __future__ import annotations
import glob
import json
import os
import re

MODELS = ["llama_1b", "llama_3b", "gemma_1b", "gemma_4b", "phi_1.5", "phi_2"]

HS_DIR = "results/healthsearch_v6_1000"
MQ_DIR = "results/medDataset_v6_1000"

# (job_id, kind, dataset, model, out_path, thresholds_path_or_None)
JOB_IDS = {
    # HealthSearchQA baseline
    124261: "llama_1b", 124262: "llama_3b", 124263: "gemma_1b",
    124264: "gemma_4b", 124265: "phi_1.5", 124266: "phi_2",
    # HealthSearchQA conformal oracle-on
    124267: "llama_1b", 124268: "llama_3b", 124269: "gemma_1b",
    124270: "gemma_4b", 124271: "phi_1.5", 124272: "phi_2",
    # HealthSearchQA conformal oracle-off
    124273: "llama_1b", 124274: "llama_3b", 124275: "gemma_1b",
    124276: "gemma_4b", 124277: "phi_1.5", 124278: "phi_2",
    # MedQuad baseline
    124279: "llama_1b", 124280: "llama_3b", 124281: "gemma_1b",
    124282: "gemma_4b", 124283: "phi_1.5", 124284: "phi_2",
    # MedQuad conformal oracle-on
    124285: "llama_1b", 124286: "llama_3b", 124287: "gemma_1b",
    124288: "gemma_4b", 124289: "phi_1.5", 124290: "phi_2",
    # MedQuad conformal oracle-off
    124291: "llama_1b", 124292: "llama_3b", 124293: "gemma_1b",
    124294: "gemma_4b", 124295: "phi_1.5", 124296: "phi_2",
}

RUNS = []
for i, key in enumerate(MODELS):
    RUNS.append(dict(job=124261 + i, kind="baseline", dataset="healthsearch", model=key,
                      out=f"{HS_DIR}/run_baseline_{key}_v5.json", thresh=None))
for i, key in enumerate(MODELS):
    RUNS.append(dict(job=124267 + i, kind="conformal-oracle-on", dataset="healthsearch", model=key,
                      out=f"{HS_DIR}/run_conformal_{key}_v6.json", thresh=f"{HS_DIR}/thresholds_{key}_v6.json"))
for i, key in enumerate(MODELS):
    RUNS.append(dict(job=124273 + i, kind="conformal-oracle-off", dataset="healthsearch", model=key,
                      out=f"{HS_DIR}/run_conformal_{key}_no_oracle_v6.json", thresh=f"{HS_DIR}/thresholds_{key}_no_oracle_v6.json"))
for i, key in enumerate(MODELS):
    RUNS.append(dict(job=124279 + i, kind="baseline", dataset="medquad", model=key,
                      out=f"{MQ_DIR}/run_baseline_{key}_v5.json", thresh=None))
for i, key in enumerate(MODELS):
    RUNS.append(dict(job=124285 + i, kind="conformal-oracle-on", dataset="medquad", model=key,
                      out=f"{MQ_DIR}/run_conformal_{key}_oracle_v6.json", thresh=f"{MQ_DIR}/thresholds_{key}_oracle_v6.json"))
for i, key in enumerate(MODELS):
    RUNS.append(dict(job=124291 + i, kind="conformal-oracle-off", dataset="medquad", model=key,
                      out=f"{MQ_DIR}/run_conformal_{key}_no_oracle_v6.json", thresh=f"{MQ_DIR}/thresholds_{key}_no_oracle_v6.json"))

# ERROR-level lines that are EXPECTED/designed (C2, C4) -- don't flag these,
# but DO count them since a lot of GROUP CALIBRATION FAILED across many
# models is itself worth knowing about at real N=1000.
EXPECTED_ERROR_PATTERNS = [
    "CLAIM CALIBRATION FAILED",
    "RISK CALIBRATION FAILED",
    "GROUP CALIBRATION FAILED",
]


def find_error_log(job_id: int) -> str | None:
    matches = glob.glob(f"logs/error_*_{job_id}.txt")
    return matches[0] if matches else None


def scan_log(path: str) -> dict:
    if path is None or not os.path.exists(path):
        return dict(found=False, unexpected_errors=[], judge_parse_failures=0,
                    expected_error_count=0)
    unexpected = []
    judge_failures = 0
    expected_count = 0
    with open(path, errors="replace") as f:
        for line in f:
            if "Judge parse failure" in line or "Judge unparseable" in line:
                judge_failures += 1
            if "| ERROR |" in line:
                if any(p in line for p in EXPECTED_ERROR_PATTERNS):
                    expected_count += 1
                else:
                    unexpected.append(line.strip()[:160])
            if re.search(r"Traceback|exited with", line, re.IGNORECASE):
                unexpected.append(line.strip()[:160])
    return dict(found=True, unexpected_errors=unexpected,
                judge_parse_failures=judge_failures, expected_error_count=expected_count)


def scan_output(path: str) -> dict:
    if not os.path.exists(path):
        return dict(found=False, n_records=None)
    try:
        d = json.load(open(path))
    except Exception as e:
        return dict(found=True, n_records=None, parse_error=str(e))
    recs = d.get("records") or d.get("individual_records") or d.get("test_instances", [])
    return dict(found=True, n_records=len(recs))


def scan_thresholds(path: str | None) -> dict:
    if path is None:
        return {}
    if not os.path.exists(path):
        return dict(found=False)
    d = json.load(open(path))
    return dict(
        found=True,
        oracle_truth=d.get("oracle_truth"),
        calibration_failed=d.get("calibration_failed"),
        tau_claim_fallback=d.get("tau_claim_fallback"),
        tau_global=d.get("tau_global"),
        tau_claim=d.get("tau_claim"),
        n_groups=len(d.get("tau_by_group") or []) if d.get("tau_by_group") else 0,
    )


def main():
    print(f"{'job':<8}{'dataset':<12}{'model':<10}{'kind':<20}{'records':<10}"
          f"{'calib_fail':<12}{'claim_fb':<10}{'tau_global':<12}{'groups':<8}"
          f"{'judge_fail':<11}{'unexpected_err'}")
    print("-" * 140)

    total_unexpected = 0
    total_judge_failures = 0

    for r in RUNS:
        log_path = find_error_log(r["job"])
        log_info = scan_log(log_path)
        out_info = scan_output(r["out"])
        th_info = scan_thresholds(r["thresh"])

        n_unexpected = len(log_info["unexpected_errors"])
        total_unexpected += n_unexpected
        total_judge_failures += log_info["judge_parse_failures"]

        print(f"{r['job']:<8}{r['dataset']:<12}{r['model']:<10}{r['kind']:<20}"
              f"{str(out_info.get('n_records', '?')):<10}"
              f"{str(th_info.get('calibration_failed', '-')):<12}"
              f"{str(th_info.get('tau_claim_fallback', '-')):<10}"
              f"{str(th_info.get('tau_global', '-')):<12}"
              f"{str(th_info.get('n_groups', '-')):<8}"
              f"{log_info['judge_parse_failures']:<11}"
              f"{n_unexpected if n_unexpected else ''}")

        if not out_info.get("found"):
            print(f"  !! MISSING output file: {r['out']}")
        if r["thresh"] and not th_info.get("found"):
            print(f"  !! MISSING thresholds file: {r['thresh']}")
        if not log_info.get("found"):
            print(f"  !! MISSING error log for job {r['job']}")
        for line in log_info["unexpected_errors"][:5]:
            print(f"  UNEXPECTED: {line}")

    print("-" * 140)
    print(f"TOTAL unexpected error lines across all 36 jobs: {total_unexpected}")
    print(f"TOTAL judge parse failures across all 36 jobs: {total_judge_failures}")
    print("(judge parse failures are handled gracefully -- C10 -- but a large")
    print(" count on one model is worth investigating before trusting its rates)")


if __name__ == "__main__":
    main()

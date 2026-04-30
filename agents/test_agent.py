"""
OpenHarness Test Agent
Validates ML model outputs, runs statistical tests, and verifies data integrity.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


RESULTS_DIR = Path("/workspace/results")
PASS = "PASS"
FAIL = "FAIL"


def check_file(path: str, description: str) -> tuple[bool, str]:
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return True, f"{PASS}: {description} ({p.stat().st_size} bytes)"
    return False, f"{FAIL}: {description} not found or empty"


def validate_model_performance(results_path: str) -> tuple[bool, str]:
    try:
        with open(results_path) as f:
            res = json.load(f)
        auc = res.get("roc_auc", 0)
        if auc >= 0.70:
            return True, f"{PASS}: ROC-AUC = {auc:.4f} (threshold ≥ 0.70)"
        return False, f"{FAIL}: ROC-AUC = {auc:.4f} below threshold"
    except Exception as e:
        return False, f"{FAIL}: Cannot read model results — {e}"


def validate_gene_list(gene_file: str) -> tuple[bool, str]:
    try:
        df = pd.read_csv(gene_file)
        aki_genes = ["HAVCR1", "LCN2", "CXCL8", "MMP7", "SOX9", "VIM"]
        found = [g for g in aki_genes if g in df["gene"].values]
        if len(found) >= 3:
            return True, f"{PASS}: AKI signature genes detected: {found}"
        return False, f"{FAIL}: Too few AKI genes found: {found}"
    except Exception as e:
        return False, f"{FAIL}: Gene validation error — {e}"


def run_permutation_test(results_path: str, n_perm: int = 100) -> tuple[bool, str]:
    try:
        with open(results_path) as f:
            res = json.load(f)
        observed_auc = res.get("roc_auc", 0)
        y_true = np.array(res.get("y_true", []))
        y_score = np.array(res.get("y_score", []))
        if len(y_true) == 0:
            return False, f"{FAIL}: No y_true/y_score for permutation test"
        perm_aucs = [
            roc_auc_score(np.random.permutation(y_true), y_score)
            for _ in range(n_perm)
        ]
        p_val = np.mean(np.array(perm_aucs) >= observed_auc)
        if p_val < 0.05:
            return True, f"{PASS}: Permutation p-value = {p_val:.4f} (n={n_perm})"
        return False, f"{FAIL}: Permutation p-value = {p_val:.4f} (not significant)"
    except Exception as e:
        return False, f"{FAIL}: Permutation test error — {e}"


def main():
    print("[TestAgent] Starting validation suite...")
    tests = [
        check_file(RESULTS_DIR / "preprocessed_adata.h5ad", "Preprocessed AnnData"),
        check_file(RESULTS_DIR / "aki_signature_scores.csv", "AKI signature scores"),
        check_file(RESULTS_DIR / "differential_expression.csv", "Differential expression"),
        check_file(RESULTS_DIR / "model_results.json", "Model results JSON"),
        check_file(RESULTS_DIR / "feature_importance.csv", "Feature importance"),
        validate_model_performance(str(RESULTS_DIR / "model_results.json")),
        validate_gene_list(str(RESULTS_DIR / "differential_expression.csv")),
        run_permutation_test(str(RESULTS_DIR / "model_results.json")),
    ]

    passed = sum(1 for ok, _ in tests if ok)
    total = len(tests)
    for ok, msg in tests:
        status = "✓" if ok else "✗"
        print(f"  [{status}] {msg}")

    summary = {"passed": passed, "total": total, "details": [msg for _, msg in tests]}
    with open(RESULTS_DIR / "test_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[TestAgent] {passed}/{total} tests passed.")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

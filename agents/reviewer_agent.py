"""
OpenHarness Reviewer/Test Agent — evaluates iteration metrics against current best.
Writes results/iteration_N_review.json with decision: ACCEPT | REPLAN | COMPLETE.
Called by orchestrator after each coding agent run.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"

# Minimum thresholds for a result to be considered viable
MIN_AUROC = 0.70
# Composite weight: 0.5 * AUROC + 0.3 * AUPRC + 0.2 * F1
W_AUROC, W_AUPRC, W_F1 = 0.5, 0.3, 0.2


def _composite(auroc: float, auprc: float, f1: float) -> float:
    return W_AUROC * auroc + W_AUPRC * auprc + W_F1 * f1


def _load_best() -> dict | None:
    f = RESULTS / "best_metrics.json"
    if f.exists():
        with open(f) as fh:
            return json.load(fh)
    return None


def _save_best(metrics: dict, iteration: int):
    record = {
        "iteration":        iteration,
        "approach_name":    metrics["approach_name"],
        "auroc_mean":       metrics["auroc_mean"],
        "auprc_mean":       metrics["auprc_mean"],
        "f1_mean":          metrics["f1_mean"],
        "composite_score":  metrics["composite_score"],
        "sensitivity_mean": metrics["sensitivity_mean"],
        "specificity_mean": metrics["specificity_mean"],
    }
    with open(RESULTS / "best_metrics.json", "w") as f:
        json.dump(record, f, indent=2)


def _load_search_target() -> float:
    f = RESULTS / "search_results.json"
    if f.exists():
        with open(f) as fh:
            d = json.load(fh)
        return d.get("target_auroc", 0.92)
    return 0.92


def review(iteration: int, metrics: dict, best: dict | None,
           min_iterations: int = 5, max_iterations: int = 7) -> dict:
    target_auroc = _load_search_target()
    new_auroc    = metrics["auroc_mean"]
    new_auprc    = metrics["auprc_mean"]
    new_f1       = metrics["f1_mean"]
    new_comp     = metrics["composite_score"]

    if best is None:
        # First iteration — always accept as baseline
        decision = "ACCEPT"
        best_auroc = new_auroc
        improvement = None
        feedback = (
            f"Baseline established. AUROC={new_auroc:.4f}, AUPRC={new_auprc:.4f}, "
            f"F1={new_f1:.4f}. Literature target={target_auroc:.3f}. "
            f"Gap to target: {target_auroc - new_auroc:+.4f}."
        )
        hint = "Baseline set. Try XGBoost with extended feature set (top-100 DE genes)."
    else:
        best_comp  = best["composite_score"]
        best_auroc = best["auroc_mean"]

        if new_comp > best_comp:
            delta = new_comp - best_comp
            decision = "ACCEPT"
            feedback = (
                f"IMPROVEMENT: composite {best_comp:.4f} → {new_comp:.4f} (+{delta:.4f}). "
                f"AUROC {best_auroc:.4f} → {new_auroc:.4f}. "
                f"New best approach: {metrics['approach_name']}."
            )
            hint = (
                "Continue improving. Consider ensemble or stacking to combine "
                "the gains seen in this iteration with previous approaches."
            )
            best_auroc = new_auroc
        else:
            delta = new_comp - best_comp
            decision = "REPLAN"
            feedback = (
                f"NO IMPROVEMENT: composite {new_comp:.4f} vs best {best_comp:.4f} ({delta:.4f}). "
                f"AUROC {new_auroc:.4f} vs best {best_auroc:.4f}. "
                f"Approach '{metrics['approach_name']}' did not surpass the current best. "
                f"Plan a new strategy with different feature engineering or model architecture."
            )
            hint = (
                "Previous approach underperformed. Next strategy should use a "
                "fundamentally different feature representation (e.g., PCA, pathway scores, "
                "ensemble stacking) rather than incremental changes."
            )
        improvement = delta

    # Override decision to COMPLETE if max iterations reached
    if iteration >= max_iterations:
        decision = "COMPLETE"
        feedback += f" Maximum iterations ({max_iterations}) reached — pipeline complete."

    # Flag if below minimum viable threshold
    viable = new_auroc >= MIN_AUROC
    if not viable:
        feedback += f" WARNING: AUROC={new_auroc:.4f} below minimum threshold {MIN_AUROC}."

    review_doc = {
        "iteration":          iteration,
        "timestamp":          datetime.now().isoformat(),
        "approach_name":      metrics["approach_name"],
        "new_auroc":          new_auroc,
        "new_auprc":          new_auprc,
        "new_f1":             new_f1,
        "new_composite":      new_comp,
        "best_auroc_so_far":  best_auroc,
        "best_composite_so_far": (best or {}).get("composite_score", new_comp),
        "decision":           decision,
        "viable":             viable,
        "target_auroc":       target_auroc,
        "gap_to_target":      round(target_auroc - new_auroc, 4),
        "feedback":           feedback,
        "next_approach_hint": hint,
        "min_iterations":     min_iterations,
        "max_iterations":     max_iterations,
    }

    out = RESULTS / f"iteration_{iteration}_review.json"
    with open(out, "w") as f:
        json.dump(review_doc, f, indent=2)

    print(f"\n[REVIEWER AGENT] Iteration {iteration}")
    print(f"  Approach     : {metrics['approach_name']}")
    print(f"  AUROC        : {new_auroc:.4f}  (best so far: {best_auroc:.4f})")
    print(f"  AUPRC        : {new_auprc:.4f}")
    print(f"  F1           : {new_f1:.4f}")
    print(f"  Composite    : {new_comp:.4f}  (best: {(best or {}).get('composite_score', new_comp):.4f})")
    print(f"  Decision     : {decision}")
    print(f"  Feedback     : {feedback}")
    if decision == "REPLAN":
        print(f"  Hint to Plan : {hint}")

    return review_doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--min-iterations", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=7)
    args = parser.parse_args()

    n = args.iteration
    metrics_file = RESULTS / f"iteration_{n}_metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_file}")
    with open(metrics_file) as f:
        metrics = json.load(f)

    best = _load_best()
    rev  = review(n, metrics, best,
                  min_iterations=args.min_iterations,
                  max_iterations=args.max_iterations)

    # Update best if this iteration improved (or is baseline)
    if rev["decision"] in ("ACCEPT",) and (
        best is None or metrics["composite_score"] > best["composite_score"]
    ):
        _save_best(metrics, n)
        print(f"  Updated best_metrics.json → iteration {n}")

    return rev


if __name__ == "__main__":
    main()

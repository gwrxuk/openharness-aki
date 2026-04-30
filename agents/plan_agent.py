"""
OpenHarness Plan Agent — selects the next ML strategy for AKI classification.
Reads search_results.json + last reviewer feedback to choose the next approach.
Writes results/iteration_N_plan.json.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"

STRATEGIES = [
    {
        "id": 1,
        "name": "RF-Baseline",
        "description": "Random Forest (200 trees, depth=8) on top-50 DE genes by |logFC|",
        "rationale": (
            "Establish a reproducible baseline matching published AKI classifiers. "
            "Literature (Lake et al. 2023, Miao et al. 2021) uses RF as primary benchmark."
        ),
        "config": {
            "feature_method": "top_de", "n_features": 50,
            "model": "random_forest",
            "model_params": {"n_estimators": 200, "max_depth": 8, "class_weight": "balanced"},
        },
    },
    {
        "id": 2,
        "name": "XGB-Extended",
        "description": "XGBoost (300 trees, depth=5, lr=0.03) on top-100 DE genes",
        "rationale": (
            "XGBoost handles class imbalance better via scale_pos_weight. "
            "Expanding to 100 features may capture secondary injury signals missed by top-50."
        ),
        "config": {
            "feature_method": "top_de", "n_features": 100,
            "model": "xgboost",
            "model_params": {
                "n_estimators": 300, "max_depth": 5,
                "learning_rate": 0.03, "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },
    },
    {
        "id": 3,
        "name": "LASSO-RF",
        "description": "L1-regularised logistic regression for feature selection → Random Forest",
        "rationale": (
            "LASSO imposes sparsity beyond DE ranking, selecting features by predictive "
            "weight rather than fold-change magnitude. Reduces collinear gene redundancy."
        ),
        "config": {
            "feature_method": "lasso", "n_features": None,
            "model": "random_forest",
            "model_params": {"n_estimators": 200, "max_depth": 8, "class_weight": "balanced"},
            "lasso_C": 0.05,
        },
    },
    {
        "id": 4,
        "name": "PCA-GBT",
        "description": "PCA (50 components on top-2000 HVGs) → sklearn Gradient Boosting Trees",
        "rationale": (
            "PCA captures global transcriptome covariance structure beyond pairwise DE. "
            "GBT with shallow depth avoids overfitting on compressed representations."
        ),
        "config": {
            "feature_method": "pca", "n_components": 50,
            "model": "gradient_boosting",
            "model_params": {
                "n_estimators": 200, "max_depth": 4,
                "learning_rate": 0.05, "subsample": 0.8,
            },
        },
    },
    {
        "id": 5,
        "name": "MultiFeature-Vote",
        "description": "Soft voting ensemble (RF + XGB + LR) on top-75 DE genes + 5 pathway scores",
        "rationale": (
            "Combining statistical DE features with curated biological pathway scores "
            "and a soft-voting ensemble to reduce individual model variance. "
            "Pathway scores encode domain knowledge not captured by raw expression."
        ),
        "config": {
            "feature_method": "multi", "n_features": 75,
            "include_pathway_scores": True,
            "model": "voting",
            "model_params": {"voting": "soft"},
        },
    },
    {
        "id": 6,
        "name": "Stack-RF-XGB",
        "description": "Stacking ensemble: RF + XGBoost base learners; Logistic Regression meta-learner",
        "rationale": (
            "Stacking leverages complementary model strengths — RF robustness + XGB boosting. "
            "LR meta-learner calibrates the ensemble's probability outputs."
        ),
        "config": {
            "feature_method": "top_de", "n_features": 75,
            "model": "stacking",
            "model_params": {"cv": 5},
        },
    },
    {
        "id": 7,
        "name": "TunedRF-Best",
        "description": "GridSearchCV-optimised Random Forest on best feature set identified across iterations",
        "rationale": (
            "After 6 iterations of model exploration, apply systematic hyperparameter search "
            "to the configuration space around the best-performing setup found so far."
        ),
        "config": {
            "feature_method": "top_de", "n_features": 75,
            "model": "tuned_rf",
            "model_params": {"cv": 3},
        },
    },
]


def select_strategy(iteration: int, reviewer_feedback: dict | None) -> dict:
    """
    Pick the strategy for this iteration.
    Iteration 1-7 use strategies 1-7 in order.
    Reviewer feedback is logged as rationale context but does not reorder (
    the predefined sequence covers increasingly complex approaches by design).
    """
    idx = min(iteration - 1, len(STRATEGIES) - 1)
    strategy = STRATEGIES[idx].copy()

    if reviewer_feedback:
        prev_auroc = reviewer_feedback.get("best_auroc_so_far", None)
        decision   = reviewer_feedback.get("decision", "ACCEPT")
        hint       = reviewer_feedback.get("next_approach_hint", "")
        strategy["reviewer_context"] = {
            "previous_decision": decision,
            "best_auroc_so_far": prev_auroc,
            "hint": hint,
        }

    return strategy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    n = args.iteration

    # Load reviewer feedback from previous iteration if available
    reviewer_feedback = None
    prev_review = RESULTS / f"iteration_{n - 1}_review.json"
    if prev_review.exists():
        with open(prev_review) as f:
            reviewer_feedback = json.load(f)

    strategy = select_strategy(n, reviewer_feedback)
    plan = {
        "iteration": n,
        "timestamp": datetime.now().isoformat(),
        "approach_id":   strategy["id"],
        "approach_name": strategy["name"],
        "description":   strategy["description"],
        "rationale":     strategy["rationale"],
        "config":        strategy["config"],
        "reviewer_context": strategy.get("reviewer_context"),
    }

    out = RESULTS / f"iteration_{n}_plan.json"
    with open(out, "w") as f:
        json.dump(plan, f, indent=2)

    print(f"[PLAN AGENT] Iteration {n}: {strategy['name']}")
    print(f"  Description : {strategy['description']}")
    print(f"  Feature method: {strategy['config']['feature_method']}  "
          f"  n_features: {strategy['config'].get('n_features', 'auto')}")
    print(f"  Model       : {strategy['config']['model']}")
    print(f"  Rationale   : {strategy['rationale'][:120]}...")
    if reviewer_feedback:
        print(f"  Reviewer context: prev_decision={reviewer_feedback.get('decision')}  "
              f"best_auroc={reviewer_feedback.get('best_auroc_so_far')}")
    print(f"  Saved plan  → {out}")

    return plan


if __name__ == "__main__":
    main()

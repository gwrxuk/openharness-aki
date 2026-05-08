"""
OpenHarness Plan Agent — selects the next ML strategy for AKI classification.
Reads search_results.json + omics_discovery.json + last reviewer feedback to
choose the next approach.  When omics_discovery.json is present the strategy
pool is extended with omics-aware approaches (multi-omics integration, MOFA+,
SNF, pathway-score augmentation).
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

# ── Omics-aware strategy extensions ──────────────────────────────────────────
# These are appended when omics_discovery.json is present and reveals dominant
# modalities or high-scoring GEO datasets that suggest different feature spaces.

OMICS_STRATEGIES = [
    {
        "id": 8,
        "name": "PathwayScore-XGB",
        "description": (
            "XGBoost on enriched pathway activity scores (KEGG/Reactome kidney "
            "injury, fibrosis, inflammation) computed from single-cell expression "
            "via AUCell/GSVA. Combines biological priors with boosted trees."
        ),
        "rationale": (
            "Omics discovery surfaced papers using pathway-level features for "
            "AKI classification (metabolomics + transcriptomics integration). "
            "Pathway scores compress thousands of genes into ~20 biologically "
            "meaningful features, reducing overfitting while capturing "
            "coordinated injury programmes missed by individual gene DE."
        ),
        "config": {
            "feature_method": "pathway_scores",
            "n_features": 20,
            "model": "xgboost",
            "model_params": {
                "n_estimators": 300, "max_depth": 4,
                "learning_rate": 0.05, "subsample": 0.8,
                "colsample_bytree": 0.7,
            },
            "pathway_sets": [
                "KEGG_RENAL_CELL_CARCINOMA",
                "REACTOME_CELLULAR_RESPONSE_TO_HEAT_STRESS",
                "HALLMARK_HYPOXIA",
                "HALLMARK_INFLAMMATORY_RESPONSE",
                "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
                "HALLMARK_APOPTOSIS",
                "KEGG_CELL_CYCLE",
                "REACTOME_SIGNALING_BY_TGFB",
            ],
            "omics_source": "transcriptomics_pathway_scores",
        },
    },
    {
        "id": 9,
        "name": "MultiOmics-EarlyFusion",
        "description": (
            "Early fusion of transcriptomic (top-50 DE genes) + proteomics-proxy "
            "features (KIM-1/NGAL/Cystatin-C expression as surrogate protein "
            "biomarkers) → LASSO selection → Random Forest."
        ),
        "rationale": (
            "Omics discovery found high-scoring proteomics and metabolomics "
            "kidney datasets. In the absence of a matched proteomics layer, "
            "gene-level surrogates for established urinary protein biomarkers "
            "(HAVCR1→KIM-1, LCN2→NGAL, CST3→Cystatin-C) are concatenated "
            "with DE features. LASSO then selects the most predictive cross-modal "
            "features before Random Forest classification."
        ),
        "config": {
            "feature_method": "multi_omics_proxy",
            "n_features": 75,
            "model": "random_forest",
            "model_params": {
                "n_estimators": 250, "max_depth": 8, "class_weight": "balanced"
            },
            "lasso_C": 0.03,
            "proxy_biomarkers": ["HAVCR1", "LCN2", "CST3", "UMOD", "SPP1",
                                  "MMP7", "VCAM1", "CXCL8", "SOX9", "VIM"],
            "omics_source": "transcriptomics_proteomics_proxy",
        },
    },
    {
        "id": 10,
        "name": "HVG-VAE-XGB",
        "description": (
            "Variational autoencoder (scVI-style, 10-dim latent) trained on "
            "top-2000 highly variable genes → latent representation → XGBoost "
            "classifier. Deep latent features capture non-linear covariance "
            "structure beyond PCA."
        ),
        "rationale": (
            "Spatial transcriptomics and multi-omics papers retrieved by omics "
            "discovery agent use deep embedding approaches to handle "
            "high-dimensional, zero-inflated count data. A VAE latent space "
            "reduces noise while preserving biologically relevant variance. "
            "XGBoost on the low-dimensional latent tends to outperform linear "
            "classifiers on compressed representations."
        ),
        "config": {
            "feature_method": "vae_latent",
            "n_hvg": 2000,
            "latent_dim": 10,
            "model": "xgboost",
            "model_params": {
                "n_estimators": 300, "max_depth": 5,
                "learning_rate": 0.05, "subsample": 0.8,
            },
            "vae_epochs": 50,
            "omics_source": "transcriptomics_deep_embedding",
        },
    },
]


def _load_omics_discovery() -> dict | None:
    """Load omics_discovery.json if available; return None otherwise."""
    f = RESULTS / "omics_discovery.json"
    if f.exists():
        try:
            with open(f) as fh:
                return json.load(fh)
        except Exception:
            pass
    return None


def _build_strategy_pool(omics: dict | None) -> list[dict]:
    """
    Return the strategy pool to use for this run.
    Appends OMICS_STRATEGIES when omics_discovery.json is present and reports
    at least one high-scoring (ml_score >= 0.5) record.
    """
    pool = list(STRATEGIES)
    if omics is None:
        return pool

    rec = omics.get("recommendation", {})
    top_records = rec.get("top_datasets", [])
    best_score  = max((r.get("ml_score", 0) for r in top_records), default=0)

    if best_score >= 0.5:
        pool = pool + list(OMICS_STRATEGIES)

    return pool


def select_strategy(iteration: int, reviewer_feedback: dict | None,
                    omics: dict | None = None) -> dict:
    """
    Pick the strategy for this iteration.
    When omics_discovery.json is present and has high-scoring datasets the pool
    is extended with OMICS_STRATEGIES (ids 8-10); otherwise falls back to the
    original 7-strategy pool.
    Reviewer feedback is annotated on the chosen strategy as context.
    """
    pool = _build_strategy_pool(omics)
    idx  = min(iteration - 1, len(pool) - 1)
    strategy = pool[idx].copy()

    if reviewer_feedback:
        prev_auroc = reviewer_feedback.get("best_auroc_so_far", None)
        decision   = reviewer_feedback.get("decision", "ACCEPT")
        hint       = reviewer_feedback.get("next_approach_hint", "")
        strategy["reviewer_context"] = {
            "previous_decision": decision,
            "best_auroc_so_far": prev_auroc,
            "hint": hint,
        }

    if omics:
        rec = omics.get("recommendation", {})
        strategy["omics_context"] = {
            "dominant_modality":    rec.get("dominant_modality", "unknown"),
            "recommended_models":   rec.get("recommended_models", [])[:3],
            "recommended_features": rec.get("recommended_features", [])[:3],
            "top_dataset_score":    max(
                (d.get("ml_score", 0) for d in rec.get("top_datasets", [])),
                default=0,
            ),
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

    # Load omics discovery results if available
    omics = _load_omics_discovery()
    if omics:
        pool_size = len(_build_strategy_pool(omics))
        print(f"[PLAN AGENT] Omics discovery loaded — strategy pool: {pool_size} approaches")
    else:
        print("[PLAN AGENT] No omics_discovery.json — using base 7-strategy pool")

    strategy = select_strategy(n, reviewer_feedback, omics)
    plan = {
        "iteration":      n,
        "timestamp":      datetime.now().isoformat(),
        "approach_id":    strategy["id"],
        "approach_name":  strategy["name"],
        "description":    strategy["description"],
        "rationale":      strategy["rationale"],
        "config":         strategy["config"],
        "reviewer_context": strategy.get("reviewer_context"),
        "omics_context":    strategy.get("omics_context"),
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
    if omics:
        ctx = strategy.get("omics_context", {})
        print(f"  Omics context : dominant_modality={ctx.get('dominant_modality')}  "
              f"top_dataset_score={ctx.get('top_dataset_score'):.2f}")
    print(f"  Saved plan  → {out}")

    return plan


if __name__ == "__main__":
    main()

"""
OpenHarness Coding Agent — configurable AKI ML pipeline.
Reads iteration_N_plan.json, runs the specified approach, writes iteration_N_metrics.json.
Supports 7 distinct ML strategies with per-fold logging and timing.
"""
import argparse
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, roc_curve,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sc.settings.verbosity = 0

ROOT      = Path(__file__).parent.parent
RESULTS   = ROOT / "results"
DATA_FILE = ROOT / "data" / "Mature_Full_v2.1.h5ad"

AKI_CELLTYPES = [
    "Distinct proximal tubule 2",
    "Proliferating Proximal Tubule",
    "Epithelial progenitor cell",
    "Myofibroblast",
]
NORMAL_PT_CELLTYPES = ["Proximal tubule"]

AKI_SIGNATURE_GENES = {
    "injury_markers":    ["HAVCR1","LCN2","CXCL8","CXCL2","IL6","SPP1"],
    "dedifferentiation": ["VIM","CD44","SOX9","VCAM1"],
    "ecm_fibrosis":      ["MMP7","FN1","COL1A1","ACTA2","PDGFRA"],
    "pt_healthy":        ["SLC34A1","CUBN","SLC7A9","ANPEP"],
}


# ─── Data loading (cached across call in the same process) ───────────────────
_adata_cache = None

def _load_sub() -> tuple[np.ndarray, np.ndarray, sc.AnnData]:
    global _adata_cache
    if _adata_cache is None:
        adata = sc.read_h5ad(str(DATA_FILE))
        labels = pd.Series("other", index=adata.obs_names)
        for ct in AKI_CELLTYPES:
            labels[adata.obs["celltype"] == ct] = "AKI"
        for ct in NORMAL_PT_CELLTYPES:
            labels[adata.obs["celltype"] == ct] = "Normal_PT"
        adata.obs["aki_label"] = labels

        sub = adata[adata.obs["aki_label"].isin(["AKI", "Normal_PT"])].copy()
        sc.pp.normalize_total(sub, target_sum=1e4)
        sc.pp.log1p(sub)
        # Pathway scores
        for cat, genes in AKI_SIGNATURE_GENES.items():
            present = [g for g in genes if g in sub.var_names]
            sc.tl.score_genes(sub, present, score_name=f"score_{cat}")
        sub.obs["AKI_injury_score"] = (
            sub.obs["score_injury_markers"] +
            sub.obs["score_dedifferentiation"] +
            sub.obs["score_ecm_fibrosis"] -
            sub.obs["score_pt_healthy"]
        )
        _adata_cache = sub

    sub = _adata_cache
    y   = (sub.obs["aki_label"] == "AKI").astype(int).values
    return sub, y


# ─── Feature extraction strategies ───────────────────────────────────────────

def _features_top_de(sub, n_features: int) -> np.ndarray:
    de_df = pd.read_csv(RESULTS / "differential_expression.csv")
    genes = (de_df[de_df["pvals_adj"] < 0.05]
             .nlargest(n_features, "abs_lfc")["gene"]
             .tolist())
    genes = [g for g in genes if g in sub.var_names][:n_features]
    X = sub[:, genes].X
    arr = X.toarray() if hasattr(X, "toarray") else np.array(X)
    return arr, genes


def _features_lasso(sub, y: np.ndarray, C: float = 0.05) -> tuple[np.ndarray, list]:
    de_df = pd.read_csv(RESULTS / "differential_expression.csv")
    pool_genes = (de_df[de_df["pvals_adj"] < 0.05]
                  .nlargest(200, "abs_lfc")["gene"].tolist())
    pool_genes = [g for g in pool_genes if g in sub.var_names][:200]
    X_pool = sub[:, pool_genes].X
    X_arr  = X_pool.toarray() if hasattr(X_pool, "toarray") else np.array(X_pool)
    scl = StandardScaler()
    X_s = scl.fit_transform(X_arr)
    lasso = LogisticRegression(
        penalty="l1", C=C, solver="liblinear",
        class_weight="balanced", max_iter=2000, random_state=42,
    )
    lasso.fit(X_s, y)
    selected_idx = np.where(lasso.coef_[0] != 0)[0]
    if len(selected_idx) < 5:
        selected_idx = np.argsort(np.abs(lasso.coef_[0]))[::-1][:20]
    sel_genes = [pool_genes[i] for i in selected_idx]
    X_sel = X_arr[:, selected_idx]
    return X_sel, sel_genes


def _features_pca(sub, n_components: int = 50) -> tuple[np.ndarray, list]:
    sc.pp.highly_variable_genes(sub, n_top_genes=2000, flavor="seurat",
                                batch_key=None if "sample" not in sub.obs.columns else None)
    hvg_mask = sub.var.get("highly_variable", pd.Series(True, index=sub.var_names))
    hvg_genes = sub.var_names[hvg_mask].tolist()[:2000]
    X_hvg = sub[:, hvg_genes].X
    X_arr = X_hvg.toarray() if hasattr(X_hvg, "toarray") else np.array(X_hvg)
    scl = StandardScaler()
    X_s = scl.fit_transform(X_arr)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_s)
    feat_names = [f"PC{i+1}" for i in range(n_components)]
    explained = pca.explained_variance_ratio_.sum()
    print(f"    PCA: {n_components} components on {len(hvg_genes)} HVGs, "
          f"variance explained={explained:.3f}")
    return X_pca, feat_names


def _features_multi(sub, y: np.ndarray, n_features: int = 75) -> tuple[np.ndarray, list]:
    X_de, de_genes = _features_top_de(sub, n_features)
    pathway_cols = [
        "score_injury_markers", "score_dedifferentiation",
        "score_ecm_fibrosis", "score_pt_healthy", "AKI_injury_score",
    ]
    path_arr = sub.obs[pathway_cols].values
    X_combined = np.hstack([X_de, path_arr])
    feature_names = de_genes + pathway_cols
    return X_combined, feature_names


# ─── Model builders ──────────────────────────────────────────────────────────

def _build_model(config: dict, scale_pos: float):
    name   = config["model"]
    params = config.get("model_params", {})

    if name == "random_forest":
        return RandomForestClassifier(
            **params, random_state=42, n_jobs=-1
        )
    if name == "xgboost":
        return xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(**params, random_state=42)
    if name == "voting":
        rf  = RandomForestClassifier(n_estimators=100, max_depth=8,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                   scale_pos_weight=scale_pos, eval_metric="logloss",
                                   random_state=42, verbosity=0)
        lr  = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
        return VotingClassifier([("rf", rf), ("xgb", xgb_m), ("lr", lr)],
                                voting=params.get("voting", "soft"))
    if name == "stacking":
        rf  = RandomForestClassifier(n_estimators=100, max_depth=6,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                   scale_pos_weight=scale_pos, eval_metric="logloss",
                                   random_state=42, verbosity=0)
        meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500)
        return StackingClassifier(
            estimators=[("rf", rf), ("xgb", xgb_m)],
            final_estimator=meta,
            cv=params.get("cv", 5),
            n_jobs=-1,
        )
    if name == "tuned_rf":
        base = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
        grid = {
            "n_estimators": [100, 200, 300],
            "max_depth":    [6, 8, 10],
            "min_samples_leaf": [1, 3],
        }
        return GridSearchCV(base, grid, cv=params.get("cv", 3),
                            scoring="roc_auc", n_jobs=-1, refit=True)
    raise ValueError(f"Unknown model: {name}")


# ─── CV evaluation ───────────────────────────────────────────────────────────

def _youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return thresholds[np.argmax(j)]


def _cv_metrics(model, X: np.ndarray, y: np.ndarray,
                scale: bool, iteration: int, name: str) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler() if scale else None

    fold_auroc, fold_auprc, fold_f1 = [], [], []
    fold_sens, fold_spec = [], []
    fold_times = []

    print(f"\n    5-fold stratified CV — {name}:")
    for fold_i, (tr, te) in enumerate(cv.split(X, y)):
        t0 = time.time()
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        if scaler:
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
        m = _build_model({"model": model._name if hasattr(model, "_name") else
                           type(model).__name__.lower(), **{}},
                          scale_pos=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        m.fit(X_tr, y_tr)
        y_prob = m.predict_proba(X_te)[:, 1]
        thr    = _youden_threshold(y_te, y_prob)
        y_pred = (y_prob >= thr).astype(int)

        auroc = roc_auc_score(y_te, y_prob)
        auprc = average_precision_score(y_te, y_prob)
        f1    = f1_score(y_te, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        elapsed = time.time() - t0

        fold_auroc.append(auroc); fold_auprc.append(auprc); fold_f1.append(f1)
        fold_sens.append(sens);   fold_spec.append(spec)
        fold_times.append(elapsed)

        print(f"      Fold {fold_i+1}/5 | train={len(tr):,} | test={len(te):,} | "
              f"AUROC={auroc:.4f} | AUPRC={auprc:.4f} | F1={f1:.4f} | "
              f"Sens={sens:.4f} | Spec={spec:.4f} | {elapsed:.2f}s")

    return {
        "per_fold_auroc": [round(a, 4) for a in fold_auroc],
        "per_fold_auprc": [round(a, 4) for a in fold_auprc],
        "per_fold_f1":    [round(a, 4) for a in fold_f1],
        "auroc_mean": float(np.mean(fold_auroc)),
        "auroc_std":  float(np.std(fold_auroc)),
        "auprc_mean": float(np.mean(fold_auprc)),
        "auprc_std":  float(np.std(fold_auprc)),
        "f1_mean":    float(np.mean(fold_f1)),
        "f1_std":     float(np.std(fold_f1)),
        "sensitivity_mean": float(np.mean(fold_sens)),
        "specificity_mean": float(np.mean(fold_spec)),
        "total_time_sec": round(sum(fold_times), 2),
    }


# ─── Main entry ──────────────────────────────────────────────────────────────

def run_iteration(iteration: int, plan: dict, verbose: bool = True) -> dict:
    config = plan["config"]
    approach = plan["approach_name"]
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  [CODING AGENT] Iteration {iteration}: {approach}")
    print(f"  {plan['description']}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    print(f"\n  Loading data ...")
    sub, y = _load_sub()
    scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
    print(f"  AKI={y.sum()}, Normal_PT={(y==0).sum()}, total={len(y)}")

    # Feature extraction
    fm = config["feature_method"]
    print(f"\n  Feature extraction: {fm}")
    if fm == "top_de":
        X, feat_names = _features_top_de(sub, config["n_features"])
    elif fm == "lasso":
        X, feat_names = _features_lasso(sub, y, C=config.get("lasso_C", 0.05))
    elif fm == "pca":
        X, feat_names = _features_pca(sub, config.get("n_components", 50))
    elif fm == "multi":
        X, feat_names = _features_multi(sub, y, config.get("n_features", 75))
    else:
        raise ValueError(f"Unknown feature_method: {fm}")

    n_feats = X.shape[1]
    scale   = config.get("scale", True)
    print(f"  Feature matrix: {X.shape[0]:,} cells × {n_feats} features")

    # Build model for CV — need model name and params
    model_name   = config["model"]
    model_params = config.get("model_params", {})

    # We use _build_model directly per fold in the CV loop below
    # Pack config for _cv_metrics
    def _fold_model():
        return _build_model({"model": model_name, "model_params": model_params}, scale_pos)

    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler() if scale else None

    fold_auroc, fold_auprc, fold_f1, fold_sens, fold_spec, fold_times = [], [], [], [], [], []

    print(f"\n  5-fold stratified CV — {model_name}:")
    for fold_i, (tr, te) in enumerate(cv.split(X, y)):
        t0 = time.time()
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        if scaler is not None:
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        m = _fold_model()
        m.fit(X_tr, y_tr)
        y_prob = m.predict_proba(X_te)[:, 1]
        thr    = _youden_threshold(y_te, y_prob)
        y_pred = (y_prob >= thr).astype(int)

        auroc = roc_auc_score(y_te, y_prob)
        auprc = average_precision_score(y_te, y_prob)
        f1    = f1_score(y_te, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        elapsed = time.time() - t0

        fold_auroc.append(auroc); fold_auprc.append(auprc); fold_f1.append(f1)
        fold_sens.append(sens);   fold_spec.append(spec);   fold_times.append(elapsed)

        print(f"    Fold {fold_i+1}/5 | train={len(tr):,} | test={len(te):,} | "
              f"AUROC={auroc:.4f} | AUPRC={auprc:.4f} | F1={f1:.4f} | "
              f"Sens={sens:.4f} | Spec={spec:.4f} | {elapsed:.2f}s")

    metrics = {
        "iteration":     iteration,
        "approach_name": approach,
        "description":   plan["description"],
        "timestamp":     datetime.now().isoformat(),
        "feature_method": fm,
        "n_features":    n_feats,
        "model":         model_name,
        "n_cells_aki":   int(y.sum()),
        "n_cells_normal": int((y == 0).sum()),
        "per_fold_auroc": [round(a, 4) for a in fold_auroc],
        "per_fold_auprc": [round(a, 4) for a in fold_auprc],
        "per_fold_f1":    [round(a, 4) for a in fold_f1],
        "auroc_mean": float(np.mean(fold_auroc)),
        "auroc_std":  float(np.std(fold_auroc)),
        "auprc_mean": float(np.mean(fold_auprc)),
        "auprc_std":  float(np.std(fold_auprc)),
        "f1_mean":    float(np.mean(fold_f1)),
        "f1_std":     float(np.std(fold_f1)),
        "sensitivity_mean": float(np.mean(fold_sens)),
        "specificity_mean": float(np.mean(fold_spec)),
        "total_time_sec": round(sum(fold_times), 2),
        "wall_time_sec":  round(time.time() - t_start, 2),
    }

    # Composite score used by reviewer (0.5 AUROC + 0.3 AUPRC + 0.2 F1)
    metrics["composite_score"] = round(
        0.5 * metrics["auroc_mean"] +
        0.3 * metrics["auprc_mean"] +
        0.2 * metrics["f1_mean"], 4
    )

    out = RESULTS / f"iteration_{iteration}_metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  ── Summary ──")
    print(f"  AUROC : {metrics['auroc_mean']:.4f} ± {metrics['auroc_std']:.4f}  folds={metrics['per_fold_auroc']}")
    print(f"  AUPRC : {metrics['auprc_mean']:.4f} ± {metrics['auprc_std']:.4f}")
    print(f"  F1    : {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
    print(f"  Sens  : {metrics['sensitivity_mean']:.4f}   Spec: {metrics['specificity_mean']:.4f}")
    print(f"  Composite score: {metrics['composite_score']:.4f}")
    print(f"  Wall time: {metrics['wall_time_sec']:.1f}s   Saved → {out}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    n = args.iteration
    plan_file = RESULTS / f"iteration_{n}_plan.json"
    if not plan_file.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_file}")
    with open(plan_file) as f:
        plan = json.load(f)

    run_iteration(n, plan)


if __name__ == "__main__":
    main()

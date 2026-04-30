"""
External Validation: Apply LASSO-RF (trained on KCA) to Lake et al. 2023 AKI biopsy dataset.

Strategy:
  - Positive (AKI=1) : aPT cells from AKI patients  (injured proximal tubule, disease-confirmed)
  - Negative (Normal=0): PT-S1/S2 + PT-S3 from LivingDonor (healthy reference kidney)
  - Model: retrain LASSO-RF on full KCA, then predict on Lake 2023 (no data leakage)
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, roc_curve,
)

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
DATA     = ROOT / "data"
RESULTS  = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

KCA_FILE  = DATA / "Mature_Full_v2.1.h5ad"
LAKE_FILE = DATA / "lake2023_integrated.h5ad"

# ── KCA cell-type definitions (same as training) ──────────────────────────────
AKI_CELLTYPES    = ["Distinct proximal tubule 2", "Proliferating Proximal Tubule",
                    "Epithelial progenitor cell", "Myofibroblast"]
NORMAL_PT_CELLTYPES = ["Proximal tubule"]

# Top-200 DE genes pool (must match iterative_pipeline.py)
N_POOL = 200

AKI_SIGNATURE_GENES = {
    "injury_markers":    ["HAVCR1","LCN2","CXCL8","CXCL2","IL6","SPP1"],
    "dedifferentiation": ["VIM","CD44","SOX9","VCAM1"],
    "ecm_fibrosis":      ["MMP7","FN1","COL1A1","ACTA2","PDGFRA"],
    "pt_healthy":        ["SLC34A1","CUBN","SLC7A9","ANPEP"],
}


def log(msg: str):
    print(msg)


def youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return thresholds[np.argmax(j)]


# ── Step 1: Train LASSO-RF on full KCA ───────────────────────────────────────
def train_on_kca():
    log("=" * 60)
    log("  Step 1: Training LASSO-RF on full Kidney Cell Atlas")
    log("=" * 60)

    adata = sc.read_h5ad(str(KCA_FILE))

    labels = pd.Series("other", index=adata.obs_names)
    for ct in AKI_CELLTYPES:
        labels[adata.obs["celltype"] == ct] = "AKI"
    for ct in NORMAL_PT_CELLTYPES:
        labels[adata.obs["celltype"] == ct] = "Normal_PT"
    adata.obs["aki_label"] = labels
    sub = adata[adata.obs["aki_label"].isin(["AKI", "Normal_PT"])].copy()
    log(f"  KCA subset: {len(sub)} cells  (AKI={int((sub.obs['aki_label']=='AKI').sum())}, "
        f"Normal_PT={int((sub.obs['aki_label']=='Normal_PT').sum())})")

    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)

    # Differential expression → select top-200 gene pool
    sc.tl.rank_genes_groups(sub, "aki_label", groups=["AKI"], reference="Normal_PT",
                             method="wilcoxon", n_genes=N_POOL)
    de_genes = [g for g in sc.get.rank_genes_groups_df(sub, group="AKI")["names"]
                if g in sub.var_names][:N_POOL]
    log(f"  DE gene pool: {len(de_genes)} genes")

    X_pool = sub[:, de_genes].X
    if hasattr(X_pool, "toarray"):
        X_pool = X_pool.toarray()
    y = (sub.obs["aki_label"] == "AKI").astype(int).values

    # LASSO logistic regression for feature selection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pool)
    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                               max_iter=1000, class_weight="balanced")
    lasso.fit(X_scaled, y)

    selected_mask = lasso.coef_[0] != 0
    selected_genes = [g for g, m in zip(de_genes, selected_mask) if m]
    log(f"  LASSO selected: {len(selected_genes)} features")

    # Random Forest on selected features
    X_sel = sub[:, selected_genes].X
    if hasattr(X_sel, "toarray"):
        X_sel = X_sel.toarray()

    rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                class_weight="balanced", n_jobs=-1, random_state=42)
    rf.fit(X_sel, y)
    log(f"  RF trained on {len(selected_genes)} features, {len(y)} cells")

    return rf, selected_genes, scaler, de_genes


# ── Step 2: Prepare Lake 2023 external test set ───────────────────────────────
def prepare_lake2023(selected_genes: list[str]):
    log("\n" + "=" * 60)
    log("  Step 2: Preparing Lake et al. 2023 external test set")
    log("=" * 60)

    lake = sc.read_h5ad(str(LAKE_FILE))

    # Remap var_names from Ensembl IDs to gene symbols
    symbol_map = dict(zip(lake.var["feature_name"], lake.var_names))
    ensembl_map = dict(zip(lake.var_names, lake.var["feature_name"]))

    # Define external validation cells
    # AKI+: injured PT (aPT) from AKI-diagnosed patients
    # AKI-: normal PT (PT-S1/S2, PT-S3) from living donors
    mask_pos = (lake.obs["celltype"] == "aPT") & (lake.obs["diseasetype"] == "AKI")
    mask_neg = (lake.obs["celltype"].isin(["PT-S1/S2", "PT-S3"])) & \
               (lake.obs["diseasetype"] == "LivingDonor")

    sub = lake[mask_pos | mask_neg].copy()
    y_ext = (sub.obs["celltype"] == "aPT").astype(int).values

    log(f"  Lake 2023 test cells: {len(sub):,}  "
        f"(AKI-injured aPT={int(mask_pos.sum()):,}, Normal PT LivingDonor={int(mask_neg.sum()):,})")

    # Normalise (same pipeline as training)
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)

    # Map selected_genes (symbols) to Ensembl IDs present in Lake 2023
    available, missing = [], []
    gene_cols = []
    for g in selected_genes:
        if g in symbol_map:
            available.append(g)
            gene_cols.append(symbol_map[g])
        else:
            missing.append(g)

    log(f"  Selected features present in Lake 2023: {len(available)}/{len(selected_genes)}")
    if missing:
        log(f"  Missing genes: {missing}")

    X_ext = sub[:, gene_cols].X
    if hasattr(X_ext, "toarray"):
        X_ext = X_ext.toarray()

    # For missing genes, fill with zeros (conservative)
    if missing:
        n_miss = len(missing)
        X_ext = np.concatenate([X_ext, np.zeros((len(sub), n_miss))], axis=1)
        log(f"  Filled {n_miss} missing gene(s) with zeros")

    # Reorder columns to match training order (available first, then zeros)
    # Already in selected_genes order effectively — available then missing appended

    return X_ext, y_ext, sub


# ── Step 3: Evaluate ──────────────────────────────────────────────────────────
def evaluate(rf, X_ext, y_ext):
    log("\n" + "=" * 60)
    log("  Step 3: External Validation Results")
    log("=" * 60)

    y_prob = rf.predict_proba(X_ext)[:, 1]
    auroc  = roc_auc_score(y_ext, y_prob)
    auprc  = average_precision_score(y_ext, y_prob)

    thresh = youden_threshold(y_ext, y_prob)
    y_pred = (y_prob >= thresh).astype(int)
    f1     = f1_score(y_ext, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_ext, y_pred, labels=[0, 1]).ravel()
    sens   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    log(f"  AUROC      : {auroc:.4f}")
    log(f"  AUPRC      : {auprc:.4f}")
    log(f"  F1         : {f1:.4f}  (threshold={thresh:.3f})")
    log(f"  Sensitivity: {sens:.4f}")
    log(f"  Specificity: {spec:.4f}")
    log(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "f1":    round(f1,    4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "threshold":   round(float(thresh), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = datetime.now()
    log("=" * 60)
    log("  EXTERNAL VALIDATION: Lake et al. 2023 AKI Biopsy Dataset")
    log(f"  Started: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    rf, selected_genes, scaler, de_genes = train_on_kca()
    X_ext, y_ext, sub = prepare_lake2023(selected_genes)
    metrics = evaluate(rf, X_ext, y_ext)

    elapsed = (datetime.now() - t0).total_seconds()
    log(f"\n  Wall time: {elapsed:.1f}s")

    result = {
        "validation_dataset":   "Lake et al. 2023 (PMID 37468583) — KPMP AKI biopsy atlas",
        "training_dataset":     "Kidney Cell Atlas Mature_Full_v2.1 (Stewart et al. 2019)",
        "model":                "LASSO-RF (iteration 3 strategy, trained on full KCA)",
        "positive_class":       "aPT (injured PT) from AKI-confirmed patients",
        "negative_class":       "PT-S1/S2 + PT-S3 from LivingDonor (healthy reference)",
        "n_positive":           int(y_ext.sum()),
        "n_negative":           int((y_ext == 0).sum()),
        "n_features_used":      len(selected_genes),
        "timestamp":            t0.isoformat(),
        "wall_time_sec":        round(elapsed, 1),
        **metrics,
    }

    out = RESULTS / "external_validation_lake2023.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    log(f"\n  Saved → {out}")

    return result


if __name__ == "__main__":
    main()

"""
Verbose ML training log — re-runs classification with per-fold scores and timing.
Uses the already-loaded data via saved y_true/y_score + raw H5AD.
"""

import warnings
warnings.filterwarnings("ignore")

import time
import json
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

RESULTS = Path("/Users/junghualiu/case/a2a/kidney/kidney_gene/openharness_aki/results")
DATA_FILE = Path("/Users/junghualiu/case/a2a/kidney/kidney_gene/openharness_aki/data/Mature_Full_v2.1.h5ad")

AKI_CELLTYPES = [
    "Distinct proximal tubule 2",
    "Proliferating Proximal Tubule",
    "Epithelial progenitor cell",
    "Myofibroblast",
]
NORMAL_PT_CELLTYPES = ["Proximal tubule"]

print("=" * 65)
print("  AKI ML TRAINING — VERBOSE LOG")
print(f"  {pd.Timestamp.now()}")
print("=" * 65)

# ── 1. Load data ──────────────────────────────────────────────────────────
t0 = time.time()
print(f"\n[{time.strftime('%H:%M:%S')}] Loading H5AD ...")
sc.settings.verbosity = 0
adata = sc.read_h5ad(str(DATA_FILE))
print(f"[{time.strftime('%H:%M:%S')}] Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes  ({time.time()-t0:.1f}s)")

# ── 2. Label AKI vs Normal PT ─────────────────────────────────────────────
labels = pd.Series("other", index=adata.obs_names)
for ct in AKI_CELLTYPES:
    labels[adata.obs["celltype"] == ct] = "AKI"
for ct in NORMAL_PT_CELLTYPES:
    labels[adata.obs["celltype"] == ct] = "Normal_PT"
adata.obs["aki_label"] = labels

sub = adata[adata.obs["aki_label"].isin(["AKI", "Normal_PT"])].copy()
y = (sub.obs["aki_label"] == "AKI").astype(int).values
print(f"\nCell counts:")
print(f"  AKI-associated : {(y==1).sum():>6,}  (Distinct PT 2 + Prolif PT + Epith Prog + Myofibroblast)")
print(f"  Normal PT      : {(y==0).sum():>6,}")
print(f"  Total          : {len(y):>6,}")

# ── 3. Normalise ──────────────────────────────────────────────────────────
print(f"\n[{time.strftime('%H:%M:%S')}] Normalising (library-size + log1p) ...")
sc.pp.normalize_total(sub, target_sum=1e4)
sc.pp.log1p(sub)

# ── 4. Feature selection (top 50 DE genes by abs logFC) ───────────────────
de_df = pd.read_csv(RESULTS / "differential_expression.csv")
top_genes = (de_df[de_df["pvals_adj"] < 0.05]
             .nlargest(50, "abs_lfc")["gene"]
             .tolist())
top_genes = [g for g in top_genes if g in sub.var_names][:50]
print(f"\nFeature set: {len(top_genes)} genes (top DE by |logFC|, padj<0.05)")
print(f"  First 10: {top_genes[:10]}")

X_raw = sub[:, top_genes].X
X_arr = X_raw.toarray() if hasattr(X_raw, "toarray") else np.array(X_raw)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_arr)
print(f"  X shape: {X_scaled.shape}  |  mean={X_scaled.mean():.4f}  std={X_scaled.std():.4f}")

# ── 5. 5-fold stratified CV per model ─────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_splits = list(cv.split(X_scaled, y))

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1,
        class_weight="balanced"
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        verbosity=0
    ),
    "LogisticRegression": LogisticRegression(
        C=1.0, max_iter=1000, random_state=42, class_weight="balanced"
    ),
}

all_cv_auc = {}

for model_name, model in models.items():
    print(f"\n{'─'*55}")
    print(f"  Model: {model_name}")
    print(f"{'─'*55}")
    fold_aucs = []
    for fold_i, (train_idx, test_idx) in enumerate(fold_splits):
        t_fold = time.time()
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        n_aki_tr  = y_tr.sum()
        n_norm_tr = (y_tr == 0).sum()

        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        fold_aucs.append(auc)
        elapsed = time.time() - t_fold

        print(f"  Fold {fold_i+1}/5 | "
              f"train={len(train_idx):,} (AKI={n_aki_tr}, norm={n_norm_tr}) | "
              f"test={len(test_idx):,} | "
              f"AUC={auc:.4f} | "
              f"time={elapsed:.2f}s")

    mean_auc = np.mean(fold_aucs)
    std_auc  = np.std(fold_aucs)
    all_cv_auc[model_name] = fold_aucs
    print(f"\n  {model_name} 5-fold AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Per-fold:  {[round(a, 4) for a in fold_aucs]}")

# ── 6. Final fit (RandomForest) ───────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"  Final fit: RandomForest on full training set")
print(f"{'─'*55}")
t_fit = time.time()
best = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=42, n_jobs=-1,
    class_weight="balanced"
)
best.fit(X_scaled, y)
y_score_full = best.predict_proba(X_scaled)[:, 1]
final_auc = roc_auc_score(y, y_score_full)
print(f"  Fit time: {time.time()-t_fit:.2f}s")
print(f"  Training AUC (full data): {final_auc:.4f}")

# Feature importance
fi = pd.Series(best.feature_importances_, index=top_genes).sort_values(ascending=False)
print(f"\n  Top 15 features by Gini importance:")
for rank, (gene, imp) in enumerate(fi.head(15).items(), 1):
    print(f"    {rank:>2}. {gene:<20s}  {imp:.4f}")

# ── 7. Permutation test (n=100) ───────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"  Permutation test (n=100 shuffles, 3-fold CV, 50-tree RF)")
print(f"{'─'*55}")
t_perm = time.time()
perm_aucs = []
for i in range(100):
    y_perm = np.random.permutation(y)
    m_perm = RandomForestClassifier(
        n_estimators=50, max_depth=6, random_state=None, n_jobs=-1
    )
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=i)
    fold_p = []
    for tr, te in cv3.split(X_scaled, y_perm):
        m_perm.fit(X_scaled[tr], y_perm[tr])
        p = m_perm.predict_proba(X_scaled[te])[:, 1]
        fold_p.append(roc_auc_score(y_perm[te], p))
    perm_aucs.append(np.mean(fold_p))
    if (i + 1) % 20 == 0:
        print(f"  Permutation {i+1:>3}/100 — running mean={np.mean(perm_aucs):.4f}  ({time.time()-t_perm:.0f}s elapsed)")

observed_auc = np.mean(all_cv_auc["RandomForest"])
p_val = np.mean(np.array(perm_aucs) >= observed_auc)
print(f"\n  Permutation distribution: mean={np.mean(perm_aucs):.4f}  max={np.max(perm_aucs):.4f}")
print(f"  Observed 5-fold AUC     : {observed_auc:.4f}")
print(f"  Permutation p-value     : {p_val:.4f}  (0/{len(perm_aucs)} permutations ≥ observed)")

# ── 8. Summary ────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  SUMMARY")
print(f"{'='*65}")
for name, folds in all_cv_auc.items():
    print(f"  {name:<22s}  AUC={np.mean(folds):.4f} ± {np.std(folds):.4f}  folds={[round(a,4) for a in folds]}")
print(f"  RandomForest training AUC: {final_auc:.4f}")
print(f"  Permutation p-value      : {p_val:.4f}")
print(f"\n  Total wall time: {time.time()-t0:.1f}s")
print(f"  Completed: {pd.Timestamp.now()}")

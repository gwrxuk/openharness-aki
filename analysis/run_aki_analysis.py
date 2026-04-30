"""
AKI Gene Expression Analysis Pipeline
Data: Kidney Cell Atlas Mature_Full_v2.1 (Stewart et al. Science 2019)
Method: scRNA-seq differential expression + ML classification
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import json
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                             confusion_matrix, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb

sc.settings.verbosity = 1
DATA_FILE = Path("data/Mature_Full_v2.1.h5ad")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# AKI biomarker panel from published literature
AKI_SIGNATURE_GENES = {
    "injury_markers": ["HAVCR1", "LCN2", "CXCL8", "CXCL2", "IL6", "SPP1"],
    "dedifferentiation": ["VIM", "CD44", "SOX9", "VCAM1"],
    "ecm_fibrosis": ["MMP7", "FN1", "COL1A1", "ACTA2", "PDGFRA"],
    "pt_healthy": ["SLC34A1", "CUBN", "SLC7A9", "ANPEP"],
}

# AKI-associated cell types (injured/maladaptive states)
# Based on Lake et al. Nature 2023: injured PT = dedifferentiated states
AKI_CELLTYPES = [
    "Distinct proximal tubule 2",
    "Proliferating Proximal Tubule",
    "Epithelial progenitor cell",
    "Myofibroblast",
]
NORMAL_PT_CELLTYPES = ["Proximal tubule"]


def load_data():
    print("=== Loading Kidney Cell Atlas Data ===")
    adata = sc.read_h5ad(str(DATA_FILE))
    print(f"  Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}")
    print(f"  Source: Stewart et al. Science 2019 / Mature_Full_v2.1.h5ad")
    return adata


def define_aki_labels(adata):
    """Label cells as AKI-associated (1) vs normal PT (0)."""
    labels = pd.Series("other", index=adata.obs_names)
    for ct in AKI_CELLTYPES:
        mask = adata.obs["celltype"] == ct
        labels[mask] = "AKI"
    for ct in NORMAL_PT_CELLTYPES:
        mask = adata.obs["celltype"] == ct
        labels[mask] = "Normal_PT"
    adata.obs["aki_label"] = labels
    n_aki = (labels == "AKI").sum()
    n_norm = (labels == "Normal_PT").sum()
    print(f"  AKI-associated cells: {n_aki:,}")
    print(f"  Normal PT cells: {n_norm:,}")
    return adata


def normalize(adata):
    print("\n=== Normalization ===")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    print("  Library-size normalization + log1p applied")
    return adata


def differential_expression(adata):
    print("\n=== Differential Expression (AKI vs Normal PT) ===")
    sub = adata[adata.obs["aki_label"].isin(["AKI", "Normal_PT"])].copy()
    sc.tl.rank_genes_groups(
        sub, groupby="aki_label", groups=["AKI"],
        reference="Normal_PT", method="wilcoxon",
        n_genes=sub.n_vars
    )
    de_df = sc.get.rank_genes_groups_df(sub, group="AKI")
    de_df.columns = ["gene", "scores", "logfoldchanges", "pvals", "pvals_adj"]
    de_df["abs_lfc"] = de_df["logfoldchanges"].abs()

    # Add known gene category
    all_sig = [g for gs in AKI_SIGNATURE_GENES.values() for g in gs]
    de_df["aki_biomarker"] = de_df["gene"].isin(all_sig)

    out = RESULTS / "differential_expression.csv"
    de_df.to_csv(out, index=False)
    print(f"  Top AKI upregulated genes:")
    top_up = de_df[de_df["pvals_adj"] < 0.05].nlargest(10, "logfoldchanges")
    for _, row in top_up.iterrows():
        print(f"    {row['gene']:20s}  logFC={row['logfoldchanges']:+.3f}  padj={row['pvals_adj']:.2e}")
    print(f"  Saved {len(de_df)} genes → {out}")
    return de_df


def compute_aki_score(adata):
    print("\n=== AKI Signature Score ===")
    all_present = {}
    for category, genes in AKI_SIGNATURE_GENES.items():
        present = [g for g in genes if g in adata.var_names]
        all_present[category] = present
        sc.tl.score_genes(adata, present, score_name=f"score_{category}")

    # Composite injury score = injury + dediff - healthy_pt
    adata.obs["AKI_injury_score"] = (
        adata.obs["score_injury_markers"] +
        adata.obs["score_dedifferentiation"] +
        adata.obs["score_ecm_fibrosis"] -
        adata.obs["score_pt_healthy"]
    )

    score_df = adata.obs[["celltype", "aki_label", "AKI_injury_score",
                           "score_injury_markers", "score_dedifferentiation",
                           "score_ecm_fibrosis", "score_pt_healthy"]].copy()
    out = RESULTS / "aki_signature_scores.csv"
    score_df.to_csv(out)

    # Summary by cell type
    summary = score_df.groupby("celltype")["AKI_injury_score"].agg(["mean", "std", "count"])
    summary = summary.sort_values("mean", ascending=False)
    print("  AKI injury score by cell type (top 10):")
    print(summary.head(10).to_string())
    return adata, score_df


def ml_classification(adata):
    print("\n=== Machine Learning Classification (AKI vs Normal PT) ===")
    sub = adata[adata.obs["aki_label"].isin(["AKI", "Normal_PT"])].copy()
    y = (sub.obs["aki_label"] == "AKI").astype(int).values

    # Feature matrix: expression of AKI-related gene panel
    all_sig_genes = [g for gs in AKI_SIGNATURE_GENES.values() for g in gs
                     if g in sub.var_names]
    X_panel = sub[:, all_sig_genes].X.toarray() if hasattr(sub[:, all_sig_genes].X, 'toarray') \
              else np.array(sub[:, all_sig_genes].X)

    # Also get top DE genes for extended model
    de_df = pd.read_csv(RESULTS / "differential_expression.csv")
    top_genes = de_df[de_df["pvals_adj"] < 0.05].nlargest(50, "abs_lfc")["gene"].tolist()
    top_genes = [g for g in top_genes if g in sub.var_names][:50]
    X_top = sub[:, top_genes].X.toarray() if hasattr(sub[:, top_genes].X, 'toarray') \
            else np.array(sub[:, top_genes].X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_top)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1,
            class_weight="balanced"
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            eval_metric="logloss", random_state=42,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1)
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, class_weight="balanced"
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = {}

    for name, model in models.items():
        print(f"\n  --- {name} ---")
        X_use = X_panel if name == "LogisticRegression" else X_scaled
        aucs = cross_val_score(model, X_use, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"  5-fold ROC-AUC: {aucs.mean():.4f} ± {aucs.std():.4f}")
        all_results[name] = {"cv_auc_mean": float(aucs.mean()), "cv_auc_std": float(aucs.std())}

    # Full fit with best model (RandomForest) for feature importance
    best_model = models["RandomForest"]
    best_model.fit(X_scaled, y)
    y_score = best_model.predict_proba(X_scaled)[:, 1]
    final_auc = roc_auc_score(y, y_score)
    print(f"\n  Best model (RandomForest) training AUC: {final_auc:.4f}")

    # Feature importance
    fi_df = pd.DataFrame({
        "gene": top_genes,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(RESULTS / "feature_importance.csv", index=False)

    # Permutation test (n=100)
    print("  Running permutation test (n=100)...")
    perm_aucs = []
    for _ in range(100):
        y_perm = np.random.permutation(y)
        m = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=None, n_jobs=-1)
        perm_cv = cross_val_score(m, X_scaled, y_perm, cv=3, scoring="roc_auc")
        perm_aucs.append(perm_cv.mean())
    observed = all_results["RandomForest"]["cv_auc_mean"]
    p_val = np.mean(np.array(perm_aucs) >= observed)
    print(f"  Permutation p-value: {p_val:.4f}")

    # Save detailed results
    results = {
        "roc_auc": final_auc,
        "cv_results": all_results,
        "permutation_p": float(p_val),
        "n_features": len(top_genes),
        "n_cells_aki": int((y == 1).sum()),
        "n_cells_normal": int((y == 0).sum()),
        "y_true": y.tolist(),
        "y_score": y_score.tolist(),
        "top_features": fi_df.head(15)["gene"].tolist(),
    }
    with open(RESULTS / "model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, fi_df, y, y_score


def plot_results(adata, de_df, fi_df, y, y_score):
    print("\n=== Generating Figures ===")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("AKI Gene Expression Analysis — Kidney Cell Atlas", fontsize=14, fontweight="bold")

    # 1. AKI injury score distribution by compartment
    ax = axes[0, 0]
    plot_df = adata.obs[adata.obs["aki_label"].isin(["AKI", "Normal_PT"])].copy()
    groups = {"Normal PT": plot_df[plot_df["aki_label"] == "Normal_PT"]["AKI_injury_score"],
              "AKI-associated": plot_df[plot_df["aki_label"] == "AKI"]["AKI_injury_score"]}
    bp = ax.boxplot(groups.values(), labels=groups.keys(), patch_artist=True)
    colors_box = ["#4e9af1", "#f7622b"]
    for patch, col in zip(bp["boxes"], colors_box):
        patch.set_facecolor(col)
    _, pval = stats.mannwhitneyu(groups["AKI-associated"], groups["Normal PT"])
    ax.set_title(f"AKI Injury Score (p={pval:.2e})")
    ax.set_ylabel("Composite AKI Score")

    # 2. Top AKI genes heatmap (gene × celltype mean expression)
    ax = axes[0, 1]
    key_genes = ["HAVCR1", "LCN2", "SOX9", "VIM", "CD44", "SPP1",
                 "MMP7", "FN1", "SLC34A1", "CUBN", "UMOD"]
    key_genes = [g for g in key_genes if g in adata.var_names]
    ct_order = ["Proximal tubule", "Distinct proximal tubule 2",
                "Proliferating Proximal Tubule", "Epithelial progenitor cell"]
    ct_order = [c for c in ct_order if c in adata.obs["celltype"].unique()]
    sub = adata[adata.obs["celltype"].isin(ct_order), key_genes].copy()
    heat_df = pd.DataFrame(
        sub.X.toarray() if hasattr(sub.X, 'toarray') else np.array(sub.X),
        index=sub.obs["celltype"].values, columns=key_genes
    ).groupby(level=0).mean()
    heat_df = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-6)
    sns.heatmap(heat_df.T, ax=ax, cmap="YlOrRd", linewidths=0.5,
                xticklabels=[c[:20] for c in heat_df.index], yticklabels=True)
    ax.set_title("AKI Gene Panel — Cell Type Expression")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)

    # 3. Volcano plot
    ax = axes[0, 2]
    sig = de_df["pvals_adj"] < 0.05
    ax.scatter(de_df.loc[~sig, "logfoldchanges"], -np.log10(de_df.loc[~sig, "pvals_adj"] + 1e-300),
               alpha=0.3, s=3, color="grey", label="NS")
    ax.scatter(de_df.loc[sig & (de_df["logfoldchanges"] > 0), "logfoldchanges"],
               -np.log10(de_df.loc[sig & (de_df["logfoldchanges"] > 0), "pvals_adj"] + 1e-300),
               alpha=0.7, s=5, color="#d62728", label="AKI up")
    ax.scatter(de_df.loc[sig & (de_df["logfoldchanges"] < 0), "logfoldchanges"],
               -np.log10(de_df.loc[sig & (de_df["logfoldchanges"] < 0), "pvals_adj"] + 1e-300),
               alpha=0.7, s=5, color="#1f77b4", label="AKI down")
    # Label top genes
    for _, row in de_df.nlargest(8, "logfoldchanges").iterrows():
        ax.annotate(row["gene"], (row["logfoldchanges"], -np.log10(row["pvals_adj"] + 1e-300)),
                    fontsize=7, ha="left")
    ax.axhline(-np.log10(0.05), color="k", ls="--", lw=0.8)
    ax.axvline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("log2 Fold Change (AKI / Normal PT)")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title("Volcano Plot: AKI vs Normal PT")
    ax.legend(fontsize=7)

    # 4. ROC curve
    ax = axes[1, 0]
    fpr, tpr, _ = roc_curve(y, y_score)
    auc_val = roc_auc_score(y, y_score)
    ax.plot(fpr, tpr, color="#d62728", lw=2, label=f"Random Forest (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — AKI Classification")
    ax.legend(fontsize=9)

    # 5. Feature importance
    ax = axes[1, 1]
    fi_top = fi_df.head(15)
    colors = ["#d62728" if g in [g2 for gs in AKI_SIGNATURE_GENES.values() for g2 in gs]
              else "#1f77b4" for g in fi_top["gene"]]
    ax.barh(fi_top["gene"][::-1], fi_top["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Top 15 Predictive Genes")
    ax.tick_params(axis="y", labelsize=8)

    # 6. Cell type composition
    ax = axes[1, 2]
    ct_counts = adata.obs["celltype"].value_counts().head(12)
    colors_pie = plt.cm.tab20.colors[:len(ct_counts)]
    ax.barh(ct_counts.index[::-1], ct_counts.values[::-1], color=list(colors_pie)[::-1])
    ax.set_xlabel("Number of Cells")
    ax.set_title("Cell Type Composition (n=40,268)")
    ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    out = RESULTS / "aki_analysis_figures.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved figures → {out}")
    plt.close()


def generate_summary(results):
    cv = results["cv_results"]
    summary = {
        "dataset": "Kidney Cell Atlas Mature_Full_v2.1 (Stewart et al. Science 2019)",
        "n_cells": 40268,
        "n_genes": 33694,
        "aki_cells": results["n_cells_aki"],
        "normal_pt_cells": results["n_cells_normal"],
        "best_model": "RandomForest",
        "training_auc": results["roc_auc"],
        "cv_rf_auc": cv["RandomForest"]["cv_auc_mean"],
        "cv_rf_std": cv["RandomForest"]["cv_auc_std"],
        "cv_xgb_auc": cv["XGBoost"]["cv_auc_mean"],
        "cv_lr_auc": cv["LogisticRegression"]["cv_auc_mean"],
        "permutation_p": results["permutation_p"],
        "top_genes": results["top_features"][:10],
    }
    with open(RESULTS / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== FINAL RESULTS SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


def main():
    adata = load_data()
    adata = define_aki_labels(adata)
    adata = normalize(adata)
    de_df = differential_expression(adata)
    adata, score_df = compute_aki_score(adata)
    results, fi_df, y, y_score = ml_classification(adata)
    plot_results(adata, de_df, fi_df, y, y_score)
    summary = generate_summary(results)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

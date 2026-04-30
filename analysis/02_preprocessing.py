"""
Step 2: Preprocessing — QC, normalization, HVG selection, dimensionality reduction
Follows standard Seurat/Scanpy pipeline (Luecken & Theis, 2019 best practices)
"""

import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path

sc.settings.verbosity = 2
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# AKI biomarker genes (Lake et al. Nature 2023; Humphreys, Ann Rev Physiol 2018)
AKI_GENES = [
    "HAVCR1",   # KIM-1 — proximal tubule injury (Han et al. 2002)
    "LCN2",     # NGAL — early AKI biomarker (Mishra et al. 2003)
    "CXCL8",    # IL-8 — neutrophil chemokine
    "MMP7",     # matrix metalloproteinase — fibrotic transition
    "SOX9",     # injury-response transcription factor (Kumar et al. 2015)
    "VIM",      # vimentin — dedifferentiation marker
    "CD44",     # injury-induced cell surface receptor
    "VCAM1",    # vascular cell adhesion — inflammatory state
    "IL6",      # interleukin-6 — systemic inflammation
    "CXCL2",    # chemokine — neutrophil recruitment
    "SPP1",     # osteopontin — tubular injury/repair
    "FN1",      # fibronectin — ECM remodeling
    "PDGFRA",   # pericyte activation — fibrosis
    "ACTA2",    # alpha-SMA — myofibroblast
    "COL1A1",   # collagen I — fibrosis
]

def main():
    print("=== Step 2: Preprocessing ===")
    h5ad = DATA_DIR / "Mature_Full_v3.h5ad"
    print(f"  Loading {h5ad}...")
    adata = sc.read_h5ad(str(h5ad))
    print(f"  Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    print(f"  Obs columns: {list(adata.obs.columns)}")

    # --- Quality control ---
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # Filter: remove low-quality cells
    before = adata.n_obs
    adata = adata[adata.obs.n_genes_by_counts > 200, :]
    adata = adata[adata.obs.pct_counts_mt < 20, :]
    print(f"  QC: {before:,} → {adata.n_obs:,} cells after filtering")

    # --- Normalization ---
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    # --- Highly variable genes ---
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=3000)
    n_hvg = adata.var.highly_variable.sum()
    print(f"  HVG selection: {n_hvg} highly variable genes")

    # Force inclusion of AKI marker genes
    for gene in AKI_GENES:
        if gene in adata.var_names:
            adata.var.loc[gene, "highly_variable"] = True

    # --- PCA + UMAP ---
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, svd_solver="arpack", n_comps=50)
    sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata_hvg)
    sc.tl.leiden(adata_hvg, resolution=0.5)

    # Transfer UMAP and leiden back to full adata
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
    adata.obs["leiden"] = adata_hvg.obs["leiden"]

    # --- Save ---
    out = RESULTS_DIR / "preprocessed_adata.h5ad"
    adata.write_h5ad(str(out))
    print(f"  Saved preprocessed data: {out}")

    # Summary stats
    stats = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_hvg": int(n_hvg),
        "n_clusters": int(adata.obs["leiden"].nunique()),
        "cell_types": adata.obs.get("celltype", pd.Series()).value_counts().to_dict()
    }
    pd.DataFrame([stats]).to_csv(RESULTS_DIR / "preprocessing_stats.csv", index=False)
    print("  Preprocessing complete.")

if __name__ == "__main__":
    main()

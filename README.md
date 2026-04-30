# OpenHarness AKI — Multi-Agent Single-Cell AKI Classification Pipeline

A multi-agent machine learning pipeline for classifying Acute Kidney Injury (AKI)-associated cell states from single-cell RNA-seq data, built on the [OpenHarness](https://openharness.ai) agent framework.

## Goal

Identify a transcriptomic signature that distinguishes AKI-associated cell states (injured/maladaptive proximal tubule, epithelial progenitors, myofibroblasts) from healthy proximal tubule cells in the [Kidney Cell Atlas](https://www.science.org/doi/10.1126/science.aat5031) (Stewart et al., *Science* 2019), and validate the resulting classifier on an independent human AKI biopsy dataset (Lake et al., *Nature Medicine* 2023).

**Best result**: LASSO-RF, AUROC = 0.953 (5-fold CV) → externally validated on Lake et al. 2023.

---

## Architecture

```
SearchAgent
    │  PubMed query (NCBI E-utilities)
    │  Extracts benchmark AUROCs from literature
    ▼
PlanAgent  ──────────────────────────────────────────────┐
    │  Selects ML strategy for each iteration             │
    │  (RF, XGBoost, LASSO-RF, PCA-GBT, ensembles…)     │
    ▼                                                      │
CodingAgent                                               │ REPLAN
    │  Executes iterative_pipeline.py with plan config    │
    │  Runs 5-fold cross-validation                       │
    ▼                                                      │
ReviewerAgent ───────────────────────────────────────────┘
    │  Compares metrics vs. best-so-far
    │  Decision: ACCEPT / REPLAN / COMPLETE
    ▼
final_report.json  +  orchestrator_full_log.txt
```

The orchestrator runs **5–7 iterations** minimum, continuing past 5 if the reviewer signals `REPLAN` and budget remains (hard cap at 7).

### Two execution modes

| Mode | Description |
|------|-------------|
| `standalone` | Agents run as Python functions in-process (no API key needed) |
| `oh` | Agents invoked via `oh` CLI subprocesses (requires `OPENAI_API_KEY`) |

---

## Repository Layout

```
openharness_aki/
├── agents/
│   ├── orchestrator.py        # Main loop: SearchAgent → PlanAgent → CodingAgent → ReviewerAgent
│   ├── coding_agent.py        # Executes iterative_pipeline with a given plan config
│   ├── plan_agent.py          # Proposes ML strategy per iteration
│   ├── reviewer_agent.py      # Evaluates metrics; returns ACCEPT / REPLAN / COMPLETE
│   ├── pubmed_search.py       # NCBI E-utilities wrapper for literature search
│   └── test_agent.py          # Smoke-test harness
│
├── analysis/
│   ├── iterative_pipeline.py  # Core ML pipeline (DE → LASSO → RF/XGB/etc., 5-fold CV)
│   ├── external_validation.py # Retrain on full KCA; predict on Lake et al. 2023
│   ├── run_aki_analysis.py    # Single-run entry point (standalone)
│   ├── verbose_training_log.py# Detailed per-fold training diagnostics
│   ├── 01_data_fetch.py       # Download KCA & Lake 2023 datasets
│   └── 02_preprocessing.py    # QC filtering, normalisation, HVG selection
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose-iterative.yml   # Full multi-agent run (oh mode)
│   ├── docker-compose.yml             # Standalone run
│   └── patch_mcp.py                   # MCP server patching utility
│
├── results/
│   ├── final_report.json              # Summary of all 7 iterations + best model
│   ├── feature_importance.csv         # RF Gini importances (top 50 genes)
│   ├── differential_expression.csv    # Wilcoxon DE results (AKI vs Normal PT)
│   ├── aki_signature_scores.csv       # Per-cell-type composite injury scores
│   ├── external_validation_lake2023.json  # Hold-out validation metrics
│   ├── iteration_*_metrics.json       # Per-iteration CV metrics
│   ├── iteration_*_plan.json          # ML strategy chosen per iteration
│   ├── iteration_*_review.json        # Reviewer decision per iteration
│   └── fetched_studies/               # PubMed abstracts + PDFs
│
├── data/                              # (empty — large .h5ad files excluded)
├── write_paper.py                     # Generates DHA2026 submission document
├── .gitignore
└── README.md
```

---

## Cell Type Label Mapping

Labels are derived from pre-existing `celltype` annotations in the Kidney Cell Atlas (`obs['celltype']`), not from clinical AKI diagnosis records.

| Class | KCA cell types |
|-------|---------------|
| **AKI (1)** | Distinct proximal tubule 2, Proliferating Proximal Tubule, Epithelial progenitor cell, Myofibroblast |
| **Normal PT (0)** | Proximal tubule |

> **Note**: The KCA is derived from nephrectomy specimens (surgical resections), not from patients with confirmed AKI. The "AKI" labels represent maladaptive and injury-associated cell states as annotated by the original atlas authors.

---

## Iteration Results

| Iter | Approach | AUROC | AUPRC | Decision |
|------|----------|-------|-------|----------|
| 1 | RF-Baseline | 0.884 ± 0.019 | 0.713 | ACCEPT |
| 2 | XGB-Extended | 0.921 ± 0.015 | 0.817 | ACCEPT |
| **3** | **LASSO-RF** | **0.953 ± 0.010** | **0.827** | **ACCEPT** |
| 4 | PCA-GBT | 0.891 ± 0.012 | 0.643 | REPLAN |
| 5 | MultiFeature-Vote | 0.923 ± 0.015 | 0.802 | REPLAN |
| 6 | Stack-RF-XGB | 0.903 ± 0.021 | 0.771 | REPLAN |
| 7 | TunedRF-Best | 0.905 ± 0.021 | 0.769 | COMPLETE |

**Best model (Iter 3 — LASSO-RF)**:
- 200-gene DE pool (Wilcoxon, AKI vs Normal PT)
- LASSO logistic regression selects 136 features
- Random Forest (200 trees, max depth 8, balanced class weight)
- Top feature: TACSTD2 (Gini importance 0.218, highly expressed in Epithelial progenitor cells)

**External validation (Lake et al. 2023)**: aPT cells from AKI patients vs. normal PT from living donors.

---

## Quick Start

### Standalone (no API key)

```bash
# Install dependencies
pip install scanpy scikit-learn xgboost pandas numpy

# Place data files in data/
# data/Mature_Full_v2.1.h5ad  (Kidney Cell Atlas)
# data/lake2023_integrated.h5ad  (Lake et al. 2023)

# Run full iterative pipeline
python agents/orchestrator.py --mode standalone --min-iterations 5 --max-iterations 7

# Run external validation
python analysis/external_validation.py
```

### Docker (OpenHarness OH mode)

```bash
export OPENAI_API_KEY=sk-...
docker compose -f docker/docker-compose-iterative.yml up --build
```

Results are written to `results/`.

---

## Data Sources

| Dataset | Description | Access |
|---------|-------------|--------|
| Kidney Cell Atlas v2.1 | 40,268 human kidney cells, nephrectomy (Stewart et al., *Science* 2019, PMID 31604275) | [HCA Data Portal](https://www.humancellatlas.org/) |
| Lake et al. 2023 | ~75,000 human AKI biopsy cells, snRNA-seq (PMID 37468583) | [KPMP](https://kpmp.org/) |

Large `.h5ad` files are excluded from this repository. Download them from the sources above and place in `data/`.

---

## Citation

If you use this pipeline, please cite:

- Stewart BJ et al. (2019) Spatiotemporal immune zonation of the human kidney. *Science* 366(6460):359–363. PMID 31604275
- Lake BB et al. (2023) An atlas of healthy and injured cell states and niches in the human kidney. *Nature Medicine* 29:2585–2599. PMID 37468583

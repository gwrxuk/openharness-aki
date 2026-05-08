"""
OmicsDiscovery Agent — searches PubMed and NCBI GEO for kidney-related omics
journals and datasets, then recommends machine learning strategies per modality.

Pipeline:
  1. PubMed search — broad kidney omics literature (scRNA-seq, proteomics,
     metabolomics, epigenomics, GWAS, multi-omics)
  2. GEO/SRA dataset search — NCBI eSearch in 'gds' and 'sra' databases
  3. Modality classification — tag each hit with its omics type
  4. ML model recommendation — map modality → best-fit model family
  5. Output — results/omics_discovery.json + per-dataset files in
     results/omics_datasets/

Integrates with orchestrator.py: the returned `recommendation` dict feeds
into plan_agent.py to bias strategy selection toward better-performing modalities.
"""

import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
RESULTS     = ROOT / "results"
DATASETS_DIR = RESULTS / "omics_datasets"
NCBI_BASE   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ── Omics modality taxonomy ───────────────────────────────────────────────────
MODALITY_PATTERNS = {
    "scRNA-seq": [
        r"single.cell RNA.seq", r"scRNA.seq", r"10[xX] [Gg]enomics",
        r"droplet.based transcriptom", r"single.nucleus RNA",
    ],
    "snRNA-seq": [
        r"snRNA.seq", r"single.nucleus RNA", r"snATAC",
    ],
    "bulk_RNA-seq": [
        r"RNA.seq(?!.*single)", r"transcriptom(?!.*single.cell)", r"mRNA.seq",
        r"bulk RNA", r"gene expression profil",
    ],
    "proteomics": [
        r"proteom", r"mass spectrometry", r"LC.MS", r"iTRAQ", r"TMT",
        r"label.free quantif", r"DIA.MS", r"SILAC",
    ],
    "metabolomics": [
        r"metabolom", r"metabolite", r"NMR spectroscop", r"GC.MS",
        r"lipidomi", r"lipidom",
    ],
    "epigenomics": [
        r"ATAC.seq", r"ChIP.seq", r"methylat", r"bisulfite.seq",
        r"chromatin accessib", r"histone modif",
    ],
    "genomics": [
        r"GWAS", r"genome.wide association", r"whole.exome", r"WES",
        r"whole.genome seq", r"WGS", r"SNP", r"copy number variat",
    ],
    "spatial_transcriptomics": [
        r"spatial transcriptom", r"Visium", r"MERFISH", r"seqFISH",
        r"spatially.resolved",
    ],
    "multi-omics": [
        r"multi.om", r"multi.modal", r"integrat.*om", r"proteogenomic",
        r"integrat.*transcriptom.*proteom",
    ],
    "microbiome": [
        r"microbiom", r"16S rRNA", r"metagenom", r"gut microb",
    ],
}

# ── ML model recommendations per modality ─────────────────────────────────────
MODALITY_ML_MAP = {
    "scRNA-seq": {
        "preferred_models": ["random_forest", "xgboost", "lasso_rf", "scvi_latent_xgb"],
        "feature_strategies": ["top_de", "lasso", "hvg_pca", "scvi_latent"],
        "rationale": (
            "High-dimensional sparse count data. LASSO feature selection or "
            "variational autoencoder (scVI) latent embeddings before tree-based "
            "classifiers outperform linear models. Class imbalance common."
        ),
        "target_metric": "AUROC",
    },
    "snRNA-seq": {
        "preferred_models": ["random_forest", "xgboost", "lasso_rf"],
        "feature_strategies": ["top_de", "lasso", "pca"],
        "rationale": (
            "Similar to scRNA-seq but nuclear fractions differ. "
            "Same LASSO→RF pipeline applies; expect slightly lower AUROC "
            "due to nuclear vs. cytoplasmic RNA differences."
        ),
        "target_metric": "AUROC",
    },
    "bulk_RNA-seq": {
        "preferred_models": ["elastic_net", "svm_rbf", "random_forest", "deseq2_lasso"],
        "feature_strategies": ["deseq2_de", "lasso", "pca_50"],
        "rationale": (
            "Lower dimensionality than single-cell. Elastic-net regularisation "
            "handles collinear gene expression well. SVM with RBF kernel often "
            "competitive. DE gene pre-selection still beneficial."
        ),
        "target_metric": "AUROC",
    },
    "proteomics": {
        "preferred_models": ["elastic_net", "random_forest", "xgboost", "pls_da"],
        "feature_strategies": ["variance_filter", "lasso", "pca"],
        "rationale": (
            "Typically 3,000–10,000 proteins, moderate missingness. "
            "Imputation step required. Elastic-net or PLS-DA for direct "
            "classification; RF/XGB for non-linear interactions. "
            "Protein co-expression modules as features can boost performance."
        ),
        "target_metric": "AUROC",
    },
    "metabolomics": {
        "preferred_models": ["pls_da", "svm_rbf", "random_forest", "elastic_net"],
        "feature_strategies": ["univariate_filter", "pca", "lasso"],
        "rationale": (
            "~100–2,000 metabolite features, strong batch effects. "
            "PLS-DA is the field standard. After normalisation, "
            "RF/XGB can outperform PLS-DA for complex phenotypes. "
            "Pathway enrichment scores as engineered features."
        ),
        "target_metric": "AUROC",
    },
    "epigenomics": {
        "preferred_models": ["lasso_rf", "gradient_boosting", "xgboost"],
        "feature_strategies": ["peak_de", "motif_scores", "pca"],
        "rationale": (
            "ATAC-seq peaks or methylation loci: highly sparse, >100 k features. "
            "Aggressive feature selection (top differential peaks) is mandatory. "
            "Gradient boosting on peak accessibility scores often best."
        ),
        "target_metric": "AUROC",
    },
    "genomics": {
        "preferred_models": ["lasso", "elastic_net", "gradient_boosting", "polygenic_risk"],
        "feature_strategies": ["ld_pruned_snps", "prs_score", "pca_ancestry"],
        "rationale": (
            "Millions of SNPs: mandatory LD pruning + pre-selection. "
            "Polygenic risk scores competitive with ML for common disease. "
            "XGB on filtered SNP sets for rare-variant analysis."
        ),
        "target_metric": "AUC",
    },
    "spatial_transcriptomics": {
        "preferred_models": ["xgboost", "random_forest", "gnn_spatial"],
        "feature_strategies": ["spatial_de", "neighborhood_features", "pca"],
        "rationale": (
            "Spatial coordinates add a graph-based feature dimension. "
            "Graph neural networks (GNN) can incorporate neighbor information. "
            "For simpler baselines, XGB on spatially DE genes with "
            "neighbourhood composition features."
        ),
        "target_metric": "AUROC",
    },
    "multi-omics": {
        "preferred_models": ["mofa_xgb", "snf_rf", "late_fusion_vote", "early_fusion_lasso"],
        "feature_strategies": ["mofa_latent", "snf_graph", "concatenate_pca", "lasso_each_block"],
        "rationale": (
            "Multi-block integration: MOFA+ or SNF latent factors as features, "
            "then standard classifier. Late fusion (train per-modality, vote) "
            "is robust when sample sizes differ across modalities. "
            "Early concatenation + LASSO is a strong baseline."
        ),
        "target_metric": "AUROC",
    },
    "microbiome": {
        "preferred_models": ["random_forest", "elastic_net", "aldex2_lasso"],
        "feature_strategies": ["clr_transform", "top_taxa", "pca"],
        "rationale": (
            "Compositional count data: CLR transformation before standard ML. "
            "Random Forest on CLR-transformed OTU tables typically best. "
            "Elastic-net on genus-level aggregates for interpretability."
        ),
        "target_metric": "AUROC",
    },
}

# ── PubMed queries — kidney omics, broad ─────────────────────────────────────
OMICS_QUERIES = [
    # Transcriptomics
    "kidney AKI single-cell RNA-seq omics machine learning classification",
    "renal tubular injury transcriptomics biomarker prediction model",
    "acute kidney injury proteomics metabolomics machine learning",
    # Epigenomics / multi-omics
    "kidney disease ATAC-seq epigenomics chromatin accessibility machine learning",
    "chronic kidney disease multi-omics integration GWAS eQTL",
    # Spatial / emerging
    "kidney spatial transcriptomics cell type deconvolution",
    "AKI CKD metabolomics urine biomarker random forest XGBoost",
    # Proteomics
    "kidney injury urinary proteomics mass spectrometry deep learning",
]

# GEO search terms for kidney omics datasets
GEO_QUERIES = [
    "kidney AKI scRNA-seq",
    "renal tubular injury RNA-seq",
    "acute kidney injury proteomics",
    "kidney ATAC-seq chromatin",
    "renal metabolomics",
    "kidney spatial transcriptomics",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(url: str, params: dict, retries: int = 3, timeout: int = 20) -> requests.Response:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def classify_modality(title: str, abstract: str) -> list[str]:
    """Return list of detected omics modalities (may be multiple for multi-omics papers)."""
    text = (title + " " + abstract).lower()
    found = []
    for modality, patterns in MODALITY_PATTERNS.items():
        if any(re.search(p, text, re.IGNORECASE) for p in patterns):
            found.append(modality)
    # Collapse: if both scRNA-seq and snRNA-seq detected, keep both
    # If multi-omics explicitly detected, ensure it's listed
    if len(found) >= 3 and "multi-omics" not in found:
        found.append("multi-omics")
    return found if found else ["unknown"]


def extract_sample_size(abstract: str) -> str:
    """Parse n= or sample count from abstract text."""
    patterns = [
        r"n\s*=\s*(\d[\d,]+)",
        r"(\d[\d,]+)\s+(?:patients?|subjects?|participants?|donors?|samples?|cells?)",
        r"cohort of\s+(\d[\d,]+)",
    ]
    for pat in patterns:
        m = re.search(pat, abstract, re.IGNORECASE)
        if m:
            return m.group(1).replace(",", "")
    return "unknown"


def score_ml_suitability(abstract: str, modality: list[str]) -> float:
    """
    0–1 score for how suitable this dataset/paper is for ML classification.
    Criteria: mentions classification/prediction, sample size parseable,
    known ML-friendly modality, includes AKI/CKD endpoint.
    """
    score = 0.0
    text = abstract.lower()

    # Classification or prediction language
    if re.search(r"classif|predict|model|machine learning|deep learning|random forest|XGBoost", text, re.I):
        score += 0.3
    # Has a clinical/biological endpoint
    if re.search(r"\bAKI\b|acute kidney injury|CKD|chronic kidney|renal failure|ESRD", text, re.I):
        score += 0.25
    # Sample size parseable and reasonable
    n_str = extract_sample_size(abstract)
    if n_str != "unknown":
        n = int(n_str)
        if n >= 20:
            score += 0.15
        if n >= 100:
            score += 0.10
    # ML-friendly modality
    good_modalities = {"scRNA-seq", "snRNA-seq", "bulk_RNA-seq", "proteomics",
                       "metabolomics", "multi-omics"}
    if any(m in good_modalities for m in modality):
        score += 0.20
    return round(min(score, 1.0), 3)


# ── PubMed functions ──────────────────────────────────────────────────────────

def pubmed_search(query: str, retmax: int = 10) -> list[str]:
    r = _get(f"{NCBI_BASE}/esearch.fcgi",
             {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json"})
    return r.json()["esearchresult"]["idlist"]


def pubmed_fetch_abstracts(pmids: list[str]) -> str:
    if not pmids:
        return ""
    r = _get(f"{NCBI_BASE}/efetch.fcgi",
             {"db": "pubmed", "id": ",".join(pmids),
              "rettype": "abstract", "retmode": "text"})
    return r.text


def pubmed_fetch_metadata(pmids: list[str]) -> dict[str, dict]:
    if not pmids:
        return {}
    r = _get(f"{NCBI_BASE}/esummary.fcgi",
             {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"})
    result = r.json().get("result", {})
    meta = {}
    for pmid in pmids:
        item = result.get(pmid, {})
        if not item or pmid == "uids":
            continue
        authors_raw = item.get("authors", [])
        authors = [a.get("name", "") for a in authors_raw[:5]]
        if len(authors_raw) > 5:
            authors.append("et al.")
        article_ids = item.get("articleids", [])
        meta[pmid] = {
            "pmid":    pmid,
            "title":   item.get("title", ""),
            "authors": authors,
            "journal": item.get("source", ""),
            "year":    item.get("pubdate", "")[:4],
            "doi":     next((i["value"] for i in article_ids if i.get("idtype") == "doi"), ""),
            "pmcid":   next((i["value"] for i in article_ids if i.get("idtype") == "pmc"), ""),
            "url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
    return meta


# ── GEO dataset search ────────────────────────────────────────────────────────

def geo_search(query: str, retmax: int = 8) -> list[str]:
    """Search NCBI GEO DataSets (gds) and return GDS UIDs."""
    try:
        r = _get(f"{NCBI_BASE}/esearch.fcgi",
                 {"db": "gds", "term": query + "[Title/Abstract]",
                  "retmax": retmax, "retmode": "json"})
        return r.json()["esearchresult"]["idlist"]
    except Exception:
        return []


def geo_fetch_summaries(uids: list[str]) -> list[dict]:
    """Fetch GEO dataset summaries via esummary (gds db)."""
    if not uids:
        return []
    try:
        r = _get(f"{NCBI_BASE}/esummary.fcgi",
                 {"db": "gds", "id": ",".join(uids), "retmode": "json"}, timeout=30)
        result = r.json().get("result", {})
        records = []
        for uid in uids:
            item = result.get(uid)
            if not item or uid == "uids":
                continue
            # accession can be GSE or GDS
            accession = item.get("accession", f"GDS{uid}")
            records.append({
                "uid":         uid,
                "accession":   accession,
                "title":       item.get("title", ""),
                "summary":     item.get("summary", "")[:800],
                "gdstype":     item.get("gdstype", ""),
                "taxon":       item.get("taxon", ""),
                "n_samples":   item.get("n_samples", ""),
                "platform":    item.get("platform_organism", ""),
                "pubmed_ids":  item.get("pubmed_ids", []),
                "url":         f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
                "ftp":         item.get("ftplink", ""),
            })
        return records
    except Exception:
        return []


def is_kidney_relevant_geo(record: dict) -> bool:
    text = (record.get("title", "") + " " + record.get("summary", "")).lower()
    return bool(re.search(r"\bkidney\b|\brenal\b|\bnephro\b|\bAKI\b|acute kidney|proximal tubule", text, re.I))


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_pubmed_record(pmid: str, meta: dict, abstract: str,
                       modalities: list[str], ml_score: float,
                       query: str) -> Path:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    out = DATASETS_DIR / f"PMID_{pmid}.json"
    record = {
        "source":       "pubmed",
        "pmid":         pmid,
        "title":        meta.get("title", ""),
        "authors":      meta.get("authors", []),
        "journal":      meta.get("journal", ""),
        "year":         meta.get("year", ""),
        "doi":          meta.get("doi", ""),
        "pmcid":        meta.get("pmcid", ""),
        "url":          meta.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
        "abstract":     abstract[:1200],
        "query":        query,
        "modalities":   modalities,
        "ml_score":     ml_score,
        "sample_size":  extract_sample_size(abstract),
        "ml_recommendations": [MODALITY_ML_MAP.get(m, {}) for m in modalities
                                if m in MODALITY_ML_MAP],
        "fetched_at":   datetime.now().isoformat(),
    }
    with open(out, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return out


def save_geo_record(record: dict, modalities: list[str], ml_score: float) -> Path:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    acc = record["accession"]
    out = DATASETS_DIR / f"GEO_{acc}.json"
    enriched = {
        "source":            "geo",
        **record,
        "modalities":        modalities,
        "ml_score":          ml_score,
        "ml_recommendations": [MODALITY_ML_MAP.get(m, {}) for m in modalities
                                if m in MODALITY_ML_MAP],
        "fetched_at":        datetime.now().isoformat(),
    }
    with open(out, "w") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    return out


# ── Aggregate recommendation ──────────────────────────────────────────────────

def build_ml_recommendation(all_records: list[dict]) -> dict:
    """
    Aggregate across all retrieved records to produce a ranked ML strategy
    recommendation for the orchestrator / plan agent.
    """
    modality_counts: dict[str, int] = {}
    modality_scores: dict[str, list[float]] = {}

    for rec in all_records:
        for mod in rec.get("modalities", []):
            modality_counts[mod] = modality_counts.get(mod, 0) + 1
            modality_scores.setdefault(mod, []).append(rec.get("ml_score", 0.0))

    # Rank modalities by (count × mean_score)
    ranked = sorted(
        [(m, modality_counts[m],
          round(sum(modality_scores[m]) / len(modality_scores[m]), 3))
         for m in modality_counts],
        key=lambda x: x[1] * x[2],
        reverse=True,
    )

    # Collect best model families
    top_models: list[str] = []
    top_feature_strategies: list[str] = []
    for mod, _, _ in ranked[:3]:
        ml = MODALITY_ML_MAP.get(mod, {})
        top_models.extend(ml.get("preferred_models", []))
        top_feature_strategies.extend(ml.get("feature_strategies", []))

    # De-duplicate preserving order
    top_models = list(dict.fromkeys(top_models))[:6]
    top_feature_strategies = list(dict.fromkeys(top_feature_strategies))[:6]

    # High-scoring individual records
    top_records = sorted(all_records, key=lambda r: r.get("ml_score", 0), reverse=True)[:5]

    return {
        "top_modalities":         [(m, cnt, sc) for m, cnt, sc in ranked[:5]],
        "dominant_modality":      ranked[0][0] if ranked else "scRNA-seq",
        "recommended_models":     top_models,
        "recommended_features":   top_feature_strategies,
        "top_datasets": [
            {
                "id":        r.get("pmid") or r.get("accession", ""),
                "source":    r.get("source", ""),
                "title":     r.get("title", "")[:80],
                "modality":  r.get("modalities", []),
                "ml_score":  r.get("ml_score", 0),
                "url":       r.get("url", ""),
            }
            for r in top_records
        ],
        "narrative": (
            f"Found {len(all_records)} kidney omics records. "
            f"Dominant modality: {ranked[0][0] if ranked else 'scRNA-seq'} "
            f"({ranked[0][1] if ranked else 0} papers/datasets). "
            f"Top ML approaches: {', '.join(top_models[:3])}. "
            f"Recommended feature strategies: {', '.join(top_feature_strategies[:3])}."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(verbose: bool = True) -> dict:
    RESULTS.mkdir(exist_ok=True)
    DATASETS_DIR.mkdir(exist_ok=True)

    # Wipe stale files
    for stale in DATASETS_DIR.glob("*.json"):
        stale.unlink()

    log: list[str] = []

    def _log(msg: str):
        if verbose:
            print(msg)
        log.append(msg)

    _log("=" * 64)
    _log("  [OMICS DISCOVERY] Kidney Omics Literature & Dataset Search")
    _log(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log(f"  Output  : {DATASETS_DIR}")
    _log("=" * 64)

    all_records: list[dict] = []
    seen_titles: set[str]   = set()

    # ── Phase 1: PubMed literature search ────────────────────────────────────
    _log("\n  ── Phase 1: PubMed Omics Literature ──")
    for q in OMICS_QUERIES:
        _log(f"\n  Query: \"{q}\"")
        try:
            pmids = pubmed_search(q, retmax=8)
            time.sleep(0.35)
        except Exception as e:
            _log(f"  Search error: {e}")
            continue

        if not pmids:
            _log("  No results.")
            continue

        try:
            abstracts_text = pubmed_fetch_abstracts(pmids)
            time.sleep(0.35)
            meta_map = pubmed_fetch_metadata(pmids)
            time.sleep(0.35)
        except Exception as e:
            _log(f"  Fetch error: {e}")
            continue

        for pmid in pmids:
            pm    = meta_map.get(pmid, {})
            title = pm.get("title", "")
            nt    = _norm_title(title)
            if nt and nt in seen_titles:
                _log(f"    Skip duplicate PMID {pmid}: {title[:55]}")
                continue

            # Extract this PMID's abstract block
            blocks = re.split(r"\n{2,}", abstracts_text.strip())
            abstract = ""
            for blk in blocks:
                if pmid in blk or (title and title[:30].lower() in blk.lower()):
                    abstract = blk
                    break
            if not abstract:
                abstract = abstracts_text[:800]

            # Kidney relevance guard
            if not re.search(
                r"\bkidney\b|\brenal\b|\bnephro\b|\bAKI\b|acute kidney|proximal tubule|glomerul",
                (title + " " + abstract), re.IGNORECASE
            ):
                _log(f"    Skip non-kidney PMID {pmid}: {title[:55]}")
                continue

            seen_titles.add(nt)
            modalities = classify_modality(title, abstract)
            ml_score   = score_ml_suitability(abstract, modalities)

            try:
                saved = save_pubmed_record(pmid, pm, abstract, modalities, ml_score, q)
                all_records.append({
                    "source":    "pubmed",
                    "pmid":      pmid,
                    "title":     title,
                    "modalities": modalities,
                    "ml_score":  ml_score,
                    "url":       pm.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                })
                _log(f"    PMID {pmid} | {modalities} | score={ml_score:.2f} | {title[:50]}")
            except Exception as e:
                _log(f"    Save error PMID {pmid}: {e}")

    _log(f"\n  PubMed phase: {len(all_records)} kidney-relevant records collected.")

    # ── Phase 2: GEO dataset search ───────────────────────────────────────────
    _log("\n  ── Phase 2: NCBI GEO Dataset Search ──")
    geo_records: list[dict] = []
    seen_geo: set[str] = set()

    for gq in GEO_QUERIES:
        _log(f"\n  GEO query: \"{gq}\"")
        try:
            uids = geo_search(gq, retmax=6)
            time.sleep(0.35)
        except Exception as e:
            _log(f"  GEO search error: {e}")
            continue

        if not uids:
            _log("  No GEO results.")
            continue

        try:
            summaries = geo_fetch_summaries(uids)
            time.sleep(0.35)
        except Exception as e:
            _log(f"  GEO fetch error: {e}")
            continue

        for rec in summaries:
            acc = rec.get("accession", "")
            if acc in seen_geo:
                _log(f"    Skip duplicate GEO {acc}")
                continue
            if not is_kidney_relevant_geo(rec):
                _log(f"    Skip non-kidney GEO {acc}: {rec.get('title','')[:50]}")
                continue

            seen_geo.add(acc)
            combined_text = rec.get("title", "") + " " + rec.get("summary", "")
            modalities = classify_modality(combined_text, "")
            ml_score   = score_ml_suitability(rec.get("summary", ""), modalities)

            try:
                saved = save_geo_record(rec, modalities, ml_score)
                geo_record = {
                    "source":     "geo",
                    "accession":  acc,
                    "title":      rec.get("title", ""),
                    "modalities": modalities,
                    "ml_score":   ml_score,
                    "n_samples":  rec.get("n_samples", ""),
                    "taxon":      rec.get("taxon", ""),
                    "url":        rec.get("url", ""),
                }
                all_records.append(geo_record)
                geo_records.append(geo_record)
                _log(f"    GEO {acc} | {modalities} | n={rec.get('n_samples','')} | score={ml_score:.2f} | {rec.get('title','')[:45]}")
            except Exception as e:
                _log(f"    GEO save error {acc}: {e}")

    _log(f"\n  GEO phase: {len(geo_records)} kidney-relevant datasets collected.")

    # ── Phase 3: Aggregate ML recommendation ─────────────────────────────────
    _log("\n  ── Phase 3: ML Recommendation Synthesis ──")
    recommendation = build_ml_recommendation(all_records)

    _log(f"  Dominant modality : {recommendation['dominant_modality']}")
    _log(f"  Top modalities    : {recommendation['top_modalities']}")
    _log(f"  Recommended models: {recommendation['recommended_models']}")
    _log(f"  Feature strategies: {recommendation['recommended_features']}")
    _log(f"  Top datasets (by ML score):")
    for ds in recommendation["top_datasets"]:
        _log(f"    [{ds['source']}] {ds['id']} | {ds['modality']} | score={ds['ml_score']:.2f} | {ds['title'][:50]}")

    # ── Save master summary ───────────────────────────────────────────────────
    summary = {
        "search_timestamp":      datetime.now().isoformat(),
        "pubmed_queries":        OMICS_QUERIES,
        "geo_queries":           GEO_QUERIES,
        "total_records":         len(all_records),
        "pubmed_records":        len([r for r in all_records if r.get("source") == "pubmed"]),
        "geo_records":           len(geo_records),
        "datasets_dir":          str(DATASETS_DIR),
        "modality_ml_map":       {k: {"preferred_models": v["preferred_models"],
                                      "feature_strategies": v["feature_strategies"]}
                                  for k, v in MODALITY_ML_MAP.items()},
        "recommendation":        recommendation,
        "all_records":           all_records,
    }

    out = RESULTS / "omics_discovery.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log_out = RESULTS / "omics_discovery_log.txt"
    with open(log_out, "w") as f:
        f.write("\n".join(log))

    _log(f"\n  Total records      : {len(all_records)}")
    _log(f"    PubMed papers    : {summary['pubmed_records']}")
    _log(f"    GEO datasets     : {len(geo_records)}")
    _log(f"  Narrative          : {recommendation['narrative']}")
    _log(f"\n  Saved omics_discovery.json → {out}")
    _log(f"  Saved omics_discovery_log   → {log_out}")
    _log("=" * 64)

    return summary


if __name__ == "__main__":
    result = main(verbose=True)
    print(f"\nDone. {result['total_records']} kidney omics records found.")
    print(f"Dominant modality: {result['recommendation']['dominant_modality']}")
    print(f"Top models: {result['recommendation']['recommended_models'][:4]}")

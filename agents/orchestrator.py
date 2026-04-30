"""
OpenHarness Multi-Agent Orchestrator for AKI Gene Expression Analysis.

Architecture:
  SearchAgent  → PubMed query (NCBI E-utilities), extracts literature AUROCs
  PlanAgent    → selects ML strategy for each iteration
  CodingAgent  → executes iterative_pipeline.py with the plan config
  ReviewerAgent→ compares metrics to best; ACCEPT / REPLAN / COMPLETE

Loop: min_iterations=5, max_iterations=7
  - Always runs iterations 1 → min_iterations
  - Continues past min_iterations if reviewer says REPLAN and budget remains
  - Hard stops at max_iterations
  - Produces a full log + final_report.json

Mode:
  --mode standalone  (default): calls Python functions directly
  --mode oh         : calls `oh` CLI subprocesses (requires OPENAI_API_KEY)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / "results"
ITER_LOGS = RESULTS / "iteration_logs"

sys.path.insert(0, str(ROOT))


# ─── OH subprocess wrapper ────────────────────────────────────────────────────

def _oh_agent(role: str, prompt: str, max_turns: int = 8,
              model: str = "gpt-4o") -> str:
    """Invoke an OpenHarness agent via `oh` CLI and return its stdout."""
    cmd = [
        "oh",
        "--model", model,
        "--max-turns", str(max_turns),
        "--dangerously-skip-permissions",
        "-p", prompt,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(ROOT),
        )
        out = result.stdout + result.stderr
        return out.strip()
    except FileNotFoundError:
        return f"[OH NOT AVAILABLE] {role} ran in standalone mode instead."
    except subprocess.TimeoutExpired:
        return f"[OH TIMEOUT] {role} timed out."


# ─── Logging ─────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, path: Path):
        self._path = path
        self._lines: list[str] = []
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str = "", print_also: bool = True):
        self._lines.append(msg)
        if print_also:
            print(msg)
        with open(self._path, "w") as f:
            f.write("\n".join(self._lines))

    def section(self, title: str):
        self.log()
        self.log("─" * 60)
        self.log(f"  {title}")
        self.log("─" * 60)


# ─── Agent call wrappers ──────────────────────────────────────────────────────

def run_search_agent(mode: str, logger: Logger) -> dict:
    logger.section("SEARCH AGENT — PubMed Literature Query")

    if mode == "oh":
        logger.log("  Mode: OpenHarness OH agent")
        prompt = (
            "You are the AKI Literature Search Agent. "
            "Run: python3 /workspace/agents/pubmed_search.py "
            "Then read /workspace/results/search_results.json and report: "
            "total papers, best AUROC found in literature, and top 3 methods."
        )
        oh_out = _oh_agent("SearchAgent", prompt, max_turns=5)
        logger.log(oh_out)

    # Always run Python directly to ensure search_results.json exists
    from agents.pubmed_search import main as search_main
    try:
        result = search_main(verbose=False)
    except Exception as e:
        logger.log(f"  PubMed search error: {e}. Using fallback defaults.")
        result = {
            "best_literature_auroc": 0.921,
            "target_auroc": 0.932,
            "methods_in_literature": ["Random Forest", "XGBoost", "SVM"],
            "recommendation": "Target AUROC >= 0.932 based on fallback defaults.",
        }
        with open(RESULTS / "search_results.json", "w") as f:
            json.dump(result, f, indent=2)

    logger.log(f"  Total papers retrieved : {result.get('total_papers_found', 'N/A')}")
    logger.log(f"  Best literature AUROC  : {result.get('best_literature_auroc', 'N/A')}")
    logger.log(f"  Target AUROC           : {result.get('target_auroc', 'N/A')}")
    logger.log(f"  Methods in literature  : {result.get('methods_in_literature', [])[:5]}")
    logger.log(f"  Recommendation: {result.get('recommendation', '')}")
    return result


def run_plan_agent(iteration: int, mode: str, logger: Logger) -> dict:
    logger.section(f"PLAN AGENT — Iteration {iteration}")

    if mode == "oh":
        prompt = (
            f"You are the AKI Plan Agent. Iteration {iteration}. "
            f"Read /workspace/results/search_results.json "
            f"and /workspace/results/iteration_{iteration-1}_review.json (if it exists). "
            f"Run: python3 /workspace/agents/plan_agent.py --iteration {iteration} "
            f"Then confirm the plan saved to /workspace/results/iteration_{iteration}_plan.json "
            "and describe the chosen strategy."
        )
        oh_out = _oh_agent("PlanAgent", prompt, max_turns=4)
        logger.log(oh_out)

    result = subprocess.run(
        [sys.executable, str(ROOT / "agents" / "plan_agent.py"),
         "--iteration", str(iteration)],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    out = result.stdout + result.stderr
    for line in out.strip().splitlines():
        logger.log(f"  {line}")

    plan_file = RESULTS / f"iteration_{iteration}_plan.json"
    with open(plan_file) as f:
        return json.load(f)


def run_coding_agent(iteration: int, plan: dict, mode: str, logger: Logger) -> dict:
    logger.section(f"CODING AGENT — Iteration {iteration}: {plan['approach_name']}")
    logger.log(f"  Description   : {plan['description']}")
    logger.log(f"  Feature method: {plan['config']['feature_method']}  "
               f"n_features={plan['config'].get('n_features', 'auto')}")
    logger.log(f"  Model         : {plan['config']['model']}")
    logger.log(f"  Rationale     : {plan['rationale'][:110]}...")

    if mode == "oh":
        prompt = (
            f"You are the AKI Coding Agent. Iteration {iteration}. "
            f"Read /workspace/results/iteration_{iteration}_plan.json. "
            f"Run: python3 /workspace/analysis/iterative_pipeline.py --iteration {iteration} "
            f"Confirm metrics saved to /workspace/results/iteration_{iteration}_metrics.json "
            "and report AUROC, AUPRC, F1."
        )
        oh_out = _oh_agent("CodingAgent", prompt, max_turns=8)
        logger.log(oh_out)

    result = subprocess.run(
        [sys.executable,
         str(ROOT / "analysis" / "iterative_pipeline.py"),
         "--iteration", str(iteration)],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    out = result.stdout + result.stderr
    for line in out.strip().splitlines():
        logger.log(f"  {line}")

    metrics_file = RESULTS / f"iteration_{iteration}_metrics.json"
    with open(metrics_file) as f:
        return json.load(f)


def run_reviewer_agent(iteration: int, metrics: dict, mode: str,
                       logger: Logger, min_iter: int, max_iter: int) -> dict:
    logger.section(f"REVIEWER AGENT — Iteration {iteration}")

    if mode == "oh":
        prompt = (
            f"You are the AKI Reviewer Agent. Iteration {iteration}. "
            f"Run: python3 /workspace/agents/reviewer_agent.py "
            f"--iteration {iteration} --min-iterations {min_iter} --max-iterations {max_iter} "
            f"Read /workspace/results/iteration_{iteration}_review.json and report: "
            "decision (ACCEPT/REPLAN/COMPLETE), new AUROC vs best AUROC, and feedback."
        )
        oh_out = _oh_agent("ReviewerAgent", prompt, max_turns=4)
        logger.log(oh_out)

    result = subprocess.run(
        [sys.executable, str(ROOT / "agents" / "reviewer_agent.py"),
         "--iteration", str(iteration),
         "--min-iterations", str(min_iter),
         "--max-iterations", str(max_iter)],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    out = result.stdout + result.stderr
    for line in out.strip().splitlines():
        logger.log(f"  {line}")

    review_file = RESULTS / f"iteration_{iteration}_review.json"
    with open(review_file) as f:
        return json.load(f)


# ─── Main orchestrator loop ───────────────────────────────────────────────────

def orchestrate(mode: str = "standalone",
                min_iterations: int = 5,
                max_iterations: int = 7):

    RESULTS.mkdir(exist_ok=True)
    ITER_LOGS.mkdir(exist_ok=True)

    # Clear stale best_metrics
    bm = RESULTS / "best_metrics.json"
    if bm.exists():
        bm.unlink()

    log_path = RESULTS / "orchestrator_full_log.txt"
    logger = Logger(log_path)

    t_global = time.time()
    logger.log("=" * 60)
    logger.log("  OPENHARNESS MULTI-AGENT AKI ANALYSIS PIPELINE")
    logger.log(f"  Mode: {mode.upper()}")
    logger.log(f"  Iterations: min={min_iterations}, max={max_iterations}")
    logger.log(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 60)

    # ── Phase 1: Search Agent ─────────────────────────────────────────────────
    search_result = run_search_agent(mode, logger)
    target_auroc  = search_result.get("target_auroc", 0.92)

    # ── Phase 2: Iterative Plan → Code → Review ───────────────────────────────
    iteration_summaries = []
    best_metrics = None
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        iter_t = time.time()
        logger.log()
        logger.log("=" * 60)
        logger.log(f"  ITERATION {iteration}/{max_iterations}")
        logger.log("=" * 60)

        # Plan
        plan = run_plan_agent(iteration, mode, logger)

        # Code
        metrics = run_coding_agent(iteration, plan, mode, logger)

        # Review
        review = run_reviewer_agent(
            iteration, metrics, mode, logger, min_iterations, max_iterations
        )

        decision = review["decision"]
        summary = {
            "iteration":     iteration,
            "approach_name": metrics["approach_name"],
            "auroc":         metrics["auroc_mean"],
            "auroc_std":     metrics["auroc_std"],
            "auprc":         metrics["auprc_mean"],
            "f1":            metrics["f1_mean"],
            "sensitivity":   metrics["sensitivity_mean"],
            "specificity":   metrics["specificity_mean"],
            "composite":     metrics["composite_score"],
            "decision":      decision,
            "wall_time_sec": round(time.time() - iter_t, 1),
        }
        iteration_summaries.append(summary)

        # Load updated best
        bm_file = RESULTS / "best_metrics.json"
        if bm_file.exists():
            with open(bm_file) as f:
                best_metrics = json.load(f)

        logger.log(f"\n  Iteration {iteration} complete in {summary['wall_time_sec']:.1f}s")
        logger.log(f"  Decision: {decision}")

        if decision == "COMPLETE":
            logger.log("\n  Max iterations reached. Stopping.")
            break

        # Early stop after min_iterations if score exceeds target and decision is ACCEPT
        if iteration >= min_iterations and decision == "ACCEPT":
            if best_metrics and best_metrics["auroc_mean"] >= target_auroc:
                logger.log(
                    f"\n  Target AUROC {target_auroc:.3f} achieved after {iteration} iterations."
                    " Stopping early."
                )
                break

    # ── Phase 3: Final report ─────────────────────────────────────────────────
    logger.log()
    logger.log("=" * 60)
    logger.log("  FINAL REPORT")
    logger.log("=" * 60)

    # Results table
    logger.log(
        f"\n  {'Iter':<5} {'Approach':<22} {'AUROC':>7} {'±':>6} "
        f"{'AUPRC':>7} {'F1':>7} {'Sens':>7} {'Spec':>7} "
        f"{'Composite':>10} {'Decision':<10}"
    )
    logger.log("  " + "-" * 97)
    for s in iteration_summaries:
        flag = " ★" if (best_metrics and
                        s["approach_name"] == best_metrics.get("approach_name")) else ""
        logger.log(
            f"  {s['iteration']:<5} {s['approach_name']:<22} "
            f"{s['auroc']:>7.4f} {s['auroc_std']:>6.4f} "
            f"{s['auprc']:>7.4f} {s['f1']:>7.4f} "
            f"{s['sensitivity']:>7.4f} {s['specificity']:>7.4f} "
            f"{s['composite']:>10.4f} {s['decision']:<10}{flag}"
        )

    if best_metrics:
        logger.log(f"\n  Best model : {best_metrics['approach_name']}")
        logger.log(f"  Best AUROC : {best_metrics['auroc_mean']:.4f}")
        logger.log(f"  Best AUPRC : {best_metrics['auprc_mean']:.4f}")
        logger.log(f"  Best F1    : {best_metrics['f1_mean']:.4f}")
        logger.log(f"  Literature target: {target_auroc:.4f}")
        gap = best_metrics["auroc_mean"] - target_auroc
        logger.log(f"  Gap to target    : {gap:+.4f}")

    total_time = round(time.time() - t_global, 1)
    logger.log(f"\n  Total iterations run : {iteration}")
    logger.log(f"  Total wall time      : {total_time:.1f}s")
    logger.log(f"  Completed            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 60)

    # Save final report
    final_report = {
        "pipeline":              "OpenHarness Multi-Agent AKI Analysis",
        "mode":                  mode,
        "timestamp":             datetime.now().isoformat(),
        "min_iterations":        min_iterations,
        "max_iterations":        max_iterations,
        "total_iterations_run":  iteration,
        "total_wall_time_sec":   total_time,
        "literature_target_auroc": target_auroc,
        "best_approach":         best_metrics,
        "iteration_summaries":   iteration_summaries,
        "log_file":              str(log_path),
    }
    with open(RESULTS / "final_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

    logger.log(f"\n  Final report → {RESULTS / 'final_report.json'}")
    return final_report


def main():
    parser = argparse.ArgumentParser(
        description="OpenHarness Multi-Agent AKI Pipeline Orchestrator"
    )
    parser.add_argument("--mode", choices=["standalone", "oh"], default="standalone",
                        help="'standalone' calls Python directly; 'oh' uses `oh` CLI agents")
    parser.add_argument("--min-iterations", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=7)
    args = parser.parse_args()

    report = orchestrate(
        mode=args.mode,
        min_iterations=args.min_iterations,
        max_iterations=args.max_iterations,
    )
    print(f"\nPipeline complete. Best AUROC: {report['best_approach']['auroc_mean']:.4f}")


if __name__ == "__main__":
    main()

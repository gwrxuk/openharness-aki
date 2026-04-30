"""
OpenHarness Coding Agent
Executes bioinformatics and ML analysis scripts for AKI gene study.
"""

import os
import sys
import subprocess
from pathlib import Path


ROLE = os.environ.get("AGENT_ROLE", "coding")

SCRIPTS = {
    "coding": [
        "analysis/01_data_fetch.py",
        "analysis/02_preprocessing.py",
        "analysis/03_aki_signature.py",
        "analysis/04_ml_model.py",
        "analysis/06_visualization.py",
    ]
}


def execute_script(script_path: str) -> bool:
    print(f"[CodingAgent] Executing: {script_path}")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        cwd="/workspace"
    )
    return result.returncode == 0


def main():
    scripts = SCRIPTS.get(ROLE, [])
    for script in scripts:
        if Path(f"/workspace/{script}").exists():
            success = execute_script(script)
            if not success:
                print(f"[CodingAgent] FAILED: {script}")
                sys.exit(1)
    print("[CodingAgent] All coding tasks complete.")


if __name__ == "__main__":
    main()

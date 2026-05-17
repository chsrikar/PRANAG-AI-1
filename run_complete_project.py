"""
run_complete_project.py
Single entrypoint to run complete project flow:
Srikar output (already trained) -> Aryan pipeline -> Divyanshu validation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path):
    print(f"\n$ {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Run complete PRANAG-AI workflow")
    parser.add_argument(
        "--data",
        default=str(Path("Model") / "datasrc" / "real_data_combined.parquet"),
        help="Input parquet for Aryan pipeline",
    )
    parser.add_argument(
        "--models",
        default=str(Path("Model") / "outputs" / "models"),
        help="Srikar model checkpoint directory",
    )
    parser.add_argument("--resume", action="store_true", help="Resume Aryan pipeline from checkpoint")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    pipelines = root / "Pipelines"
    validators = root / "Validators"
    data_path = Path(args.data)
    model_path = Path(args.models)
    if not data_path.is_absolute():
        data_path = (root / data_path).resolve()
    if not model_path.is_absolute():
        model_path = (root / model_path).resolve()

    print("=== COMPLETE PROJECT RUN ===")
    print(f"Data   : {data_path}")
    print(f"Models : {model_path}")

    run_cmd(
        [
            sys.executable,
            "run_pipeline.py",
            "--data",
            str(data_path),
            "--models",
            str(model_path),
            *(["--resume"] if args.resume else []),
        ],
        cwd=pipelines,
    )

    handoff = pipelines / "results" / "handoff_for_divyanshu.csv"
    if not handoff.exists():
        raise FileNotFoundError(f"Handoff not found: {handoff}")

    run_cmd(
        [
            sys.executable,
            "run_validation.py",
            "--input",
            str(handoff),
        ],
        cwd=validators,
    )

    print("\n=== PROJECT COMPLETE ===")
    print(f"Handoff file      : {handoff}")
    print(f"Validation report : {validators / 'results' / 'validation_report.json'}")
    print(f"Dashboard         : {validators / 'results' / 'validation_dashboard.html'}")
    print(f"FP sweep results  : {validators / 'results' / 'fp_sweep_results.json'}")
    print(f"Feedback report   : {validators / 'results' / 'feedback_report.json'}")


if __name__ == "__main__":
    main()

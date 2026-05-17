"""
config.py  —  ARYAN  (Task 7)
Central configuration for the entire simulation pipeline.
Edit this file to plug in real paths and settings.
"""

import os
from pathlib import Path

# ── DATA PATHS ────────────────────────────────────────────────────────────────
# PLUG IN YOUR REAL PATHS HERE when data is available

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = str(ROOT_DIR / "Model" / "datasrc")
RESULTS_DIR = str(Path(__file__).resolve().parent / "results")
CHECKPOINT_DIR = str(Path(__file__).resolve().parent / "checkpoints")

# ── SRIKAR'S MODEL PATHS ──────────────────────────────────────────────────────
# Set these when Srikar hands over his trained models

SRIKAR_MODEL_DIR = os.getenv(
    "SRIKAR_MODEL_DIR",
    str(ROOT_DIR / "Model" / "outputs" / "models"),
)
# Example: SRIKAR_MODEL_DIR = "models/srikar/"

# Individual model files (inside SRIKAR_MODEL_DIR)
MODEL_FILES = {
    "heat":     "heat_pinn.pt",
    "stress":   "stress_pinn.pt",
    "growth":   "growth_pinn.pt",
    "biology":  "biology_pinn.pt",
    "chemistry":"chemistry_pinn.pt",
    "surrogate":"surrogate_model.pt",
}

# ── DATA FILES ────────────────────────────────────────────────────────────────
# From data curators team

PARQUET_FILES = {
    "main":       os.path.join(DATA_DIR, "real_data_combined.parquet"),
    "materials":  os.path.join(DATA_DIR, "aflow_materials.parquet"),
    "molecules":  os.path.join(DATA_DIR, "chembl_compounds.parquet"),
    "proteins":   os.path.join(DATA_DIR, "pdb_proteins.parquet"),
    "soil":       os.path.join(DATA_DIR, "openlandmap_soil_india.parquet"),
    "index":      os.path.join(DATA_DIR, "universal_index_final.parquet"),
}

# ── PROMPT PARSER JSON FILES ──────────────────────────────────────────────────
# From prompt parser team

PROMPT_JSON_FILES = {
    "query_1": str(ROOT_DIR / "Model" / "datasrc" / "out1.json"),   # sugarcane UP 37°C
    "query_2": str(ROOT_DIR / "Model" / "datasrc" / "out2.json"),   # wheat Delhi 45°C
}

# ── SIMULATION SETTINGS ───────────────────────────────────────────────────────

BATCH_SIZE        = 50000      # traits per batch (vectorized — 10x faster amortisation)
SCORE_THRESHOLD   = 0.70       # below this → filtered before validation
TARGET_SPEED_HRS  = 4.0        # 1M traits in 4 hours
CHECKPOINT_EVERY  = 10         # save checkpoint every N batches

# ── VALIDATION THRESHOLDS (for Divyanshu) ────────────────────────────────────

VALIDATION = {
    "pass_threshold":   0.60,  # Adjusted to reflect improved scoring
    "biology_min":      0.65,
    "materials_min":    0.68,
    "physics_min":      0.70,
    "chemistry_min":    0.72,
    "accuracy_target":  0.95,
    "fp_rate_max":      0.05,
    "fn_rate_max":      0.10,
}

# ── OUTPUT FILES ──────────────────────────────────────────────────────────────

OUTPUT_FILES = {
    "simulation_results":    os.path.join(RESULTS_DIR, "simulation_results.parquet"),
    "multi_domain_results":  os.path.join(RESULTS_DIR, "multi_domain_results.parquet"),
    "performance_log":       os.path.join(RESULTS_DIR, "performance_log.json"),
    "performance_report":    os.path.join(RESULTS_DIR, "performance_report.txt"),
    "validation_report":     str(ROOT_DIR / "Validators" / "results" / "validation_report.json"),
    "validation_log_csv":    str(ROOT_DIR / "Validators" / "results" / "validation_log.csv"),
}


def print_config():
    print("\n── PIPELINE CONFIG ───────────────────────────────────")
    print(f"  Srikar models   : {SRIKAR_MODEL_DIR}")
    print(f"  Main parquet    : {PARQUET_FILES['main']}")
    print(f"  Batch size      : {BATCH_SIZE:,}")
    print(f"  Score threshold : {SCORE_THRESHOLD}")
    print(f"  Speed target    : 1M traits in {TARGET_SPEED_HRS}hrs")
    print(f"  Results dir     : {RESULTS_DIR}/")
    print("─" * 52)


if __name__ == "__main__":
    print_config()

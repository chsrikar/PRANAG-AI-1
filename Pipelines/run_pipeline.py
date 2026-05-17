"""
run_pipeline.py  —  ARYAN  (Master Runner)
Run this file to execute the full simulation pipeline end-to-end.

USAGE:
  python run_pipeline.py                        # uses mock data
  python run_pipeline.py --data real_data.parquet --models models/srikar/
  python run_pipeline.py --resume               # resume from checkpoint
"""

import argparse
import os
import time
from config import print_config, PARQUET_FILES, SRIKAR_MODEL_DIR, BATCH_SIZE
from data_loader import DataLoader
from batch_simulator import BatchSimulator
from multi_domain_simulator import MultiDomainSimulator, get_results_for_divyanshu
from performance_monitor import PerformanceMonitor


def run(parquet_path=None, model_dir=None, resume=True):
    print("\n" + "="*55)
    print("  ARYAN — FULL SIMULATION PIPELINE")
    print("="*55)
    print_config()

    monitor = PerformanceMonitor()
    monitor.start()
    t_total = time.perf_counter()

    # ── STEP 1: Load data ─────────────────────────────────────────
    print("\n[STEP 1] Loading data...")
    path = parquet_path or PARQUET_FILES.get("main")
    loader = DataLoader(path)
    print(f"  Total traits: {loader.count():,}")
    loader.speed_test(min(loader.count(), 10000))

    # ── STEP 2: Batch simulation (all traits → viability scores) ──
    print("\n[STEP 2] Running batch simulation...")
    sim = BatchSimulator(
        model_dir    = model_dir or SRIKAR_MODEL_DIR,
        parquet_path = path,
        batch_size   = BATCH_SIZE,
    )
    results = sim.run_all(resume=resume)

    # ── STEP 3: Multi-domain scoring ──────────────────────────────
    print("\n[STEP 3] Running multi-domain simulation...")
    mds = MultiDomainSimulator(
        model_dir    = model_dir or SRIKAR_MODEL_DIR,
        parquet_path = path,
    )
    domain_results = mds.run(batch_size=BATCH_SIZE)

    # ── STEP 4: Prepare handoff for Divyanshu ─────────────────────
    print("\n[STEP 4] Preparing results for Divyanshu's validator...")
    handoff = get_results_for_divyanshu(domain_results)

    # Save handoff CSV for easy sharing
    handoff_path = "results/handoff_for_divyanshu.csv"
    handoff.to_csv(handoff_path, index=False)
    print(f"  ✅ Handoff file saved: {handoff_path}")
    print(f"  ✅ Designs ready for validation: {len(handoff):,}")

    # ── STEP 5: Performance report ────────────────────────────────
    print("\n[STEP 5] Generating performance report...")
    total_elapsed = time.perf_counter() - t_total
    monitor.record(1, len(domain_results), total_elapsed)
    monitor.report()

    print(f"\n{'='*55}")
    print(f"  ✅ PIPELINE COMPLETE")
    print(f"  Total time : {total_elapsed:.1f}s")
    print(f"  Next step  : Hand results to Divyanshu")
    print(f"  File       : {handoff_path}")
    print(f"{'='*55}")

    return handoff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aryan Simulation Pipeline")
    parser.add_argument("--data",    type=str, default=None, help="Path to parquet file")
    parser.add_argument("--models",  type=str, default=None, help="Path to Srikar's model dir")
    parser.add_argument("--resume",  action="store_true",    help="Resume from checkpoint")
    args = parser.parse_args()

    run(
        parquet_path = args.data,
        model_dir    = args.models,
        resume       = args.resume,
    )

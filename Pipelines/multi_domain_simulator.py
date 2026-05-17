"""
multi_domain_simulator.py  —  ARYAN  (Task 5)
Runs traits through ALL 5 PINNs and outputs domain-level scores.
Output per trait: biology_score, physics_score, material_score,
                  chemistry_score, overall_score.
"""

import pandas as pd
import numpy as np
import time
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from data_loader import DataLoader
from batch_simulator import SrikarModelInterface


MULTI_DOMAIN_OUTPUT = "results/multi_domain_results.parquet"


@dataclass
class MultiDomainResult:
    trait_id:         str
    entity_type:      str
    biology_score:    float
    physics_score:    float
    material_score:   float
    chemistry_score:  float
    growth_score:     float
    overall_score:    float
    passed:           bool
    source:           str = ""
    location:         str = ""
    temperature_max:  float = 0.0
    simulated_at:     str  = ""

    def __post_init__(self):
        if not self.simulated_at:
            self.simulated_at = datetime.now().isoformat()


class MultiDomainSimulator:
    """
    Runs all 5 PINN domains on every trait.
    Produces the full score profile needed by Divyanshu's validator.
    """
    PASS_THRESHOLD = 0.60  # Adjusted to reflect improved scoring

    def __init__(self, model_dir: str = None, parquet_path: str = None):
        self.model  = SrikarModelInterface(model_dir)
        self.loader = DataLoader(parquet_path)
        os.makedirs("results", exist_ok=True)

    def simulate_row(self, row: dict) -> MultiDomainResult:
        scores = self.model.predict_all(row)
        return MultiDomainResult(
            trait_id        = str(row.get("trait_id",  f"T_{id(row)}")),
            entity_type     = str(row.get("entity_type", "")),
            biology_score   = scores["biology_score"],
            physics_score   = scores["physics_score"],
            material_score  = scores["material_score"],
            chemistry_score = scores["chemistry_score"],
            growth_score    = scores["growth_score"],
            overall_score   = scores["viability_score"],
            passed          = scores["viability_score"] >= self.PASS_THRESHOLD,
            source          = str(row.get("source",   "")),
            location        = str(row.get("location", "")),
            temperature_max = float(row.get("temperature_max", 0.0)),
        )

    def run(self, batch_size: int = 5000) -> pd.DataFrame:
        """Run multi-domain simulation and save output parquet."""
        total = self.loader.count()
        print(f"\n🔬 Multi-Domain Simulation")
        print(f"   Traits  : {total:,}")
        print(f"   Domains : Heat · Stress · Growth · Biology · Chemistry")
        print(f"{'─'*52}")

        all_results = []
        t_start     = time.perf_counter()

        for batch_num, df in self.loader.get_batches(batch_size):
            batch = [self.simulate_row(r) for r in df.to_dict("records")]
            all_results.extend(batch)
            passed = sum(1 for r in batch if r.passed)
            print(f"  Batch {batch_num:04d} | {len(df):,} traits | "
                  f"✅ {passed} passed | ❌ {len(df)-passed} failed")

        elapsed = time.perf_counter() - t_start
        results_df = pd.DataFrame([asdict(r) for r in all_results])
        results_df.to_parquet(MULTI_DOMAIN_OUTPUT, index=False)

        passed_total = results_df["passed"].sum()
        print(f"\n{'='*52}")
        print(f"  Done in {elapsed:.1f}s")
        print(f"  Passed : {passed_total:,}/{len(results_df):,} "
              f"({passed_total/len(results_df)*100:.1f}%)")
        print(f"  Saved  : {MULTI_DOMAIN_OUTPUT}")

        # Domain averages
        print(f"\n── Domain Averages ──────────────────────────────")
        for col in ["biology_score","physics_score","material_score",
                    "chemistry_score","overall_score"]:
            avg = results_df[col].mean()
            bar = "█" * int(avg * 20)
            print(f"  {col:<20}: {bar:<20} {avg:.3f}")

        return results_df


# ── HANDOFF TO DIVYANSHU ──────────────────────────────────────────────────────
def get_results_for_divyanshu(results_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Returns the simulation results in the exact format
    Divyanshu's validator.py expects.

    If results_df not passed, loads from saved parquet.
    """
    if results_df is None:
        if os.path.exists(MULTI_DOMAIN_OUTPUT):
            results_df = pd.read_parquet(MULTI_DOMAIN_OUTPUT)
        else:
            raise FileNotFoundError(f"Run multi_domain_simulator first.")

    # Pass all designs to validator for comprehensive evaluation
    filtered = results_df.copy()
    if filtered.empty:
        raise ValueError("No results to hand off to validator.")
    
    filtered = filtered.rename(columns={
        "overall_score":  "score",
        "material_score": "materials_score",
    })

    print(f"✅ Ready for Divyanshu: {len(filtered):,} designs ready for validation")
    return filtered


if __name__ == "__main__":
    sim = MultiDomainSimulator()
    results = sim.run(batch_size=5000)

    handoff = get_results_for_divyanshu(results)
    print(f"\nHandoff sample (first 3 for Divyanshu):")
    print(handoff[["trait_id","score","biology_score",
                   "physics_score","materials_score",
                   "chemistry_score"]].head(3).to_string(index=False))

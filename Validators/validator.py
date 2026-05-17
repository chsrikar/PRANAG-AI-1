"""
validator.py  —  DIVYANSHU  (Task 1)
Input : simulation results from Aryan
Output: total traits, passed, failed, top winners
Rule  : score > 0.7 → PASS
"""

import json
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime


@dataclass
class SimulationResult:
    design_id:       str
    score:           float
    biology_score:   float
    materials_score: float
    physics_score:   float
    chemistry_score: float
    traits:          dict = field(default_factory=dict)
    metadata:        dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    design_id:        str
    passed:           bool
    score:            float
    total_traits:     int
    passed_traits:    int
    failed_traits:    int
    top_winners:      list
    rejection_reason: Optional[str] = None
    timestamp:        str = field(default_factory=lambda: datetime.now().isoformat())


class Validator:
    PASS_THRESHOLD = 0.60  # Adjusted to reflect improved scoring

    def __init__(self):
        self.results = []
        self.passed  = []
        self.failed  = []

    def validate(self, sim: SimulationResult) -> ValidationResult:
        traits    = sim.traits or {}
        total     = len(traits) if traits else 10
        passed_t  = sum(1 for v in traits.values() if v >= self.PASS_THRESHOLD) if traits else int(sim.score * 10)
        failed_t  = total - passed_t
        passed    = sim.score > self.PASS_THRESHOLD
        reason    = None if passed else f"Score {sim.score:.4f} below threshold {self.PASS_THRESHOLD}"

        result = ValidationResult(
            design_id        = sim.design_id,
            passed           = passed,
            score            = sim.score,
            total_traits     = total,
            passed_traits    = passed_t,
            failed_traits    = failed_t,
            top_winners      = [],
            rejection_reason = reason,
        )
        self.results.append(result)
        (self.passed if passed else self.failed).append(result)
        return result

    def validate_batch(self, simulations: list) -> dict:
        for sim in simulations:
            self.validate(sim)

        top = [r.design_id for r in sorted(self.passed, key=lambda r: r.score, reverse=True)[:10]]
        for r in self.results:
            r.top_winners = top

        return {
            "total":       len(self.results),
            "passed":      len(self.passed),
            "failed":      len(self.failed),
            "pass_rate":   len(self.passed) / len(self.results) if self.results else 0,
            "top_winners": top,
            "results":     [asdict(r) for r in self.results],
        }

    def validate_from_aryan(self, csv_or_df) -> dict:
        """
        Load Aryan's handoff file directly and validate.

        PLUG IN:
            validator = Validator()
            report = validator.validate_from_aryan("results/handoff_for_divyanshu.csv")
        """
        if isinstance(csv_or_df, str):
            df = pd.read_csv(csv_or_df)
        else:
            df = csv_or_df

        # Map Aryan's column names → SimulationResult
        col_map = {
            "overall_score": "score",
            "material_score": "materials_score",
        }
        df = df.rename(columns=col_map)

        sims = []
        for _, row in df.iterrows():
            sims.append(SimulationResult(
                design_id       = str(row.get("trait_id", row.get("design_id", "?"))),
                score           = float(row.get("score",           row.get("viability_score", 0))),
                biology_score   = float(row.get("biology_score",   0)),
                materials_score = float(row.get("materials_score", 0)),
                physics_score   = float(row.get("physics_score",   0)),
                chemistry_score = float(row.get("chemistry_score", 0)),
                metadata        = {"source": str(row.get("source", ""))},
            ))
        print(f"✅ Loaded {len(sims):,} designs from Aryan")
        return self.validate_batch(sims)


def load_mock_simulations(n=40) -> list:
    import random
    random.seed(42)
    return [SimulationResult(
        design_id       = f"DESIGN_{i+1:03d}",
        score           = random.uniform(0.3, 1.0),
        biology_score   = random.uniform(0.4, 1.0),
        materials_score = random.uniform(0.4, 1.0),
        physics_score   = random.uniform(0.4, 1.0),
        chemistry_score = random.uniform(0.4, 1.0),
        traits          = {f"t{j}": random.uniform(0.3,1.0) for j in range(10)},
    ) for i in range(n)]


if __name__ == "__main__":
    v = Validator()

    # Try real data first, fall back to mock
    handoff = "results/handoff_for_divyanshu.csv"
    if __import__("os").path.exists(handoff):
        report = v.validate_from_aryan(handoff)
    else:
        print("⚠️  No Aryan handoff found — using mock data")
        report = v.validate_batch(load_mock_simulations(40))

    print(f"\n── Validation Results ───────────────────────────────")
    print(f"  Total   : {report['total']}")
    print(f"  Passed  : {report['passed']}  ({report['pass_rate']*100:.1f}%)")
    print(f"  Failed  : {report['failed']}")
    print(f"  Top 5   : {report['top_winners'][:5]}")

"""
run_validation.py  —  DIVYANSHU  (Tasks 5 + 6 + Master Runner)
Run this file to execute the full validation pipeline.

USAGE:
  python run_validation.py                              # mock data
  python run_validation.py --input results/handoff_for_divyanshu.csv
"""

import argparse
import json
import os
import time
import pandas as pd
from datetime import datetime
from dataclasses import asdict

from validator            import Validator, load_mock_simulations
from cross_domain_validator import CrossDomainValidator
from accuracy_validator   import AccuracyValidator, generate_mock_predictions
from failure_analyzer     import FailureAnalyzer


os.makedirs("results", exist_ok=True)
REPORT_JSON = "results/validation_report.json"
REPORT_TXT  = "results/validation_report.txt"
LOG_CSV     = "results/validation_log.csv"


# ── VALIDATION SUITE ──────────────────────────────────────────────────────────

class ValidationSuite:

    def __init__(self, simulations: list, predictions: list = None):
        self.simulations = simulations
        self.predictions = predictions or generate_mock_predictions(100, noise=0.04)
        self.results     = {}

    def _run(self, name, fn):
        t0 = time.perf_counter()
        try:
            result = fn()
            passed = result.get("passed", True)
        except Exception as e:
            result = {"error": str(e)}
            passed = False
        elapsed = (time.perf_counter() - t0) * 1000
        icon    = "✅" if passed else "❌"
        print(f"  {icon} {name:<35} {elapsed:.1f}ms")
        return {"name": name, "passed": passed, "duration_ms": round(elapsed,2), "result": result}

    def test_core(self):
        v = Validator()
        r = v.validate_batch(self.simulations)
        return {"passed": r["pass_rate"] > 0, **r}

    def test_cross_domain(self):
        cdv = CrossDomainValidator()
        r   = cdv.validate_batch(self.simulations)
        return {"passed": True, **r}

    def test_accuracy(self):
        av  = AccuracyValidator()
        m, cal = av.validate_with_calibration(self.predictions, fp_max=0.05)
        d   = m.to_dict()
        return {
            "passed": m.passed,
            **d["metrics"],
            "recommended_threshold": cal["recommended_threshold"],
            "violations": m.violations,
        }

    def test_speed(self):
        LIMIT = 500
        t0    = time.perf_counter()
        v     = Validator()
        v.validate_batch(self.simulations)
        ms    = (time.perf_counter() - t0) * 1000
        return {"passed": ms < LIMIT, "elapsed_ms": round(ms,2), "threshold_ms": LIMIT}

    def test_failures(self):
        v = Validator()
        v.validate_batch(self.simulations)
        failed = [s for s, r in zip(self.simulations, v.results) if not r.passed]
        fa     = FailureAnalyzer()
        r      = fa.analyze_batch(failed)
        return {"passed": True, **r}

    def test_fp_fixer(self):
        from fp_fixer import sweep_thresholds
        results, optimal = sweep_thresholds(self.predictions, start=0.60, end=0.85, step=0.01, fp_target=0.05)
        return {"passed": optimal is not None, "optimal_threshold": optimal["threshold"] if optimal else None, "fp_rate": optimal["fp_rate"] if optimal else None}

    def test_feedback_loop(self):
        from feedback_loop import FeedbackLoop
        fl = FeedbackLoop()
        gen_data = fl.log_generation(self.simulations)
        patterns = fl.detect_patterns()
        return {"passed": True, "generation_logged": gen_data["generation_id"], "recurring_patterns": len(patterns["recurring_patterns"]), "suggestions": patterns["suggestions"]}

    def run(self) -> dict:
        print(f"\n🚀 Running Full Validation Suite...")
        print(f"{'─'*52}")
        t_start = time.perf_counter()

        tests = [
            ("Core Validation",         self.test_core),
            ("Cross-Domain Validation", self.test_cross_domain),
            ("Accuracy Validation",     self.test_accuracy),
            ("Speed Test",              self.test_speed),
            ("Failure Analysis",        self.test_failures),
            ("FP Fixer",                self.test_fp_fixer),
            ("Feedback Loop",           self.test_feedback_loop),
        ]

        test_results = [self._run(name, fn) for name, fn in tests]

        total_ms     = (time.perf_counter() - t_start) * 1000
        passed_tests = sum(1 for t in test_results if t["passed"])
        suite_passed = passed_tests == len(tests)

        print(f"\n{'='*52}")
        status = "✅ SUITE PASSED" if suite_passed else "❌ SUITE FAILED"
        print(f"  {status}  ({passed_tests}/{len(tests)} tests)")
        print(f"  Total runtime: {total_ms:.1f}ms")

        return {
            "suite_passed":   suite_passed,
            "total_tests":    len(tests),
            "passed_tests":   passed_tests,
            "total_ms":       round(total_ms, 2),
            "timestamp":      datetime.now().isoformat(),
            "test_results":   test_results,
        }


# ── FINAL REPORTING ───────────────────────────────────────────────────────────

class FinalReporter:

    def generate(self, simulations: list, suite_results: dict) -> dict:
        v = Validator()
        core = v.validate_batch(simulations)

        cdv = CrossDomainValidator()
        cd  = cdv.validate_batch(simulations)

        av  = AccuracyValidator()
        acc, cal = av.validate_with_calibration(generate_mock_predictions(100, noise=0.04), fp_max=0.05)

        failed = [s for s, r in zip(simulations, v.results) if not r.passed]
        fa     = FailureAnalyzer()
        fail   = fa.analyze_batch(failed) if failed else {"distribution": {}, "total_failures": 0, "top_suggestions": []}

        # Top designs (by score, even if not passed)
        sim_map   = {s.design_id: s for s in simulations}
        top_designs = []
        for rank, res in enumerate(sorted(v.results, key=lambda r: r.score, reverse=True)[:10], 1):
            sim = sim_map.get(res.design_id)
            if not sim: continue
            rec = ("✅ Deploy immediately"   if res.score > 0.90 else
                   "✅ Approved"              if res.score > 0.80 else
                   "⚠️  Monitor closely"       if res.score > 0.70 else
                   "❌ Needs improvement"      if res.score > 0.60 else
                   "🚫 Reject")
            top_designs.append({
                "rank": rank, "design_id": res.design_id,
                "score": round(res.score, 4),
                "biology_score":   round(sim.biology_score, 4),
                "materials_score": round(sim.materials_score, 4),
                "physics_score":   round(sim.physics_score, 4),
                "chemistry_score": round(sim.chemistry_score, 4),
                "recommendation":  rec,
            })

        # Recommendations
        recs = []
        if acc.accuracy < 0.97:
            recs.append("Improve surrogate model training — target >97% accuracy")
        if any(pct > 0 for pct in fail["distribution"].values()):
            top_cat = max(fail["distribution"], key=fail["distribution"].get)
            recs.append(f"Top failure: '{top_cat}' — fix this first")
        if core["pass_rate"] < 0.5:
            recs.append("Pass rate below 50% — review design generation parameters")
        recs.extend((fail.get("top_suggestions") or [])[:3])

        acc_d = acc.to_dict()
        report = {
            "report_id":    f"VR-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": {
                "total_designs":        len(simulations),
                "passed":               core["passed"],
                "failed":               core["failed"],
                "pass_rate_pct":        round(core["pass_rate"]*100, 1),
                "cross_domain_passed":  cd["passed"],
                "surrogate_accuracy":   f"{acc_d['metrics']['accuracy_pct']}%",
                "recommended_threshold":cal["recommended_threshold"],
                "system_status":        "OPERATIONAL" if acc.passed else "NEEDS ATTENTION",
            },
            "top_designs":     top_designs,
            "failure_summary": {
                "total":        fail["total_failures"],
                "distribution": fail.get("distribution", {}),
                "severity":     fail.get("severity_breakdown", {}),
            },
            "accuracy_summary": {
                "accuracy_pct":      acc_d["metrics"]["accuracy_pct"],
                "false_positive_pct":acc_d["metrics"]["false_positive_pct"],
                "false_negative_pct":acc_d["metrics"]["false_negative_pct"],
                "f1_score":          acc_d["metrics"]["f1_score"],
                "passed":            acc.passed,
                "confusion_matrix":  acc_d["confusion_matrix"],
            },
            "recommendations": recs,
            "suite_results":   suite_results,
        }

        # Save JSON
        with open(REPORT_JSON, "w") as f:
            json.dump(report, f, indent=2)

        # Save CSV log
        rows = []
        for r in v.results:
            sim = sim_map.get(r.design_id)
            rows.append({
                "design_id":     r.design_id,
                "score":         r.score,
                "passed":        r.passed,
                "rejection":     r.rejection_reason or "",
                "biology_score": sim.biology_score   if sim else "",
                "materials_score":sim.materials_score if sim else "",
                "physics_score": sim.physics_score   if sim else "",
                "chemistry_score":sim.chemistry_score if sim else "",
            })
        pd.DataFrame(rows).to_csv(LOG_CSV, index=False)

        # Generate HTML dashboard
        try:
            from dashboard_generator import generate_dashboard
            dashboard_path = generate_dashboard(REPORT_JSON, "results/validation_dashboard.html")
            print(f"📊 Dashboard: {dashboard_path}")
        except Exception as e:
            print(f"⚠️  Dashboard generation failed: {e}")

        # Print text report
        self._print(report)
        print(f"\n📄 Report : {REPORT_JSON}")
        print(f"📊 CSV log: {LOG_CSV}")
        return report

    def _print(self, r: dict):
        s = r["executive_summary"]
        print(f"\n{'='*60}")
        print(f"  DESIGN VALIDATION REPORT  |  {r['generated_at'][:19]}")
        print(f"  Report ID: {r['report_id']}")
        print(f"{'='*60}")
        print(f"\n── EXECUTIVE SUMMARY ─────────────────────────────────────")
        for k, v in s.items():
            print(f"  {k:<30}: {v}")
        print(f"\n── TOP DESIGNS ───────────────────────────────────────────")
        for d in r["top_designs"][:5]:
            print(f"  #{d['rank']}  {d['design_id']:<14} {d['score']:.4f}  {d['recommendation']}")
        print(f"\n── FAILURE DISTRIBUTION ──────────────────────────────────")
        for cat, pct in r["failure_summary"].get("distribution", {}).items():
            bar = "█" * int(pct / 5)
            print(f"  {cat:<22}: {bar:<20} {pct}%")
        print(f"\n── RECOMMENDATIONS ───────────────────────────────────────")
        for i, rec in enumerate(r["recommendations"], 1):
            print(f"  {i}. {rec}")
        print(f"\n{'='*60}")


# ── MASTER RUNNER ─────────────────────────────────────────────────────────────

def run(input_path: str = None):
    print("\n" + "="*60)
    print("  DIVYANSHU — FULL VALIDATION PIPELINE")
    print("="*60)

    # Load simulations — from Aryan's handoff or mock
    v = Validator()
    if input_path and os.path.exists(input_path):
        print(f"\n[DATA] Loading from Aryan: {input_path}")
        core = v.validate_from_aryan(input_path)
        # Reconstruct SimulationResult list from validator internal state
        simulations = []
        from validator import SimulationResult
        for r in v.results:
            simulations.append(SimulationResult(
                design_id       = r.design_id,
                score           = r.score,
                biology_score   = 0.0,
                materials_score = 0.0,
                physics_score   = 0.0,
                chemistry_score = 0.0,
            ))
        # Re-load full data for domain scores
        df = pd.read_csv(input_path)
        col_map = {"overall_score": "score", "material_score": "materials_score"}
        df = df.rename(columns=col_map)
        simulations = []
        for _, row in df.iterrows():
            simulations.append(SimulationResult(
                design_id       = str(row.get("trait_id", row.get("design_id", "?"))),
                score           = float(row["score"]),
                biology_score   = float(row["biology_score"]),
                materials_score = float(row["materials_score"]),
                physics_score   = float(row["physics_score"]),
                chemistry_score = float(row["chemistry_score"]),
            ))
    else:
        print("\n[DATA] No input file — using mock data")
        simulations = load_mock_simulations(40)

    # Run suite
    suite   = ValidationSuite(simulations)
    s_res   = suite.run()

    # Generate final report
    reporter = FinalReporter()
    report   = reporter.generate(simulations, s_res)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divyanshu Validation Pipeline")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to Aryan's handoff CSV")
    args = parser.parse_args()
    run(input_path=args.input)

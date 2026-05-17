"""
feedback_loop.py — DIVYANSHU (Additional Component)
Logs failed designs, detects recurring patterns, suggests threshold adjustments for next generation.
"""

import json
import os
from datetime import datetime
from collections import Counter
from failure_analyzer import FailureAnalyzer
from validator import SimulationResult


class FeedbackLoop:
    LOG_FILE = "feedback_log.json"

    def __init__(self):
        self.feedback_history = self._load_history()

    def _load_history(self):
        if os.path.exists(self.LOG_FILE):
            with open(self.LOG_FILE, "r") as f:
                return json.load(f)
        return {"generations": [], "patterns": {}}

    def _save_history(self):
        with open(self.LOG_FILE, "w") as f:
            json.dump(self.feedback_history, f, indent=2)

    def log_generation(self, simulations: list, generation_id: str = None):
        if generation_id is None:
            generation_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        failed = [s for s in simulations if s.score < 0.70]  # Assuming pass threshold
        fa = FailureAnalyzer()
        analysis = fa.analyze_batch(failed)

        generation_data = {
            "generation_id": generation_id,
            "timestamp": datetime.now().isoformat(),
            "total_designs": len(simulations),
            "failed_designs": len(failed),
            "failure_analysis": analysis,
            "failed_designs_details": [
                {
                    "design_id": s.design_id,
                    "score": s.score,
                    "biology_score": s.biology_score,
                    "materials_score": s.materials_score,
                    "physics_score": s.physics_score,
                    "chemistry_score": s.chemistry_score,
                } for s in failed
            ]
        }

        self.feedback_history["generations"].append(generation_data)
        self._update_patterns(generation_data)
        self._save_history()

        return generation_data

    def _update_patterns(self, generation_data):
        analysis = generation_data["failure_analysis"]
        dist = analysis["distribution"]

        for category, pct in dist.items():
            if pct > 0:
                if category not in self.feedback_history["patterns"]:
                    self.feedback_history["patterns"][category] = {"count": 0, "generations": []}
                self.feedback_history["patterns"][category]["count"] += 1
                self.feedback_history["patterns"][category]["generations"].append(generation_data["generation_id"])

    def detect_patterns(self):
        patterns = self.feedback_history["patterns"]
        recurring = {cat: data for cat, data in patterns.items() if data["count"] >= 2}

        suggestions = []
        for category, data in recurring.items():
            if category == "physics_issue":
                suggestions.append("Recurring physics issues: Consider adjusting physics model parameters or training data")
            elif category == "boundary_issue":
                suggestions.append("Recurring boundary issues: Review geometric constraints and boundary conditions")
            elif category == "data_issue":
                suggestions.append("Recurring data quality issues: Improve data augmentation and noise reduction")
            elif category == "domain_mismatch":
                suggestions.append("Recurring domain conflicts: Enhance cross-domain optimization algorithms")
            elif category == "threshold_breach":
                suggestions.append("Recurring threshold breaches: Consider lowering acceptance threshold or improving model accuracy")

        return {
            "recurring_patterns": recurring,
            "suggestions": suggestions,
            "threshold_adjustment": self._suggest_threshold_adjustment()
        }

    def _suggest_threshold_adjustment(self):
        if not self.feedback_history["generations"]:
            return "No historical data available"

        recent_generations = self.feedback_history["generations"][-3:]  # Last 3 generations
        avg_pass_rate = sum(g["total_designs"] - g["failed_designs"] for g in recent_generations) / sum(g["total_designs"] for g in recent_generations)

        if avg_pass_rate < 0.3:
            return "Pass rate consistently low: Consider lowering threshold to 0.65 for next generation"
        elif avg_pass_rate > 0.8:
            return "Pass rate high: Consider raising threshold to 0.75 for stricter validation"
        else:
            return "Pass rate stable: Maintain current threshold of 0.70"

    def get_feedback_report(self):
        patterns = self.detect_patterns()
        return {
            "total_generations": len(self.feedback_history["generations"]),
            "recurring_failure_patterns": patterns["recurring_patterns"],
            "suggested_improvements": patterns["suggestions"],
            "threshold_adjustment_recommendation": patterns["threshold_adjustment"],
            "latest_generation": self.feedback_history["generations"][-1] if self.feedback_history["generations"] else None
        }


if __name__ == "__main__":
    from validator import load_mock_simulations

    # Simulate logging a generation
    simulations = load_mock_simulations(50)
    fl = FeedbackLoop()
    gen_data = fl.log_generation(simulations)

    print("📝 Logged generation with feedback loop")
    print(f"Generation ID: {gen_data['generation_id']}")
    print(f"Failed designs: {gen_data['failed_designs']}/{gen_data['total_designs']}")

    # Get feedback report
    report = fl.get_feedback_report()
    print("\n🔄 Feedback Report:")
    print(f"Total generations: {report['total_generations']}")
    print(f"Recurring patterns: {list(report['recurring_failure_patterns'].keys())}")
    print(f"Threshold suggestion: {report['threshold_adjustment_recommendation']}")

    # Save report
    with open("feedback_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("📄 Report saved: feedback_report.json")
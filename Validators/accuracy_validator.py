"""
accuracy_validator.py  —  DIVYANSHU  (Task 3)
Compares surrogate predictions vs full physics.
Targets: accuracy >95%, FP <5%, FN <10%
"""

import random
from dataclasses import dataclass, field


@dataclass
class SurrogatePrediction:
    design_id:         str
    surrogate_score:   float
    full_physics_score:float
    surrogate_pass:    bool
    full_physics_pass: bool


@dataclass
class AccuracyMetrics:
    total_samples:      int
    true_positives:     int
    true_negatives:     int
    false_positives:    int
    false_negatives:    int
    accuracy:           float
    false_positive_rate:float
    false_negative_rate:float
    precision:          float
    recall:             float
    f1_score:           float
    passed:             bool
    violations:         list = field(default_factory=list)

    def to_dict(self):
        return {
            "total_samples": self.total_samples,
            "confusion_matrix": {
                "TP": self.true_positives, "TN": self.true_negatives,
                "FP": self.false_positives,"FN": self.false_negatives,
            },
            "metrics": {
                "accuracy_pct":      round(self.accuracy * 100, 2),
                "false_positive_pct":round(self.false_positive_rate * 100, 2),
                "false_negative_pct":round(self.false_negative_rate * 100, 2),
                "precision":         round(self.precision, 4),
                "recall":            round(self.recall, 4),
                "f1_score":          round(self.f1_score, 4),
            },
            "passed":     self.passed,
            "violations": self.violations,
        }


class AccuracyValidator:
    ACCURACY_THRESHOLD  = 0.95
    FALSE_POSITIVE_MAX  = 0.05
    FALSE_NEGATIVE_MAX  = 0.10
    PASS_THRESHOLD      = 0.60  # Adjusted to reflect improved scoring

    def compute_metrics(self, predictions: list) -> AccuracyMetrics:
        tp = tn = fp = fn = 0
        for p in predictions:
            if   p.surrogate_pass and p.full_physics_pass:     tp += 1
            elif not p.surrogate_pass and not p.full_physics_pass: tn += 1
            elif p.surrogate_pass and not p.full_physics_pass: fp += 1
            else:                                               fn += 1

        n        = len(predictions)
        accuracy = (tp + tn) / n if n else 0
        fp_rate  = fp / (fp + tn) if (fp + tn) else 0
        fn_rate  = fn / (fn + tp) if (fn + tp) else 0
        precision= tp / (tp + fp) if (tp + fp) else 0
        recall   = tp / (tp + fn) if (tp + fn) else 0
        f1       = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        violations = []
        if accuracy  < self.ACCURACY_THRESHOLD: violations.append(f"Accuracy {accuracy*100:.2f}% < 95%")
        if fp_rate   > self.FALSE_POSITIVE_MAX: violations.append(f"FP rate {fp_rate*100:.2f}% > 5%")
        if fn_rate   > self.FALSE_NEGATIVE_MAX: violations.append(f"FN rate {fn_rate*100:.2f}% > 10%")

        return AccuracyMetrics(
            total_samples=n, true_positives=tp, true_negatives=tn,
            false_positives=fp, false_negatives=fn,
            accuracy=accuracy, false_positive_rate=fp_rate,
            false_negative_rate=fn_rate, precision=precision,
            recall=recall, f1_score=f1,
            passed=len(violations)==0, violations=violations,
        )

    def validate(self, predictions: list) -> AccuracyMetrics:
        return self.compute_metrics(predictions)

    def compute_metrics_at_threshold(self, predictions: list, threshold: float) -> AccuracyMetrics:
        projected = []
        for p in predictions:
            projected.append(SurrogatePrediction(
                design_id=p.design_id,
                surrogate_score=p.surrogate_score,
                full_physics_score=p.full_physics_score,
                surrogate_pass=p.surrogate_score >= threshold,
                full_physics_pass=p.full_physics_score >= self.PASS_THRESHOLD,
            ))
        return self.compute_metrics(projected)

    def calibrate_threshold(
        self,
        predictions: list,
        fp_max: float = 0.05,
        fp_target: float = 0.04,
        fp_floor: float = 0.01,
        min_accuracy: float = 0.97,  # Increased from 0.95
        min_recall: float = 0.90,
    ) -> dict:
        candidates = []
        for i in range(50, 91):
            t = i / 100.0
            m = self.compute_metrics_at_threshold(predictions, t)
            candidates.append((t, m))

        # Prioritize metrics that minimize false negatives
        in_band = [
            (t, m) for (t, m) in candidates
            if fp_floor <= m.false_positive_rate <= fp_max
            and m.accuracy >= min_accuracy
            and m.recall >= min_recall
        ]
        if in_band:
            # Prefer lower threshold to catch more true positives (reduce FN)
            best = min(
                in_band,
                key=lambda tm: (
                    -tm[1].recall,  # Maximize recall (minimize FN)
                    abs(tm[1].false_positive_rate - fp_target),
                    -tm[1].accuracy,
                ),
            )
        else:
            # If strict criteria not met, find best compromise
            under_cap = [
                (t, m) for (t, m) in candidates
                if m.false_positive_rate <= fp_max
                and m.accuracy >= 0.95
                and m.recall >= 0.85  # More lenient recall requirement
            ]
            if under_cap:
                best = max(under_cap, key=lambda tm: (tm[1].recall, tm[1].accuracy, -tm[1].false_positive_rate))
            else:
                # Ultimate fallback: maximize recall to minimize false negatives
                best = max(candidates, key=lambda tm: tm[1].recall)

        t, metrics = best
        return {
            "recommended_threshold": t,
            "meets_fp_target": metrics.false_positive_rate <= fp_max,
            "metrics": metrics.to_dict(),
        }

    def validate_with_calibration(self, predictions: list, fp_max: float = 0.05):
        cal = self.calibrate_threshold(predictions, fp_max=fp_max)
        metrics = self.compute_metrics_at_threshold(predictions, cal["recommended_threshold"])
        return metrics, cal


def generate_mock_predictions(n=100, noise=0.08) -> list:
    """Generate mock predictions with realistic surrogate model behavior.
    Higher noise (0.08) simulates real surrogate model uncertainty.
    """
    random.seed(99)
    preds = []
    for i in range(n):
        full  = random.uniform(0.4, 1.0)
        # Simulate realistic surrogate errors: biased toward underestimating
        surr  = full + random.gauss(-0.02, noise)  # Systematic bias + noise
        surr  = max(0.0, min(1.0, surr))
        preds.append(SurrogatePrediction(
            design_id=f"D{i+1:03d}",
            surrogate_score=surr,
            full_physics_score=full,
            surrogate_pass=surr >= 0.60,  # Use calibrated threshold
            full_physics_pass=full >= 0.60,
        ))
    return preds


if __name__ == "__main__":
    import json
    preds   = generate_mock_predictions(100)
    av      = AccuracyValidator()
    metrics = av.validate(preds)
    print(json.dumps(metrics.to_dict(), indent=2))
    print("✅ PASSED" if metrics.passed else "❌ FAILED")

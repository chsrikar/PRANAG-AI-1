"""
fp_fixer.py — DIVYANSHU (Additional Component)
Sweeps thresholds 0.60–0.85 to find optimal false positive rate below 5%.
"""

import json
from accuracy_validator import AccuracyValidator, generate_mock_predictions


def sweep_thresholds(predictions, start=0.60, end=0.85, step=0.01, fp_target=0.05):
    av = AccuracyValidator()
    results = []
    best = None
    best_fp = float('inf')

    print("🔍 Sweeping thresholds for optimal FP rate...")
    print(f"Range: {start} to {end}, Target FP: <{fp_target*100}%")
    print("-" * 50)

    threshold = start
    while threshold <= end:
        metrics = av.compute_metrics_at_threshold(predictions, threshold)
        fp_rate = metrics.false_positive_rate
        acc = metrics.accuracy
        f1 = metrics.f1_score

        results.append({
            "threshold": round(threshold, 2),
            "fp_rate": round(fp_rate, 4),
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "below_target": fp_rate <= fp_target
        })

        if fp_rate <= fp_target and (best is None or fp_rate < best_fp or (fp_rate == best_fp and acc > best["accuracy"])):
            best = results[-1]
            best_fp = fp_rate

        print(f"Threshold {threshold:.2f}: FP={fp_rate*100:.2f}%, Acc={acc*100:.2f}%, F1={f1:.4f} {'✅' if fp_rate <= fp_target else '❌'}")

        threshold += step

    print("-" * 50)
    if best:
        print(f"🎯 Optimal threshold: {best['threshold']} (FP: {best['fp_rate']*100:.2f}%, Acc: {best['accuracy']*100:.2f}%)")
    else:
        print("❌ No threshold found below FP target")

    return results, best


if __name__ == "__main__":
    predictions = generate_mock_predictions(1000, noise=0.05)  # More samples for better sweep
    results, optimal = sweep_thresholds(predictions)

    # Save results
    with open("fp_sweep_results.json", "w") as f:
        json.dump({"results": results, "optimal": optimal}, f, indent=2)

    print("📄 Results saved: fp_sweep_results.json")
# PRANAG-AI Validation Stack


## Files

- `validator.py` - Core validator + real data loader
- `cross_domain_validator.py` - Sequential domain gate validator
- `accuracy_validator.py` - Accuracy + FPR/FNR metrics + threshold calibration
- `failure_analyzer.py` - Failure root-cause analysis + feedback dataset export
- `decision_logger.py` - Persistent decision logging
- `validation_suite.py` - Full suite runner
- `reporting.py` - Executive + technical reporting

## Real Data Integration

Use `load_real_simulations(input_path, limit)` from `validator.py`.

Supported:
- `.csv`
- `.parquet` (requires pandas + pyarrow)

Common aliases are auto-mapped:
- `design_id`, `trait_id`, `id`
- `biology_score` / `gene_expression`
- `materials_score` / `strength`
- `physics_score` / `temperature`
- `chemistry_score` / `conductivity` / `chemistry_signal`
- `score` / `viability_score` / `overall_score`

## Persistent Logging

Validation decisions are written automatically when using logger-enabled validators.

Outputs:
- `logs/validation_decisions.jsonl`
- `logs/validation_decisions.db`

## False Positive Reduction

Use:
- `AccuracyValidator.calibrate_threshold(predictions, fp_max=0.05)`

This sweeps threshold values and returns a recommended value prioritizing:
1. Lower false-positive rate
2. Higher accuracy

## Feedback Loop

Use:
- `FailureAnalyzer.export_feedback_dataset(failed_sims, out_dir="feedback")`

Outputs:
- `feedback/failed_designs_feedback.json`
- `feedback/hard_negatives.csv`

`hard_negatives.csv` can be fed into retraining as negative examples.

## Run Examples

### 1) Full validation suite (mock data)

```bash
py -3.14 validation_suite.py
```

### 2) Full validation suite with real data

```python
from validation_suite import ValidationSuite

suite = ValidationSuite(n_designs=200, real_data_path="data/raw/materials.csv")
report = suite.run()
print(report.to_dict())
```

### 3) Reporting with real data

```python
from reporting import ReportingSystem

rs = ReportingSystem(n_designs=200, real_data_path="data/processed/universal_index.parquet")
report = rs.generate()
print(report.to_text())
```

## What changed vs before

- Removed mock-only limitation by adding `load_real_simulations()`
- Added decision persistence for auditability
- Added FPR-aware threshold optimizer
- Added failure-to-retraining feedback artifacts

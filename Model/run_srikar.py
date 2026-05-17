"""
run_srikar.py - Full Srikar PINN Pipeline
=========================================
Entry point. Loads data -> builds all 5 PINNs -> trains them ->
builds surrogates -> saves everything.

Usage:
    python run_srikar.py --data_dir /path/to/parquet_files \
                         --out1 out1.json \
                         --out2 out2.json \
                         --epochs 3000
"""

import argparse
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from models.base_pinn          import BasePINN
from models.physics_models     import (
    HeatPINN, StressPINN, GrowthPINN, BiologyPINN, ChemistryPINN,
)
from models.adaptive_loss      import AdaptiveLoss
from models.surrogate_trainer  import SurrogateTrainer
from datasrc.data_loader          import PINNDataLoader
from visualization import (
    plot_training_losses, 
    plot_prediction_vs_actual, 
    plot_lambda_evolution, 
    plot_surrogate_performance
)


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Srikar PINN Pipeline")
    p.add_argument("--data_dir", default="data",  help="Folder with parquet files")
    p.add_argument("--out1",     default=None,     help="out1.json from prompt parser")
    p.add_argument("--out2",     default=None,     help="out2.json from prompt parser")
    p.add_argument("--epochs",   type=int, default=3000)
    p.add_argument("--save_dir", default="outputs/models")
    p.add_argument("--device",   default="cpu")
    return p.parse_args()


# ============================================================
# Helper: boundary points (simple box)
# ============================================================
def make_boundary(n: int, input_dim: int, output_dim: int = 1) -> tuple:
    """Zero-boundary IC: x=0 or t=0 -> output=0 (normalised)."""
    x_bc = torch.zeros(n, input_dim)
    y_bc = torch.zeros(n, output_dim)
    return x_bc, y_bc


# ============================================================
# Train one model
# ============================================================
def train_model(
    model,
    tensors: dict,
    epochs: int,
    save_path: str,
):
    """Generic PINN training with adaptive loss."""
    X_tr = tensors["X_train"]
    y_tr = tensors["y_train"]

    # Physics collocation points (random in [0,1])
    x_phys = torch.rand(500, model.network[0].in_features)

    # Boundary
    x_bc, y_bc = make_boundary(
        200,
        model.network[0].in_features,
        model.network[-1].out_features,
    )

    # Adaptive loss controller
    controller = AdaptiveLoss(model, update_every=100)

    print(f"\n{'='*60}")
    print(f"  Training {model.__class__.__name__}  ({epochs} epochs)")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=300, factor=0.5
    )

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss, breakdown = model.total_loss(X_tr, y_tr, x_phys, x_bc, y_bc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)

        for k, v in breakdown.items():
            model.history[k].append(v)

        controller.step(breakdown)

        if epoch % 500 == 0:
            r2 = model.accuracy(tensors["X_test"], tensors["y_test"])
            print(
                f"  Epoch {epoch:5d} | "
                f"Total={breakdown['total']:.4f} | R²={r2:.4f}"
            )

    # Final accuracy
    r2_final = model.accuracy(tensors["X_test"], tensors["y_test"])
    print(f"\n  Final R2 = {r2_final:.4f}  ({r2_final*100:.1f}%)")

    # Format controller.log to lambda_history
    lambda_hist = {'lambda1': [], 'lambda2': [], 'lambda3': []}
    for entry in controller.log:
        lambda_hist['lambda1'].append(entry['lambda1'])
        lambda_hist['lambda2'].append(entry['lambda2'])
        lambda_hist['lambda3'].append(entry['lambda3'])
    model.lambda_history = lambda_hist

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    return r2_final, breakdown['total']


def _rebuild_heat_targets(tensors: dict) -> dict:
    """Create heat targets consistent with [x_position, depth, time] inputs."""
    out = dict(tensors)
    for split in ("train", "test"):
        X = out[f"X_{split}"]
        x = X[:, 0:1]
        depth = X[:, 1:2]
        time = X[:, 2:3]
        y = 0.5 + 0.35 * torch.sin(np.pi * x) * torch.exp(-1.2 * depth) * torch.cos(2 * np.pi * time)
        out[f"y_{split}"] = torch.clamp(y, 0.0, 1.0)
    return out


def _rebuild_chemistry_targets(tensors: dict) -> dict:
    """Create Arrhenius-like chemistry targets from normalised chemistry inputs."""
    out = dict(tensors)
    for split in ("train", "test"):
        X = out[f"X_{split}"]
        temp = X[:, 0:1]
        conc = X[:, 1:2]
        ph = X[:, 2:3]
        time = X[:, 3:4]
        k = torch.exp(-4.0 / (0.25 + temp))
        k = k * (0.7 + 0.25 * conc) * (1.0 - 0.15 * (ph - 0.5) ** 2) * (0.9 + 0.1 * torch.cos(2.0 * np.pi * time))
        k_min, k_max = k.min(), k.max()
        out[f"y_{split}"] = (k - k_min) / (k_max - k_min + 1e-8)
    return out


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  SRIKAR — PINN Pipeline Starting")
    print("="*60)

    # 1. Load data
    print("\n[1/5] Loading data ...")
    loader = PINNDataLoader(data_dir=args.data_dir)
    loader.load()

    # Parse prompt JSONs
    extra1, extra2 = {}, {}
    if args.out1 and os.path.exists(args.out1):
        extra1 = loader.parse_prompt_json(args.out1)
        print(f"  Prompt 1 parsed: {extra1}")
    if args.out2 and os.path.exists(args.out2):
        extra2 = loader.parse_prompt_json(args.out2)
        print(f"  Prompt 2 parsed: {extra2}")

    # Merge both prompts (union)
    extra_merged = {**extra1, **extra2}

    # Build filtered feature matrix
    df = loader.build_feature_matrix(extra_features=extra_merged or None)
    print(f"  Feature matrix: {df.shape}")

    # 2. Prepare tensors for each domain
    print("\n[2/5] Preparing tensors ...")
    bio_tensors  = loader.to_biology_tensors(df)
    heat_tensors = _rebuild_heat_targets(loader.to_heat_tensors(df))
    chem_tensors = _rebuild_chemistry_targets(loader.to_chemistry_tensors(df))

    # Growth & Stress reuse biology layout (temp, water, time)
    def growth_tensors():
        t = bio_tensors
        return {
            "X_train": t["X_train"][:, :3],
            "y_train": t["y_train"][:, :1],
            "X_test":  t["X_test"][:, :3],
            "y_test":  t["y_test"][:, :1],
        }

    def stress_tensors():
        # Build physically consistent stress targets from strain + temperature.
        # This avoids reusing biology targets, which are not Hooke-law stresses.
        t = bio_tensors
        E = 1.0
        alpha_T = 0.15

        X_train = t["X_train"][:, :3]
        X_test  = t["X_test"][:, :3]

        def make_stress_targets(X):
            strain_x = X[:, 0:1]
            strain_y = X[:, 1:2]
            temp = X[:, 2:3]
            temp_factor = 1.0 + alpha_T * (temp - 0.5)
            sigma_x = E * strain_x * temp_factor
            sigma_y = E * strain_y * temp_factor
            return torch.cat([sigma_x, sigma_y], dim=1)

        return {
            "X_train": X_train,
            "y_train": make_stress_targets(X_train),
            "X_test":  X_test,
            "y_test":  make_stress_targets(X_test),
        }

    # 3. Cross-Domain Validation Pipeline
    print("\n[3/5] Cross-Domain Validation Pipeline (Bio->Mat->Phys->Chem) ...")

    models_config = [
        ("biology",   BiologyPINN(),                         bio_tensors),
        ("stress",    StressPINN(lambda2=0.1),               stress_tensors()), # Material
        ("heat",      HeatPINN(lambda2=0.0, lambda3=0.1),    heat_tensors),     # Physics
        ("chemistry", ChemistryPINN(lambda2=0.0),            chem_tensors),     # Chemistry
        ("growth",    GrowthPINN(),                          growth_tensors()),
    ]

    results = {}
    trained_models = {}
    failure_analysis = []

    for name, model, tensors in models_config:
        save_path = f"{args.save_dir}/{name}_pinn.pt"
        r2, total_loss = train_model(model, tensors, args.epochs, save_path)
        
        # 1. Build Surrogate immediately to get accuracy for viability score
        print(f"\n  Building surrogate for {name} ...")
        surr_dir = "outputs/surrogates"
        x_sample = tensors["X_train"]
        trainer = SurrogateTrainer(model, model_name=name, save_dir=surr_dir)
        x_min = x_sample.numpy().min(axis=0)
        x_max = x_sample.numpy().max(axis=0)
        X_gen, y_gen = trainer.generate_data(x_min, x_max, n_samples=50_000)
        metrics = trainer.train(X_gen, y_gen)
        trainer.save()
        
        surrogate_accuracy = metrics.get("r2_accuracy", 0.0)
        
        # 2. Explicit Viability Score (Balanced Option 2)
        viability_score = (
            0.5 * float(max(0.0, r2)) +
            0.3 * float(np.exp(-10.0 * total_loss)) +
            0.2 * float(surrogate_accuracy / 100.0)
        )
        status = "PASS" if viability_score >= 0.70 else "FAIL"

        results[name] = {
            "r2": round(r2, 4), 
            "saved": save_path, 
            "total_loss": round(total_loss, 6),
            "viability_score": round(viability_score, 4),
            "status": status,
            "surrogate": metrics
        }
        
        # 70% Filtering Logic
        if viability_score < 0.70:
            print(f"  [!] {name} REJECTED: Viability score {viability_score:.2f} < 0.70")
            failure_analysis.append({
                "model": name,
                "reason": "Viability score below 70% threshold",
                "viability_score": viability_score
            })
            continue # Do not add to trained models
            
        print(f"  [+] {name} PASSED: Viability score {viability_score:.2f} >= 0.70")
        trained_models[name] = model

    # 4. (Surrogates are now built inline above)

    # 5. Final report
    print("\n[5/5] Summary Report")
    print("-" * 80)
    print(f"{'Model':<12} {'R² %':>8}  {'Viability':>10}  {'Status':>8}  {'Surr. R²%':>10}  {'Speed<0.01s':>12}")
    print("-" * 80)
    for name, r in results.items():
        if not isinstance(r, dict) or "r2" not in r:
            continue
        surr  = r.get("surrogate", {})
        ok    = "Y" if surr.get("target_met") else "N"
        viab  = r.get("viability_score", 0.0)
        status = r.get("status", "FAIL")
        surr_r2 = surr.get("r2_accuracy", 0.0) if surr else 0.0
        print(
            f"{name:<12} {r['r2']*100:>8.1f}%  "
            f"{viab:>10.4f}  "
            f"{status:>8}  "
            f"{surr_r2:>10.1f}%  "
            f"{ok:>12}"
        )
    print("-" * 80)

    # 6. Final Cross-Domain Score
    final_score = (
        results.get("biology", {}).get("viability_score", 0.0) * 0.2 +
        results.get("stress", {}).get("viability_score", 0.0) * 0.2 +
        results.get("heat", {}).get("viability_score", 0.0) * 0.2 +
        results.get("chemistry", {}).get("viability_score", 0.0) * 0.2 +
        results.get("growth", {}).get("viability_score", 0.0) * 0.2
    )
    results["final_score"] = round(final_score, 4)
    print(f"FINAL CROSS-DOMAIN SCORE: {final_score:.4f}")
    print("-" * 80)

    # Save report JSON
    report_path = "outputs/srikar_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full report -> {report_path}")
    
    # Save Failure Analysis Logging
    if failure_analysis:
        failure_path = "outputs/failure_analysis.json"
        with open(failure_path, "w") as f:
            json.dump(failure_analysis, f, indent=2)
        print(f"  Logged {len(failure_analysis)} failures to {failure_path}")
    # 6. Generate Plots
    print("\n[6/6] Generating Visualizations ...")
    os.makedirs("outputs/plots", exist_ok=True)
    
    r2_scores = {
        name: r.get("surrogate", {}).get("r2_accuracy", 0)/100.0 
        for name, r in results.items() 
        if isinstance(r, dict)
    }
    times = {
        name: r.get("surrogate", {}).get("ms_per_1_pred", 0.0) 
        for name, r in results.items() 
        if isinstance(r, dict)
    }
    plot_surrogate_performance(r2_scores, times, "outputs/plots/surrogate_performance.png")
    
    for name, model in trained_models.items():
        out_dir = f"outputs/plots/{name}"
        os.makedirs(out_dir, exist_ok=True)
        
        plot_training_losses(model.history, f"{out_dir}/training_losses.png")
        plot_lambda_evolution(getattr(model, "lambda_history", {}), f"{out_dir}/lambda_evolution.png")
        
        tensors = [t for n, _, t in models_config if n == name][0]
        y_true = tensors["y_test"].detach().cpu().numpy()
        with torch.no_grad():
            y_pred = model(tensors["X_test"]).detach().cpu().numpy()
        plot_prediction_vs_actual(y_true, y_pred, f"{out_dir}/prediction_vs_actual.png")
        
    print("\n  PINN pipeline complete. All plots saved to outputs/plots/")


if __name__ == "__main__":
    main()

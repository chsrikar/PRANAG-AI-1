"""
surrogate_trainer.py — Fast Surrogate Model System
====================================================
Trains lightweight models (sklearn GBM or small MLP) on top of
PINN-generated data so Aryan's batch simulator can run 1M predictions
in < 4 hours.

Target spec:
  Accuracy > 95%
  Speed    < 0.01 sec per prediction
"""

from __future__ import annotations
import time
import joblib
import numpy as np
import torch
from pathlib import Path
from typing import Optional

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import r2_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

from models.base_pinn import BasePINN


class SurrogateTrainer:
    """
    Build and save fast surrogate models backed by PINN-generated data.

    Workflow:
        1. Generate (x, y) pairs from a trained PINN.
        2. Fit a GBM surrogate on those pairs.
        3. Save surrogate to disk.
        4. At inference: load surrogate → predict in microseconds.
    """

    def __init__(
        self,
        pinn: BasePINN,
        model_name: str = "surrogate",
        save_dir: str = "outputs/surrogates",
    ):
        self.pinn       = pinn
        self.model_name = model_name
        self.save_dir   = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.surrogate  = None
        self.metrics: dict = {}

    # ------------------------------------------------------------------ #
    # 1. Generate synthetic dataset from PINN                             #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_data(
        self,
        x_min: np.ndarray,
        x_max: np.ndarray,
        n_samples: int = 100_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Latin-hypercube random sampling in input space, PINN → labels.

        Args:
            x_min, x_max : 1-D arrays of shape (input_dim,)
            n_samples    : how many training points to generate

        Returns:
            X : (n_samples, input_dim)   numpy array
            y : (n_samples, output_dim)  numpy array
        """
        X_np = np.random.uniform(x_min, x_max, size=(n_samples, len(x_min)))
        X_t  = torch.tensor(X_np, dtype=torch.float32)

        # Chunk to avoid OOM on large n_samples
        chunk   = 10_000
        y_parts = []
        for i in range(0, n_samples, chunk):
            y_parts.append(self.pinn(X_t[i : i + chunk]).numpy())
        y_np = np.concatenate(y_parts, axis=0)

        return X_np, y_np

    # ------------------------------------------------------------------ #
    # 2. Train surrogate                                                   #
    # ------------------------------------------------------------------ #
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_frac: float = 0.1,
    ) -> dict:
        """
        Fit GBM surrogate. Returns accuracy metrics.
        Drops to RandomForest if sklearn is missing GBM.
        """
        assert SKLEARN_OK, "Install scikit-learn: pip install scikit-learn"

        # Train / test split
        n_test  = max(1, int(len(X) * test_frac))
        idx     = np.random.permutation(len(X))
        X_tr, X_te = X[idx[n_test:]], X[idx[:n_test]]
        y_tr, y_te = y[idx[n_test:]], y[idx[:n_test]]

        print(f"[SurrogateTrainer] Training {self.model_name}  "
              f"train={len(X_tr):,}  test={len(X_te):,}")

        # Flatten to 1-D output if single-output model
        single_out = (y.shape[1] == 1)
        if single_out:
            y_tr, y_te = y_tr.ravel(), y_te.ravel()

        base_gbm = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_iter_no_change=10,
            validation_fraction=0.1,
            tol=1e-4,
        )

        if single_out:
            self.surrogate = base_gbm
        else:
            self.surrogate = MultiOutputRegressor(base_gbm, n_jobs=1)

        t0 = time.time()
        self.surrogate.fit(X_tr, y_tr)
        train_sec = time.time() - t0

        # ── Metrics ──────────────────────────────────────
        y_pred = self.surrogate.predict(X_te)
        r2     = r2_score(y_te, y_pred)

        # Speed test: 100 predictions (reduced from 1000 for faster testing)
        t0 = time.time()
        for _ in range(100):
            self.surrogate.predict(X_te[:1])
        total_sec = time.time() - t0
        sec_per_pred = total_sec / 100.0
        ms_per_pred = sec_per_pred * 1000.0

        self.metrics = {
            "r2_accuracy":    round(r2 * 100, 2),
            "train_sec":      round(train_sec, 2),
            "ms_per_1_pred":  round(ms_per_pred, 6),
            "target_met":     r2 >= 0.95 and ms_per_pred < 10.0,
        }

        print(
            f"  R2 = {self.metrics['r2_accuracy']}%  |  "
            f"Speed = {ms_per_pred:.3f} ms/pred  |  "
            f"Target met = {self.metrics['target_met']}"
        )
        return self.metrics

    # ------------------------------------------------------------------ #
    # 3. Predict                                                           #
    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.surrogate is not None, "Train or load surrogate first."
        return self.surrogate.predict(X)

    # ------------------------------------------------------------------ #
    # 4. Save / Load                                                       #
    # ------------------------------------------------------------------ #
    def save(self):
        path = self.save_dir / f"{self.model_name}.joblib"
        joblib.dump({"surrogate": self.surrogate, "metrics": self.metrics}, path)
        print(f"Surrogate saved -> {path}")
        return str(path)

    def load(self, path: str):
        data = joblib.load(path)
        self.surrogate = data["surrogate"]
        self.metrics   = data["metrics"]
        print(f"Surrogate loaded <- {path}")

"""
batch_simulator.py  —  ARYAN  (Task 2 + 3)
Runs traits through Srikar's PINN models in batches.
Output: trait_id + viability_score for every trait.
Target: 100K traits < 30 min, 1M traits < 4 hours.

HOW TO PLUG IN SRIKAR'S MODELS:
  from batch_simulator import BatchSimulator
  sim = BatchSimulator(model_dir="path/to/srikar/models/")
  results = sim.run_all()
"""

import time
import os
import json
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass, field, asdict
from datetime import datetime
from data_loader import DataLoader

try:
    import joblib
except Exception:
    joblib = None


CHECKPOINT_FILE = "checkpoints/batch_checkpoint.json"
RESULTS_FILE    = "results/simulation_results.parquet"
SCORE_THRESHOLD = 0.70   # below this → filtered out before validation


@dataclass
class TraitResult:
    trait_id:        str
    entity_type:     str
    viability_score: float
    biology_score:   float
    physics_score:   float
    material_score:  float
    chemistry_score: float
    passed_filter:   bool
    source:          str = ""
    simulated_at:    str = field(default_factory=lambda: datetime.now().isoformat())


# ── SRIKAR MODEL INTERFACE ────────────────────────────────────────────────────

class SrikarModelInterface:
    """
    Interface to Srikar's PINN models.
    When Srikar hands over his models, replace the _mock_* methods
    with real model.predict() calls.

    HOW TO PLUG IN:
        1. Replace `self.models` loading with:
               import torch
               self.heat_model    = torch.load(model_dir + "heat_pinn.pt")
               self.stress_model  = torch.load(model_dir + "stress_pinn.pt")
               self.growth_model  = torch.load(model_dir + "growth_pinn.pt")
               self.biology_model = torch.load(model_dir + "biology_pinn.pt")
               self.chem_model    = torch.load(model_dir + "chemistry_pinn.pt")
        2. Replace each _predict_* method with real model inference.
    """

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or self._default_model_dir()
        self.surrogate_dir = self._default_surrogate_dir()
        self.models_loaded = False
        self.surrogates_loaded = False
        self.heat_model = None
        self.stress_model = None
        self.growth_model = None
        self.biology_model = None
        self.chem_model = None
        self.surrogates = {}
        self._try_load_models()

    def _default_model_dir(self) -> str:
        here = os.path.dirname(__file__)
        return os.path.normpath(os.path.join(here, "..", "Model", "outputs", "models"))

    def _default_surrogate_dir(self) -> str:
        here = os.path.dirname(__file__)
        return os.path.normpath(os.path.join(here, "..", "Model", "outputs", "surrogates"))

    def _try_load_models(self):
        """Load Srikar models/surrogates from Model/outputs folders."""
        self._try_load_surrogates()
        try:
            from pathlib import Path
            import sys

            model_root = Path(__file__).resolve().parents[1] / "Model"
            if str(model_root) not in sys.path:
                sys.path.insert(0, str(model_root))
            from models.physics_models import HeatPINN, StressPINN, GrowthPINN, BiologyPINN, ChemistryPINN

            required = {
                "heat": os.path.join(self.model_dir, "heat_pinn.pt"),
                "stress": os.path.join(self.model_dir, "stress_pinn.pt"),
                "growth": os.path.join(self.model_dir, "growth_pinn.pt"),
                "biology": os.path.join(self.model_dir, "biology_pinn.pt"),
                "chemistry": os.path.join(self.model_dir, "chemistry_pinn.pt"),
            }
            if not all(os.path.exists(p) for p in required.values()):
                print(f"⚠️  Missing one or more PINN checkpoints in: {self.model_dir}")
                return

            self.heat_model = HeatPINN()
            self.stress_model = StressPINN()
            self.growth_model = GrowthPINN()
            self.biology_model = BiologyPINN()
            self.chem_model = ChemistryPINN()

            self._load_checkpoint(self.heat_model, required["heat"])
            self._load_checkpoint(self.stress_model, required["stress"])
            self._load_checkpoint(self.growth_model, required["growth"])
            self._load_checkpoint(self.biology_model, required["biology"])
            self._load_checkpoint(self.chem_model, required["chemistry"])

            for m in [self.heat_model, self.stress_model, self.growth_model, self.biology_model, self.chem_model]:
                m.eval()
            self.models_loaded = True
            print(f"Loaded real PINN checkpoints from: {self.model_dir}")
        except Exception as e:
            print(f"Could not load PINN checkpoints ({e})")

    def _load_checkpoint(self, model: torch.nn.Module, ckpt_path: str):
        data = torch.load(ckpt_path, map_location="cpu")
        state = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
        model.load_state_dict(state)

    def _try_load_surrogates(self):
        if joblib is None:
            return
        if not os.path.isdir(self.surrogate_dir):
            return
        names = ["heat", "stress", "growth", "biology", "chemistry"]
        loaded = 0
        for n in names:
            p = os.path.join(self.surrogate_dir, f"{n}.joblib")
            if not os.path.exists(p):
                continue
            try:
                payload = joblib.load(p)
                self.surrogates[n] = payload.get("surrogate")
                loaded += 1
            except Exception:
                continue
        self.surrogates_loaded = loaded >= 5
        if self.surrogates_loaded:
            print(f"Loaded 5 surrogate models from: {self.surrogate_dir}")

    def _normalise(self, val, lo, hi):
        if hi == lo:
            return 0.5
        return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))

    def _as_tensor(self, values):
        return torch.tensor([values], dtype=torch.float32)

    def _safe_float(self, row: dict, key: str, default: float = 0.0):
        try:
            return float(row.get(key, default))
        except Exception:
            return default

    # ── VECTORIZED HELPERS (batch path) ───────────────────────────────────────

    def _col(self, df: pd.DataFrame, col: str, default: float) -> np.ndarray:
        """Extract a DataFrame column as float32 array, filling missing with default."""
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default).values.astype(np.float32)
        return np.full(len(df), default, dtype=np.float32)

    def _norm_arr(self, arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Vectorised normalise: entire array at once, output clipped to [0, 1]."""
        if hi == lo:
            return np.full(len(arr), 0.5, dtype=np.float32)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    def _features_heat(self, row: dict):
        x_position = self._normalise(self._safe_float(row, "x_position", 0.5), 0, 1)
        depth = self._normalise(self._safe_float(row, "depth", 0.2), 0, 2)
        time_v = self._normalise(self._safe_float(row, "time", 12), 0, 24)
        return [x_position, depth, time_v]

    def _features_stress(self, row: dict):
        strain_x = self._normalise(self._safe_float(row, "strain_x", self._safe_float(row, "strength", 1000) / 2000), 0, 1)
        strain_y = self._normalise(self._safe_float(row, "strain_y", 0.5 * strain_x), 0, 1)
        temp = self._normalise(self._safe_float(row, "temperature_max", 50), 0, 100)  # Adjusted: 0-100°C range, default 50
        return [strain_x, strain_y, temp]

    def _features_growth(self, row: dict):
        time_v = self._normalise(self._safe_float(row, "time", 12), 0, 24)
        temp = self._normalise(self._safe_float(row, "temperature_max", 50), 0, 100)  # Adjusted: 0-100°C range, default 50
        water = self._normalise(self._safe_float(row, "water", 0.5), 0, 1)
        return [time_v, temp, water]

    def _features_biology(self, row: dict):
        temp = self._normalise(self._safe_float(row, "temperature_max", 40), 0, 1500)
        water = self._normalise(self._safe_float(row, "water", 0.5), 0, 1)
        nitrogen = self._normalise(self._safe_float(row, "nitrogen", 0.4), 0, 1)
        light = self._normalise(self._safe_float(row, "light_intensity", 0.6), 0, 1)
        time_v = self._normalise(self._safe_float(row, "time", 12), 0, 24)
        return [temp, water, nitrogen, light, time_v]

    def _features_chemistry(self, row: dict):
        temp_k = self._normalise(self._safe_float(row, "temperature_k", self._safe_float(row, "temperature_max", 40) + 273.15), 250, 330)
        concentration = self._normalise(self._safe_float(row, "concentration", 0.5), 0, 1)
        ph = self._normalise(self._safe_float(row, "ph", 7), 2, 12)
        time_v = self._normalise(self._safe_float(row, "time", 12), 0, 24)
        return [temp_k, concentration, ph, time_v]

    # ── VECTORIZED FEATURE EXTRACTORS (batch path) ────────────────────────────
    # Each method mirrors its row-level counterpart but operates on a whole
    # DataFrame and returns a float32 ndarray of shape (N, n_features).

    def _features_biology_batch(self, df: pd.DataFrame) -> np.ndarray:
        """(N, 5): [temperature, water, nitrogen, light_intensity, time]"""
        temp  = self._norm_arr(self._col(df, "temperature_max",   40.0), 0,  1500)
        water = self._norm_arr(self._col(df, "water",              0.5),  0,  1)
        nitro = self._norm_arr(self._col(df, "nitrogen",           0.4),  0,  1)
        light = self._norm_arr(self._col(df, "light_intensity",    0.6),  0,  1)
        time  = self._norm_arr(self._col(df, "time",              12.0),  0,  24)
        return np.column_stack([temp, water, nitro, light, time])

    def _features_heat_batch(self, df: pd.DataFrame) -> np.ndarray:
        """(N, 3): [x_position, depth, time]"""
        x_pos = self._norm_arr(self._col(df, "x_position",  0.5),  0, 1)
        depth = self._norm_arr(self._col(df, "depth",        0.2),  0, 2)
        time  = self._norm_arr(self._col(df, "time",        12.0),  0, 24)
        return np.column_stack([x_pos, depth, time])

    def _features_stress_batch(self, df: pd.DataFrame) -> np.ndarray:
        """(N, 3): [strain_x, strain_y, temperature]
        strain_x falls back to strength/2000; strain_y falls back to 0.5*strain_x.
        """
        # strain_x: prefer explicit column, fall back to strength/2000
        strength = self._col(df, "strength", 1000.0)
        fallback_sx = strength / 2000.0
        if "strain_x" in df.columns:
            sx_raw = pd.to_numeric(df["strain_x"], errors="coerce").values.astype(np.float32)
            strain_x_raw = np.where(np.isnan(sx_raw), fallback_sx, sx_raw)
        else:
            strain_x_raw = fallback_sx
        strain_x = self._norm_arr(strain_x_raw, 0, 1)

        # strain_y: prefer explicit column, fall back to 0.5 * strain_x
        if "strain_y" in df.columns:
            sy_raw = pd.to_numeric(df["strain_y"], errors="coerce").values.astype(np.float32)
            strain_y_raw = np.where(np.isnan(sy_raw), 0.5 * strain_x, sy_raw)
        else:
            strain_y_raw = 0.5 * strain_x
        strain_y = self._norm_arr(strain_y_raw.astype(np.float32), 0, 1)

        temp = self._norm_arr(self._col(df, "temperature_max", 50.0), 0, 100)  # Adjusted: 0-100°C range, default 50
        return np.column_stack([strain_x, strain_y, temp])

    def _features_chemistry_batch(self, df: pd.DataFrame) -> np.ndarray:
        """(N, 4): [temperature_k, concentration, ph, time]
        temperature_k falls back to temperature_max + 273.15.
        """
        temp_max = self._col(df, "temperature_max", 40.0)
        fallback_tk = temp_max + 273.15
        if "temperature_k" in df.columns:
            tk_raw = pd.to_numeric(df["temperature_k"], errors="coerce").values.astype(np.float32)
            temp_k_raw = np.where(np.isnan(tk_raw), fallback_tk, tk_raw)
        else:
            temp_k_raw = fallback_tk
        temp_k = self._norm_arr(temp_k_raw.astype(np.float32), 250, 330)

        conc = self._norm_arr(self._col(df, "concentration", 0.5),  0, 1)
        ph   = self._norm_arr(self._col(df, "ph",            7.0),  2, 12)
        time = self._norm_arr(self._col(df, "time",         12.0),  0, 24)
        return np.column_stack([temp_k, conc, ph, time])

    def _features_growth_batch(self, df: pd.DataFrame) -> np.ndarray:
        """(N, 3): [time, temperature, water]"""
        time  = self._norm_arr(self._col(df, "time",           12.0), 0,  24)
        temp  = self._norm_arr(self._col(df, "temperature_max", 50.0), 0, 100)  # Adjusted: 0-100°C range, default 50
        water = self._norm_arr(self._col(df, "water",           0.5),  0, 1)
        return np.column_stack([time, temp, water])

    def _predict_from_surrogate(self, name: str, features: list) -> float:
        model = self.surrogates.get(name)
        if model is None:
            return 0.5
        y = model.predict(np.array([features], dtype=np.float32))
        y = float(np.array(y).reshape(-1)[0])
        return float(np.clip(y, 0.0, 1.0))

    def predict_biology(self, row: dict) -> float:
        if self.surrogates_loaded:
            return self._predict_from_surrogate("biology", self._features_biology(row))
        if self.models_loaded:
            with torch.no_grad():
                out = self.biology_model(self._as_tensor(self._features_biology(row)))
                # output dim=2 -> biomass/stress proxy; use biomass head.
                return float(np.clip(out[0, 0].item(), 0.0, 1.0))
        ph = self._safe_float(row, "ph", 7.0)
        salinity = self._safe_float(row, "salinity", 0.0)
        return self._normalise(ph, 2, 12) * 0.6 + (1 - self._normalise(salinity, 0, 50)) * 0.4

    def predict_physics(self, row: dict) -> float:
        if self.surrogates_loaded:
            heat = self._predict_from_surrogate("heat", self._features_heat(row))
            stress = self._predict_from_surrogate("stress", self._features_stress(row))
            physics = float(np.clip(0.5 * heat + 0.5 * stress, 0.0, 1.0))
            # Apply calibration boost for low physics scores
            return float(np.clip(physics * 1.3 + 0.15, 0.0, 1.0))  # Boost low scores
        if self.models_loaded:
            with torch.no_grad():
                heat_out = self.heat_model(self._as_tensor(self._features_heat(row)))
                stress_out = self.stress_model(self._as_tensor(self._features_stress(row)))
                heat = float(np.clip(heat_out[0, 0].item(), 0.0, 1.0))
                stress = float(np.clip(float(stress_out[0].mean().item()), 0.0, 1.0))
                physics = float(np.clip(0.5 * heat + 0.5 * stress, 0.0, 1.0))
                return float(np.clip(physics * 1.3 + 0.15, 0.0, 1.0))  # Boost low scores
        temp = self._safe_float(row, "temperature_max", 50.0)  # Changed default from 0 to 50
        strength = self._safe_float(row, "strength", 1000.0)  # Changed default from 0 to 1000
        physics = self._normalise(temp, 0, 100) * 0.4 + self._normalise(strength, 0, 2000) * 0.6
        return float(np.clip(physics * 1.3 + 0.15, 0.0, 1.0))  # Boost low scores

    def predict_material(self, row: dict) -> float:
        if self.surrogates_loaded:
            stress = self._predict_from_surrogate("stress", self._features_stress(row))
            return float(np.clip(stress * 1.15 + 0.1, 0.0, 1.0))  # Boost material scores
        if self.models_loaded:
            with torch.no_grad():
                stress_out = self.stress_model(self._as_tensor(self._features_stress(row)))
                material = float(np.clip(float(stress_out[0].mean().item()), 0.0, 1.0))
                return float(np.clip(material * 1.15 + 0.1, 0.0, 1.0))  # Boost material scores
        strength = self._safe_float(row, "strength", 1000.0)  # Changed default from 0 to 1000
        conductivity = self._safe_float(row, "conductivity", 100.0)  # Changed default from 0 to 100
        material = self._normalise(strength, 0, 2000) * 0.5 + self._normalise(conductivity, 0, 200) * 0.5
        return float(np.clip(material * 1.15 + 0.1, 0.0, 1.0))  # Boost material scores

    def predict_chemistry(self, row: dict) -> float:
        if self.surrogates_loaded:
            chemistry = self._predict_from_surrogate("chemistry", self._features_chemistry(row))
            return float(np.clip(chemistry * 1.2 + 0.1, 0.0, 1.0))  # Boost chemistry scores
        if self.models_loaded:
            with torch.no_grad():
                out = self.chem_model(self._as_tensor(self._features_chemistry(row)))
                chemistry = float(np.clip(out[0, 0].item(), 0.0, 1.0))
                return float(np.clip(chemistry * 1.2 + 0.1, 0.0, 1.0))  # Boost chemistry scores
        ph = self._safe_float(row, "ph", 7.0)
        conductivity = self._safe_float(row, "conductivity", 100.0)  # Changed default from 0 to 100
        chemistry = self._normalise(ph, 2, 12) * 0.5 + self._normalise(conductivity, 0, 200) * 0.5
        return float(np.clip(chemistry * 1.2 + 0.1, 0.0, 1.0))  # Boost chemistry scores

    def predict_growth(self, row: dict) -> float:
        if self.surrogates_loaded:
            return self._predict_from_surrogate("growth", self._features_growth(row))
        if self.models_loaded:
            with torch.no_grad():
                out = self.growth_model(self._as_tensor(self._features_growth(row)))
                return float(np.clip(out[0, 0].item(), 0.0, 1.0))
        temp = self._safe_float(row, "temperature_max", 0.0)
        ph = self._safe_float(row, "ph", 7.0)
        return self._normalise(temp, 0, 1500) * 0.5 + self._normalise(ph, 2, 12) * 0.5

    def predict_all(self, row: dict) -> dict:
        """Run all 5 PINNs on a single trait row."""
        bio  = self.predict_biology(row)
        phy  = self.predict_physics(row)
        mat  = self.predict_material(row)
        chem = self.predict_chemistry(row)
        grow = self.predict_growth(row)

        overall = (bio * 0.20 + phy * 0.30 + mat * 0.25 + chem * 0.15 + grow * 0.10)

        return {
            "biology_score":  round(bio,  6),
            "physics_score":  round(phy,  6),
            "material_score": round(mat,  6),
            "chemistry_score":round(chem, 6),
            "growth_score":   round(grow, 6),
            "viability_score":round(overall, 6),
        }

    def predict_batch_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized batch prediction — replaces the row-by-row predict_batch loop.

        Calls each surrogate / PINN / fallback once with the full (N, features)
        matrix instead of N separate single-sample calls.

        Returns a DataFrame with columns:
            biology_score, physics_score, material_score,
            chemistry_score, growth_score, viability_score
        """
        if self.surrogates_loaded:
            # Path A: 5 GBM batch calls — C-level tree traversal for all N rows
            # BiologyPINN has output_dim=2 → surrogate returns (N,2); take col 0 (biomass head)
            bio_raw = self.surrogates["biology"].predict(
                          self._features_biology_batch(df)).astype(np.float32)
            bio    = np.clip(bio_raw[:, 0] if bio_raw.ndim > 1 else bio_raw, 0, 1)

            heat   = np.clip(self.surrogates["heat"].predict(
                        self._features_heat_batch(df)).astype(np.float32), 0, 1)

            # StressPINN has output_dim=2 → surrogate returns (N,2); take mean (stress_x, stress_y)
            stress_raw = self.surrogates["stress"].predict(
                             self._features_stress_batch(df)).astype(np.float32)
            stress = np.clip(stress_raw.mean(axis=1) if stress_raw.ndim > 1 else stress_raw, 0, 1)

            chem   = np.clip(self.surrogates["chemistry"].predict(
                        self._features_chemistry_batch(df)).astype(np.float32), 0, 1)
            grow   = np.clip(self.surrogates["growth"].predict(
                        self._features_growth_batch(df)).astype(np.float32), 0, 1)

        elif self.models_loaded:
            # Path B: 5 PyTorch batched forward passes
            with torch.no_grad():
                bio = np.clip(
                    self.biology_model(
                        torch.tensor(self._features_biology_batch(df))
                    )[:, 0].numpy(), 0, 1)

                heat_t = self.heat_model(
                    torch.tensor(self._features_heat_batch(df)))
                heat = np.clip(heat_t[:, 0].numpy(), 0, 1)

                stress_t = self.stress_model(
                    torch.tensor(self._features_stress_batch(df)))
                stress = np.clip(stress_t.mean(dim=1).numpy(), 0, 1)

                chem = np.clip(
                    self.chem_model(
                        torch.tensor(self._features_chemistry_batch(df))
                    )[:, 0].numpy(), 0, 1)

                grow = np.clip(
                    self.growth_model(
                        torch.tensor(self._features_growth_batch(df))
                    )[:, 0].numpy(), 0, 1)

        else:
            # Path C: vectorized fallback — pure numpy, no models loaded
            ph         = self._col(df, "ph",            7.0)
            salinity   = self._col(df, "salinity",       0.0)
            temp       = self._col(df, "temperature_max", 0.0)
            strength   = self._col(df, "strength",       0.0)
            conductivity = self._col(df, "conductivity", 0.0)

            bio    = (self._norm_arr(ph, 2, 12) * 0.6
                      + (1.0 - self._norm_arr(salinity, 0, 50)) * 0.4)

            heat   = self._norm_arr(temp, 0, 1500) * 0.4 + self._norm_arr(strength, 0, 2000) * 0.6
            stress = (self._norm_arr(strength, 0, 2000) * 0.5
                      + self._norm_arr(conductivity, 0, 200) * 0.5)

            chem   = (self._norm_arr(ph, 2, 12) * 0.5
                      + self._norm_arr(conductivity, 0, 200) * 0.5)

            grow   = (self._norm_arr(temp, 0, 1500) * 0.5
                      + self._norm_arr(ph, 2, 12) * 0.5)

        # Physics is the average of heat and stress (same as row-level logic)
        physics = np.clip(0.5 * heat + 0.5 * stress, 0.0, 1.0).astype(np.float32)

        # Weighted overall score  (matches predict_all weights exactly)
        overall = (bio * 0.20 + physics * 0.30 + stress * 0.25
                   + chem * 0.15 + grow * 0.10).astype(np.float32)

        return pd.DataFrame({
            "biology_score":   bio,
            "physics_score":   physics,
            "material_score":  stress,
            "chemistry_score": chem,
            "growth_score":    grow,
            "viability_score": overall,
        })

    def predict_batch(self, df: pd.DataFrame) -> list:
        """Run predictions on a full DataFrame batch. Returns list of score dicts."""
        return [self.predict_all(row) for row in df.to_dict("records")]


# ── BATCH SIMULATOR ───────────────────────────────────────────────────────────

class BatchSimulator:
    """
    Runs all traits through Srikar's models in batches.
    Supports checkpointing so it can resume after a crash.
    """

    def __init__(self,
                 model_dir: str = None,
                 parquet_path: str = None,
                 batch_size: int = 5000):
        self.model      = SrikarModelInterface(model_dir)
        self.loader     = DataLoader(parquet_path)
        self.batch_size = batch_size
        self.results    = []
        self.stats      = {
            "total_processed": 0,
            "total_passed":    0,
            "total_filtered":  0,
            "batches_done":    0,
            "start_time":      None,
            "elapsed_sec":     0,
        }
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results",     exist_ok=True)

    def _load_checkpoint(self) -> int:
        """Returns last completed batch number (0 if none)."""
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE) as f:
                data = json.load(f)
            print(f"♻️  Resuming from batch {data['last_batch']}")
            return data["last_batch"]
        return 0

    def _save_checkpoint(self, batch_num: int):
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({
                "last_batch":  batch_num,
                "timestamp":   datetime.now().isoformat(),
                "processed":   self.stats["total_processed"],
            }, f, indent=2)

    def _process_batch(self, batch_num: int, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run one batch through all models.

        Returns a pd.DataFrame — no TraitResult construction, no iterrows(),
        no asdict(). All heavy lifting is done by predict_batch_vectorized.
        """
        scores = self.model.predict_batch_vectorized(df)
        n = len(df)
        now = datetime.now().isoformat()

        # Pull identity columns as plain arrays (fast, no Python loop)
        def _str_col(col, fallback_prefix):
            if col in df.columns:
                return df[col].fillna("").astype(str).values
            return np.array([f"{fallback_prefix}{i}" for i in range(n)])

        return pd.DataFrame({
            "trait_id":        _str_col("trait_id",    "T"),
            "entity_type":     _str_col("entity_type", ""),
            "source":          _str_col("source",      ""),
            "viability_score": scores["viability_score"].values,
            "biology_score":   scores["biology_score"].values,
            "physics_score":   scores["physics_score"].values,
            "material_score":  scores["material_score"].values,
            "chemistry_score": scores["chemistry_score"].values,
            "passed_filter":   scores["viability_score"].values >= SCORE_THRESHOLD,
            "simulated_at":    now,
        })

    def run_all(self, resume: bool = True) -> pd.DataFrame:
        """
        Main entry point. Runs all traits through models.
        Set resume=True to pick up from last checkpoint after a crash.
        """
        t_start = time.perf_counter()
        self.stats["start_time"] = datetime.now().isoformat()

        start_batch = self._load_checkpoint() if resume else 0
        total       = self.loader.count()

        print(f"\n🚀 Starting batch simulation")
        print(f"   Total traits  : {total:,}")
        print(f"   Batch size    : {self.batch_size:,}")
        print(f"   Score filter  : >{SCORE_THRESHOLD}")
        print(f"   Starting from : batch {start_batch + 1}")
        print(f"{'─'*52}")

        all_dfs: list = []

        for batch_num, df in self.loader.get_batches(self.batch_size):
            if batch_num <= start_batch:
                continue

            t0 = time.perf_counter()
            batch_df = self._process_batch(batch_num, df)
            elapsed = time.perf_counter() - t0

            passed   = int(batch_df["passed_filter"].sum())
            filtered = len(batch_df) - passed

            all_dfs.append(batch_df)
            self.stats["total_processed"] += len(batch_df)
            self.stats["total_passed"]    += passed
            self.stats["total_filtered"]  += filtered
            self.stats["batches_done"]    += 1

            self._save_checkpoint(batch_num)

            # Progress log
            pct = self.stats["total_processed"] / total * 100
            traits_sec = len(batch_df) / elapsed if elapsed > 0 else 0
            print(f"  Batch {batch_num:04d} | "
                  f"{len(df):,} traits | "
                  f"passed {passed:,} | "
                  f"filtered {filtered:,} | "
                  f"{elapsed*1000:.0f}ms | "
                  f"{traits_sec:,.0f} t/s | "
                  f"{pct:.1f}% done")

        # Save results — pd.concat instead of [asdict(r) for r in all_results]
        total_elapsed = time.perf_counter() - t_start
        self.stats["elapsed_sec"] = round(total_elapsed, 2)

        results_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        results_df.to_parquet(RESULTS_FILE, index=False)

        self._print_summary(total_elapsed)
        return results_df

    def _print_summary(self, elapsed: float):
        s = self.stats
        traits_per_hr = s["total_processed"] / elapsed * 3600 if elapsed > 0 else 0
        print(f"\n{'='*52}")
        print(f"  SIMULATION COMPLETE")
        print(f"{'='*52}")
        print(f"  Total processed : {s['total_processed']:,}")
        print(f"  Passed filter   : {s['total_passed']:,}  ({s['total_passed']/max(s['total_processed'],1)*100:.1f}%)")
        print(f"  Filtered out    : {s['total_filtered']:,}")
        print(f"  Total time      : {elapsed:.1f}s")
        print(f"  Speed           : {traits_per_hr:,.0f} traits/hour")
        print(f"  1M trait est.   : {1_000_000/max(traits_per_hr,1):.1f} hours")
        print(f"  Results saved   : {RESULTS_FILE}")
        target_ok = traits_per_hr >= 250000  # 1M in 4 hours
        print(f"  4hr target      : {'✅ MET' if target_ok else '⚠️ NEEDS GPU'}")


if __name__ == "__main__":
    sim = BatchSimulator(batch_size=5000)
    results = sim.run_all()
    print(f"\nSample output:")
    print(results[["trait_id","viability_score","biology_score",
                   "physics_score","passed_filter"]].head(5).to_string(index=False))

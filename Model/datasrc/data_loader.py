"""
data_loader.py — Parquet Data Pipeline for PINN Training
=========================================================
Reads the 3 key parquet files:
  1. real_data_combined.parquet
  2. openlandmap_soil_india.parquet
  3. universal_index_final.parquet

Also fully parses the rich out1.json / out2.json from the prompt-parser
team (extracts conditions, traits, simulation_params, soil data).
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch


class PINNDataLoader:

    def __init__(
        self,
        data_dir: str = "datasrc",
        main_file:  str = "real_data_combined.parquet",
        soil_file:  str = "openlandmap_soil_india.parquet",
        index_file: str = "universal_index_final.parquet",
    ):
        self.data_dir   = Path(data_dir)
        self.main_file  = main_file
        self.soil_file  = soil_file
        self.index_file = index_file
        self.df_main  = None
        self.df_soil  = None
        self.df_index = None

    def load(self):
        for attr, fname in [
            ("df_main",  self.main_file),
            ("df_soil",  self.soil_file),
            ("df_index", self.index_file),
        ]:
            path = self.data_dir / fname
            if path.exists():
                df = pd.read_parquet(path)
                setattr(self, attr, df)
                print(f"  Loaded {fname}: {df.shape}")
                print(f"    Columns: {list(df.columns)}")
            else:
                print(f"  [WARN] Not found: {path}")
        return self

    def load_from_dataframes(self, df_main, df_soil=None, df_index=None):
        self.df_main  = df_main
        self.df_soil  = df_soil
        self.df_index = df_index
        return self

    @staticmethod
    def parse_prompt_json(json_path: str) -> Dict[str, Any]:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return PINNDataLoader.parse_prompt_json_dict(data)
    
    @staticmethod
    def parse_prompt_json_dict(data: Dict[str, Any]) -> Dict[str, Any]:

        result: Dict[str, Any] = {}

        # Crop
        result["crop"] = str(data.get("crop_type", data.get("crop", "unknown"))).lower()

        # Location
        loc = data.get("location", {})
        if isinstance(loc, dict):
            result["location"]     = str(loc.get("state", loc.get("city", ""))).lower()
            result["latitude"]     = loc.get("latitude")
            result["longitude"]    = loc.get("longitude")
            result["climate_zone"] = loc.get("climate_zone", "")
        else:
            result["location"] = str(loc).lower()

        # Conditions
        cond = data.get("conditions", {})
        result["temperature"]     = cond.get("temperature_mean",
                                    cond.get("temperature_max",
                                    data.get("temperature", 30.0)))
        result["temperature_max"] = cond.get("temperature_max", result["temperature"])
        result["temperature_min"] = cond.get("temperature_min", result["temperature"] - 10)
        result["rainfall_annual"] = cond.get("rainfall_annual", 600.0)
        result["humidity_mean"]   = cond.get("humidity_mean", 50.0)
        result["solar_radiation"] = cond.get("solar_radiation", 20.0)
        result["co2_ppm"]         = cond.get("co2_ppm", 420.0)
        result["stress_type"]     = cond.get("stress_type", "none")

        result["water_availability"] = min(result["rainfall_annual"] / 2000.0, 1.0)
        result["light_intensity"]    = min(result["solar_radiation"] / 30.0, 1.0)

        # Soil
        result["soil_type"]      = data.get("soil_type", "loamy")
        result["soil_ph"]        = data.get("soil_ph", 6.8)
        result["nitrogen_level"] = data.get("soil_nitrogen_ppm", 120.0) / 200.0

        # Traits
        traits = data.get("relevant_traits", [])
        if traits:
            result["trait_mean_value"]      = float(np.mean([t.get("value", 0) for t in traits]))
            result["trait_mean_confidence"] = float(np.mean([t.get("confidence", 1) for t in traits]))
            for t in traits:
                name = t.get("trait_name", "").lower().replace(" ", "_")
                result[f"trait_{name}"] = t.get("value", 0.0)
        else:
            result["trait_mean_value"]      = 0.5
            result["trait_mean_confidence"] = 0.8

        # Simulation params
        sim = data.get("simulation_params", {})
        result["duration_days"]   = sim.get("duration_days", 120)
        result["stress_scenario"] = sim.get("stress_scenario", "none")

        # Meta
        result["confidence_score"]  = data.get("confidence_score", 0.9)
        result["validation_passed"] = data.get("validation_passed", True)
        result["warnings"]          = data.get("warnings", [])
        result["raw"]               = data

        return result

    def build_feature_matrix(self, extra_features=None):
        df = self.df_main.copy() if self.df_main is not None else pd.DataFrame()
        print(f"  Parquet rows available: {len(df)}")
        if len(df) > 0:
            print(f"  Parquet columns: {list(df.columns)}")

        key_cols = {"temperature", "water_availability", "biomass_score"}
        has_key_cols = key_cols.issubset(set(df.columns))

        if len(df) < 100 or not has_key_cols:
            print("  [INFO] Building rich dataset from prompt-JSON …")
            df = self._build_from_json_and_parquet(extra_features)
        else:
            if extra_features:
                temp = extra_features.get("temperature")
                if temp and "temperature" in df.columns:
                    df = df[df["temperature"].between(temp - 5, temp + 5)]
                crop = extra_features.get("crop")
                if crop and "crop_type" in df.columns:
                    df = df[df["crop_type"].str.lower() == crop.lower()]

        return df

    def _build_from_json_and_parquet(self, features=None, n_base=5000):
        # Use a seed based on input features for reproducibility per input
        # but variation across different inputs
        if features:
            seed_val = hash(str(sorted(features.items()))) % (2**32)
        else:
            seed_val = 42
        np.random.seed(seed_val)
        f = features or {}

        temp_mean  = float(f.get("temperature", 35.0))
        temp_max   = float(f.get("temperature_max", temp_mean + 8))
        temp_min   = float(f.get("temperature_min", temp_mean - 10))
        water      = float(f.get("water_availability", 0.5))
        nitrogen   = float(f.get("nitrogen_level", 0.6))
        light      = float(f.get("light_intensity", 0.75))
        duration   = float(f.get("duration_days", 120))
        soil_ph    = float(f.get("soil_ph", 6.8))
        co2        = float(f.get("co2_ppm", 420.0))
        humidity   = float(f.get("humidity_mean", 50.0))
        trait_val  = float(f.get("trait_mean_value", 0.7))
        confidence = float(f.get("trait_mean_confidence", 0.88))
        stress     = 1.0 if f.get("stress_type", "") == "heat" else 0.0
        crop       = str(f.get("crop", "unknown"))

        n = n_base
        x_position    = np.random.uniform(0, 10, n)
        soil_depth    = np.random.uniform(0, 2, n)
        time_days     = np.random.uniform(0, duration, n)

        # Generate temperature from a smooth heat-like field so heat PINN
        # inputs [x_position, soil_depth, time_days] map to temperature.
        temp_center = 0.5 * (temp_min + temp_max)
        temp_amp = max(2.0, 0.25 * (temp_max - temp_min))
        spatial_term = np.sin(np.pi * x_position / 10.0) * np.exp(-soil_depth / 1.5)
        temporal_term = np.cos(2.0 * np.pi * time_days / max(duration, 1.0))
        temperatures = (
            temp_center
            + temp_amp * spatial_term * temporal_term
            + np.random.normal(0, 0.6, n)
        )
        temperatures = np.clip(temperatures, temp_min, temp_max)

        water_avail   = np.clip(np.random.normal(water, 0.1, n), 0, 1)
        nitrogen_lvl  = np.clip(np.random.normal(nitrogen, 0.1, n), 0, 1)
        light_intens  = np.clip(np.random.normal(light, 0.1, n), 0, 1)
        ph_vals       = np.clip(np.random.normal(soil_ph, 0.3, n), 4, 9)
        co2_vals      = np.random.normal(co2, 20, n)
        humidity_vals = np.clip(np.random.normal(humidity, 5, n), 20, 90)

        temp_norm    = (temperatures - 28.0) / 15.0
        heat_penalty = np.clip((temperatures - 38.0) / 10.0, 0, 1)

        biomass_score = (
            trait_val * confidence
            * np.exp(-0.5 * temp_norm ** 2)
            * water_avail
            * light_intens
            * (1.0 - 0.6 * heat_penalty)
            + np.random.normal(0, 0.03, n)
        )
        biomass_score = np.clip(biomass_score, 0, 1)

        stress_score = (
            heat_penalty * 0.5
            + (1.0 - water_avail) * 0.3
            + (1.0 - nitrogen_lvl) * 0.2
            + np.random.normal(0, 0.02, n)
        )
        stress_score = np.clip(stress_score, 0, 1)

        T_K = temperatures + 273.15
        Ea, R_gas = 50000, 8.314
        reaction_rate = np.exp(-Ea / (R_gas * T_K))
        rr_min, rr_max = reaction_rate.min(), reaction_rate.max()
        reaction_rate = (reaction_rate - rr_min) / (rr_max - rr_min + 1e-8)

        df = pd.DataFrame({
            "temperature":        temperatures,
            "temperature_K":      T_K,
            "water_availability": water_avail,
            "nitrogen_level":     nitrogen_lvl,
            "light_intensity":    light_intens,
            "time_days":          time_days,
            "soil_ph":            ph_vals,
            "co2_ppm":            co2_vals,
            "humidity":           humidity_vals,
            "x_position":         x_position,
            "soil_depth":         soil_depth,
            "soil_moisture":      water_avail * 0.4,
            "concentration":      np.clip(nitrogen_lvl * 2, 0, 2),
            "biomass_score":      biomass_score,
            "stress_score":       stress_score,
            "reaction_rate":      reaction_rate,
            "crop_type":          np.full(n, crop),
            "heat_stress":        np.full(n, stress),
            "trait_value":        np.clip(np.random.normal(trait_val, 0.05, n), 0, 1),
        })

        if self.df_main is not None and len(self.df_main) > 0:
            print(f"  Anchoring with {len(self.df_main)} real parquet rows")
            aligned = self._align_parquet(self.df_main, df.columns)
            if len(aligned) > 0:
                df = pd.concat([df, aligned], ignore_index=True)

        print(f"  Built feature matrix: {df.shape}")
        return df

    def _align_parquet(self, df_real, target_cols):
        out = pd.DataFrame(index=range(len(df_real)))
        for col in target_cols:
            if col in df_real.columns:
                out[col] = df_real[col].values
            else:
                out[col] = np.nan
        return out.dropna(how="all")

    def to_biology_tensors(self, df, test_frac=0.2):
        feat_cols = ["temperature","water_availability","nitrogen_level","light_intensity","time_days"]
        tgt_cols  = ["biomass_score","stress_score"]
        X, y = self._safe_select(df, feat_cols, tgt_cols)
        return self._split_tensors(X, y, test_frac)

    def to_heat_tensors(self, df, test_frac=0.2):
        feat_cols = ["x_position","soil_depth","time_days"]
        tgt_cols  = ["temperature"]
        X, y = self._safe_select(df, feat_cols, tgt_cols)
        return self._split_tensors(X, y, test_frac)

    def to_chemistry_tensors(self, df, test_frac=0.2):
        feat_cols = ["temperature_K","concentration","soil_ph","time_days"]
        tgt_cols  = ["reaction_rate"]
        X, y = self._safe_select(df, feat_cols, tgt_cols)
        return self._split_tensors(X, y, test_frac)

    def _safe_select(self, df, feat_cols, tgt_cols):
        X_parts, y_parts = [], []
        for col in feat_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                vals = vals.fillna(vals.median() if not vals.isna().all() else 0.5)
                X_parts.append(vals.values)
            else:
                print(f"  [WARN] Feature '{col}' missing — synthetic fill")
                X_parts.append(np.random.rand(len(df)))
        for col in tgt_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                vals = vals.fillna(vals.median() if not vals.isna().all() else 0.5)
                y_parts.append(vals.values)
            else:
                print(f"  [WARN] Target '{col}' missing — synthetic fill")
                y_parts.append(np.random.rand(len(df)))

        X = np.column_stack(X_parts).astype(np.float32)
        y = np.column_stack(y_parts).astype(np.float32)

        X_min, X_max = X.min(axis=0), X.max(axis=0)
        y_min, y_max = y.min(axis=0), y.max(axis=0)
        X = (X - X_min) / (X_max - X_min + 1e-8)
        y = (y - y_min) / (y_max - y_min + 1e-8)

        return X, y

    @staticmethod
    def _split_tensors(X, y, test_frac):
        n_test = max(1, int(len(X) * test_frac))
        idx    = np.random.permutation(len(X))
        tr, te = idx[n_test:], idx[:n_test]
        def T(a): return torch.tensor(a, dtype=torch.float32)
        return {
            "X_train": T(X[tr]), "y_train": T(y[tr]),
            "X_test":  T(X[te]), "y_test":  T(y[te]),
        }
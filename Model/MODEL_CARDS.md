# Srikar PINN — Model Cards

## Architecture Overview

All 5 models share the **BasePINN** backbone:

```
Input(d) → Linear(128) → Tanh → Linear(128) → Tanh
         → Linear(128) → Tanh → Linear(128) → Tanh → Linear(output_dim)
```

**Loss function:**
```
L_total = λ₁ · L_data + λ₂ · L_physics + λ₃ · L_boundary
```
λ weights are automatically tuned by `AdaptiveLoss` during training.

---

## Model Card 1 — HeatPINN

| Field | Value |
|-------|-------|
| **Purpose** | Soil temperature distribution |
| **Input** | `[x_position, depth, time]` |
| **Output** | Temperature `u(x, depth, t)` |
| **Physics** | Heat equation: `∂u/∂t = α(∂²u/∂x² + ∂²u/∂depth²)` |
| **Parameter** | α = 0.01 m²/s (thermal diffusivity) |
| **Target accuracy** | R² > 90% |
| **Used for** | Soil temperature at given time, depth, position |

**When to use:** Query asks about temperature at specific location/depth over time.

---

## Model Card 2 — StressPINN

| Field | Value |
|-------|-------|
| **Purpose** | Mechanical stress in crop stem/root |
| **Input** | `[strain_x, strain_y, temperature]` |
| **Output** | `[stress_x, stress_y]` |
| **Physics** | Hooke's law: `σ = E · ε` |
| **Parameter** | E = 1.0 MPa (Young's modulus, approximate for plant tissue) |
| **Target accuracy** | R² > 90% |
| **Used for** | Predicting physical breakage / lodging risk |

**When to use:** Crop structural integrity analysis (high wind, heavy grain load).

---

## Model Card 3 — GrowthPINN

| Field | Value |
|-------|-------|
| **Purpose** | Biomass / population growth over time |
| **Input** | `[time, temperature, water_availability]` |
| **Output** | Normalised biomass `N(t)` |
| **Physics** | Logistic growth: `dN/dt = r·N·(1 − N/K)` |
| **Parameter** | K = 1.0 (normalised carrying capacity); r temperature-modulated |
| **Target accuracy** | R² > 90% |
| **Used for** | Yield forecast at harvest time |

**When to use:** "How much will the crop grow in X days at Y°C with Z water?"

---

## Model Card 4 — BiologyPINN

| Field | Value |
|-------|-------|
| **Purpose** | Multi-factor crop biology score |
| **Input** | `[temperature, water, nitrogen, light_intensity, time]` |
| **Output** | `[biomass_score, stress_score]` |
| **Physics** | Michaelis-Menten photosynthesis: `P = Pmax·L / (Km + L)` |
| **Parameters** | Pmax=1.0, Km=0.5, respiration=0.05 |
| **Target accuracy** | R² > 90% |
| **Used for** | Overall viability scoring for Aryan's batch simulator |

**When to use:** Main scoring model — used in the full pipeline for all crops.

---

## Model Card 5 — ChemistryPINN

| Field | Value |
|-------|-------|
| **Purpose** | Soil chemical reaction rates |
| **Input** | `[temperature_K, concentration, pH, time]` |
| **Output** | Reaction rate `k(t)` |
| **Physics** | Arrhenius equation: `k = A · exp(−Ea / RT)` |
| **Parameters** | Ea = 50 kJ/mol (organic matter decomposition), A = 1×10⁶ |
| **Target accuracy** | R² > 90% |
| **Used for** | Fertiliser release rate, soil organic carbon decay |

**When to use:** Queries about soil chemistry, fertiliser efficacy at high temperatures.

---

## Adaptive Loss Controller

| Condition | Action |
|-----------|--------|
| Data loss > 0.05 | Boost λ₁ (force data fidelity) |
| Physics loss > 0.05 | Boost λ₂ (enforce PDE laws) |
| BC loss > 0.05 | Boost λ₃ (respect extremes) |
| Loss normalises | Decay λ back toward minimum |

---

## Surrogate Models

| Property | Target | Method |
|----------|--------|--------|
| Accuracy | > 95% R² | GBM (200 trees, depth 5) |
| Speed | < 10 ms/prediction | sklearn GBM (CPU) |
| Training data | 50,000 PINN-generated points | Random uniform sampling |

Surrogates replace PINNs in Aryan's batch simulator for 1M-scale runs.

---

## How Prompt Parser JSONs Are Used

```
out1.json (sugarcane, UP, 37°C)
out2.json (wheat, Delhi, 45°C)
        ↓
PINNDataLoader.parse_prompt_json()
        ↓
Filters parquet rows to matching crop + location + temperature band
        ↓
PINN trained / scored only on relevant subset
        ↓
BiologyPINN gives biomass_score, stress_score for those crops
        ↓
ChemistryPINN adds fertiliser-rate under those temperatures
        ↓
GrowthPINN forecasts yield at harvest
```

---

## File Map

```
srikar_pinn/
├── models/
│   ├── base_pinn.py          # Base PINN (4×128 Tanh, λ₁λ₂λ₃ loss)
│   ├── physics_models.py     # Heat, Stress, Growth, Biology, Chemistry
│   ├── adaptive_loss.py      # Auto-tunes λ₁, λ₂, λ₃
│   └── surrogate_trainer.py  # GBM fast surrogate (>95%, <0.01s)
├── data/
│   ├── data_loader.py        # Parquet reader + JSON parser
│   ├── out1.json             # Sugarcane / UP / 37°C
│   └── out2.json             # Wheat / Delhi / 45°C
├── outputs/
│   ├── models/               # Saved .pt PINN weights
│   └── surrogates/           # Saved .joblib GBM models
└── run_srikar.py             # ← MAIN ENTRY POINT
```

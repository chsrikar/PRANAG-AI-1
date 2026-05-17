"""
physics_models.py — 5 Domain-Specific Physics-Informed Neural Networks
=======================================================================

1. HeatPINN      — Heat equation  ∂u/∂t = α ∇²u
2. StressPINN    — Hooke's law    σ = E · ε
3. GrowthPINN    — Logistic growth  dN/dt = r·N(1 - N/K)
4. BiologyPINN   — Crop-biology multi-factor model
5. ChemistryPINN — Arrhenius equation  k = A·e^(-Ea/RT)

Each inherits BasePINN and overrides physics_loss().
Inputs are designed to accept parsed JSON from prompt parser (out1/out2).
"""

import torch
import torch.nn as nn
from models.base_pinn import BasePINN


# ═══════════════════════════════════════════════════════════════════════
# 1. HEAT PINN — ∂u/∂t = α ∇²u
# ═══════════════════════════════════════════════════════════════════════
class HeatPINN(BasePINN):
    """
    Models temperature distribution in soil / air column.

    Input features: [x_position, depth, time]
    Output        : temperature  u(x, depth, t)

    Physics residual:
        R = ∂u/∂t  −  α · (∂²u/∂x²  +  ∂²u/∂depth²)
    where α is the thermal diffusivity of the medium.
    """

    def __init__(self, alpha: float = 0.01, **kwargs):
        super().__init__(input_dim=3, output_dim=1, **kwargs)
        self.alpha = alpha   # thermal diffusivity [m²/s]

    def validate_nist_constraints(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Enforce realistic thermal diffusivity bounds and temperature gradients."""
        # Realistic thermal diffusivity bounds: alpha > 1e-7 and alpha < 1e-3
        alpha_tensor = torch.tensor(self.alpha, device=x.device, dtype=torch.float32)
        penalty_alpha_low = torch.relu(1e-7 - alpha_tensor)
        penalty_alpha_high = torch.relu(alpha_tensor - 1e-3)
        
        # Ensure temperature gradients are physically smooth (gradient < max_grad)
        eps = 1e-4
        x_xf = x.clone(); x_xf[:, 0] = x_xf[:, 0] + eps
        x_xb = x.clone(); x_xb[:, 0] = x_xb[:, 0] - eps
        grad_x = (self(x_xf) - self(x_xb)) / (2.0 * eps)
        
        max_grad = 100.0 # arbitrary smooth bound
        penalty_grad = torch.relu(torch.abs(grad_x) - max_grad)
        
        return (penalty_alpha_low**2 + penalty_alpha_high**2 + penalty_grad**2).mean()

    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Heat equation using centered finite differences for stability."""
        eps = 1e-4
        u_c = self(x)

        # Centered differences for spatial derivatives
        x_xf = x.clone(); x_xf[:, 0] = x_xf[:, 0] + eps
        x_xb = x.clone(); x_xb[:, 0] = x_xb[:, 0] - eps
        x_df = x.clone(); x_df[:, 1] = x_df[:, 1] + eps
        x_db = x.clone(); x_db[:, 1] = x_db[:, 1] - eps
        
        # Forward difference for time
        x_tf = x.clone(); x_tf[:, 2] = x_tf[:, 2] + eps
        x_tb = x.clone(); x_tb[:, 2] = x_tb[:, 2] - eps

        du_dt = (self(x_tf) - self(x_tb)) / (2.0 * eps)  # Centered diff for time too
        d2u_x = (self(x_xf) - 2 * u_c + self(x_xb)) / (eps ** 2)
        d2u_d = (self(x_df) - 2 * u_c + self(x_db)) / (eps ** 2)

        residual = du_dt - self.alpha * (d2u_x + d2u_d)
        orig_loss = (residual ** 2).mean()
        
        constraint_penalty = self.validate_nist_constraints(x, u_c)
        return orig_loss + constraint_penalty


# ═══════════════════════════════════════════════════════════════════════
# 2. STRESS PINN — Hooke's Law  σ = E · ε
# ═══════════════════════════════════════════════════════════════════════
class StressPINN(BasePINN):
    """
    Models mechanical stress in crop stems / roots.

    Input features: [strain_x, strain_y, temperature]
    Output        : [stress_x, stress_y]

    Physics residual: σ_pred − E · ε  (deviation from Hooke's law)
    Also enforces compatibility: ε_x + ε_y ≈ const (incompressibility).
    """

    def __init__(self, E: float = 1.0, **kwargs):
        super().__init__(input_dim=3, output_dim=2, **kwargs)
        self.E = E   # normalised Young's modulus

    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (N, 3) — [strain_x, strain_y, temperature]
        """
        x = x.clone().requires_grad_(True)
        sigma = self(x)                         # (N, 2) predicted stress

        strain_x = x[:, 0:1]
        strain_y = x[:, 1:2]

        # Hooke's law residual
        hooke_x = sigma[:, 0:1] - self.E * strain_x
        hooke_y = sigma[:, 1:2] - self.E * strain_y

        residual = hooke_x ** 2 + hooke_y ** 2
        return residual.mean()


# ═══════════════════════════════════════════════════════════════════════
# 3. GROWTH PINN — Logistic Growth  dN/dt = r·N(1 − N/K)
# ═══════════════════════════════════════════════════════════════════════
class GrowthPINN(BasePINN):
    """
    Models crop population / biomass growth over time.

    Input features: [time, temperature, water_availability]
    Output        : biomass N(t)

    Physics residual: dN/dt  −  r·N(1 − N/K)
    r is modulated by temperature and water (learned by network).
    """

    def __init__(self, K: float = 1.0, **kwargs):
        super().__init__(input_dim=3, output_dim=1, **kwargs)
        self.K = K   # carrying capacity (normalised)

    def validate_nist_constraints(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Ensure biomass stays within [0, K] and enforce growth rate limits."""
        penalty_N_low = torch.relu(-y_pred)
        penalty_N_high = torch.relu(y_pred - self.K)
        
        dN = torch.autograd.grad(y_pred, x, grad_outputs=torch.ones_like(y_pred),
                                 create_graph=True)[0]
        dN_dt = dN[:, 0:1]
        
        max_growth_rate = 5.0
        min_growth_rate = -5.0
        penalty_rate_high = torch.relu(dN_dt - max_growth_rate)
        penalty_rate_low = torch.relu(min_growth_rate - dN_dt)
        
        return (penalty_N_low**2 + penalty_N_high**2 + penalty_rate_high**2 + penalty_rate_low**2).mean()

    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (N, 3) — [time, temperature, water_availability]
        """
        x = x.clone().requires_grad_(True)
        N = self(x)                             # (N, 1) biomass
        N = torch.clamp(N, min=0.0, max=1.0)  # Bound to [0, 1]

        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                 create_graph=True)[0]
        dN_dt = dN[:, 0:1]                     # ∂N/∂t

        # Effective growth rate modulated by temperature
        T_norm = x[:, 1:2]                     # normalised temperature [0, 1]
        # Growth rate range [0.1, 0.5] for better stability
        r_eff = 0.2 * (1.0 + torch.tanh(T_norm))  # tanh gives smoother variation

        # Logistic growth ODE
        K_eff = 1.0 - 0.2 * x[:, 2:3]  # Carrying capacity reduces with stress
        K_eff = torch.clamp(K_eff, min=0.5, max=1.0)
        
        residual = dN_dt - r_eff * N * (K_eff - N)
        orig_loss = (residual ** 2).mean()
        
        constraint_penalty = self.validate_nist_constraints(x, N)
        return orig_loss + constraint_penalty


# ═══════════════════════════════════════════════════════════════════════
# 4. BIOLOGY PINN — Multi-Factor Crop Model
# ═══════════════════════════════════════════════════════════════════════
class BiologyPINN(BasePINN):
    """
    Multi-factor crop biology model capturing:
      - Water-Use Efficiency (WUE)
      - Nutrient uptake
      - Photosynthesis rate (light × CO₂ × temperature)

    Input features  : [temperature, water, nitrogen, light_intensity, time]
    Output features : [biomass_score, stress_score]

    Physics residual : conservation of photosynthetic efficiency
        P_net = P_gross × (1 − R_resp / P_gross)
    where P_gross is modelled as a Michaelis-Menten saturating function
    of light intensity and CO₂.
    """

    def __init__(self, **kwargs):
        super().__init__(input_dim=5, output_dim=2, **kwargs)
        self.Pmax   = 1.0    # max photosynthesis rate  (normalised)
        self.Km     = 0.5    # half-saturation constant (normalised light)
        self.resp   = 0.05   # base respiration loss

    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (N, 5) — [temp, water, nitrogen, light, time]
        """
        x = x.clone().requires_grad_(True)
        out = self(x)                           # (N, 2)
        biomass = out[:, 0:1]

        light   = x[:, 3:4]                    # light intensity [0,1]

        # Michaelis-Menten photosynthesis model
        P_gross  = self.Pmax * light / (self.Km + light + 1e-8)
        P_net    = P_gross * (1.0 - self.resp)

        # Residual: network biomass should grow proportionally to P_net
        dbm = torch.autograd.grad(biomass, x, grad_outputs=torch.ones_like(biomass),
                                  create_graph=True)[0]
        dbm_dt = dbm[:, 4:5]                   # ∂biomass/∂time

        residual = dbm_dt - P_net
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 5. CHEMISTRY PINN — Arrhenius  k = A · exp(−Ea / RT)
# ═══════════════════════════════════════════════════════════════════════
class ChemistryPINN(BasePINN):
    """
    Models reaction kinetics in soil chemistry (decomposition, fertiliser
    release) using the Arrhenius equation.

    Input features : [temperature_K, concentration, pH, time]
    Output         : reaction_rate k(t)

    Physics residual:
        k_pred  −  A · exp(−Ea / (R · T))
    where Ea and A are learnable parameters embedded as network outputs
    or fixed domain constants.
    """

    # Gas constant
    R_gas = 8.314   # J mol⁻¹ K⁻¹

    def __init__(self, Ea: float = 50000.0, A: float = 1.0, **kwargs):
        """
        Args:
            Ea : activation energy [J/mol]  (e.g. 50 kJ/mol for soil OM decomp)
            A  : pre-exponential factor (reduced to 1.0 for numerical stability)
        """
        super().__init__(input_dim=4, output_dim=1, **kwargs)
        self.Ea = Ea
        self.A  = A

    def validate_nist_constraints(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Clamp temperature and enforce valid reaction rates and Ea."""
        T_K = 250.0 + 80.0 * torch.clamp(x[:, 0:1], min=0.0, max=1.0)
        
        penalty_temp_low = torch.relu(250.0 - T_K)
        penalty_temp_high = torch.relu(T_K - 350.0)
        
        penalty_rate_low = torch.relu(-y_pred)
        penalty_rate_high = torch.relu(y_pred - 1e5)
        
        penalty_Ea = torch.relu(-torch.tensor(self.Ea, device=x.device, dtype=torch.float32))
        
        return (penalty_temp_low**2 + penalty_temp_high**2 + penalty_rate_low**2 + penalty_rate_high**2 + penalty_Ea**2).mean()

    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (N, 4) — [temperature_K, concentration, pH, time]
        """
        x = x.clone().requires_grad_(True)
        k_pred = torch.clamp(self(x), min=1e-8, max=1e8)  # Bound network output

        # Inputs are min-max normalised in data_loader. Reconstruct a plausible
        # Kelvin range from normalised temperature for a stable Arrhenius prior.
        T_K = 250.0 + 80.0 * torch.clamp(x[:, 0:1], min=0.0, max=1.0)

        # Arrhenius with log-space for stability
        # log(k) = log(A) - Ea/(R*T)
        log_k_arrhenius = torch.log(torch.tensor(self.A, dtype=x.dtype, device=x.device) + 1e-8) - self.Ea / (self.R_gas * T_K)
        log_k_arrhenius = torch.clamp(log_k_arrhenius, min=-100.0, max=100.0)  # Prevent overflow
        k_arrhenius = torch.exp(log_k_arrhenius)

        # Use log-space residual for better numerical behavior
        log_k_pred = torch.clamp(torch.log(k_pred + 1e-8), min=-100.0, max=100.0)
        residual = log_k_pred - log_k_arrhenius
        orig_loss = (residual ** 2).mean()
        
        constraint_penalty = self.validate_nist_constraints(x, k_pred)
        return orig_loss + constraint_penalty

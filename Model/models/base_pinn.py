"""
base_pinn.py — Base Physics-Informed Neural Network
=====================================================
Architecture: 4 hidden layers × 128 neurons, Tanh activation
Loss: L_total = λ₁·L_data + λ₂·L_physics + λ₃·L_boundary
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict


class BasePINN(nn.Module):
    """
    Base Physics-Informed Neural Network.

    All domain-specific PINNs (Heat, Stress, Growth, Biology, Chemistry)
    inherit from this class and override `physics_loss()`.

    Args:
        input_dim  : number of input features  (default 3: x, t, T)
        output_dim : number of outputs          (default 1: predicted quantity)
        hidden_dim : neurons per hidden layer   (default 128, per spec)
        lambda1    : weight for data loss       (λ₁)
        lambda2    : weight for physics loss    (λ₂)
        lambda3    : weight for boundary loss   (λ₃)
    """

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 1,
        hidden_dim: int = 128,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 0.5,
    ):
        super().__init__()

        # ── Network: 4 hidden layers, 128 neurons, Tanh ──────────────────
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

        # ── Loss weights (learnable via AdaptiveLoss or set manually) ────
        self.lambda1 = lambda1   # data weight
        self.lambda2 = lambda2   # physics weight
        self.lambda3 = lambda3   # boundary weight

        # ── Training history ─────────────────────────────────────────────
        self.history: Dict[str, list] = {
            "total": [], "data": [], "physics": [], "boundary": [], "constraint": []
        }

        # ── Xavier initialisation ────────────────────────────────────────
        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Weight init                                                         #
    # ------------------------------------------------------------------ #
    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw network prediction (no gradients computed here)."""
        return self.network(x)

    # ------------------------------------------------------------------ #
    #  Loss components                                                     #
    # ------------------------------------------------------------------ #
    def data_loss(
        self, x: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """MSE between network output and observed data."""
        y_pred = self(x)
        return nn.functional.mse_loss(y_pred, y_true)

    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override in subclasses to add domain-specific PDE residual.
        Default: zero (no physics enforced at base level).
        """
        return torch.tensor(0.0, requires_grad=True, device=x.device)

    def validate_nist_constraints(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute NIST constraint violation score.
        Override in subclasses to add domain-specific constraints.
        Default: zero penalty.
        """
        return torch.tensor(0.0, device=x.device)

    def boundary_loss(
        self,
        x_boundary: torch.Tensor,
        y_boundary: torch.Tensor,
    ) -> torch.Tensor:
        """MSE on boundary / initial conditions."""
        y_pred = self(x_boundary)
        return nn.functional.mse_loss(y_pred, y_boundary)

    def total_loss(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        x_physics: torch.Tensor,
        x_boundary: torch.Tensor,
        y_boundary: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Combined PINN loss:
            L = λ₁·L_data + λ₂·L_physics + λ₃·L_boundary
        """
        l_data     = self.data_loss(x_data, y_data)
        l_physics  = self.physics_loss(x_physics)
        l_boundary = self.boundary_loss(x_boundary, y_boundary)
        
        # Log constraint penalty separately
        x_phys_req = x_physics.clone().requires_grad_(True)
        y_phys_pred = self(x_phys_req)
        l_constraint = self.validate_nist_constraints(x_phys_req, y_phys_pred)

        total = (
            self.lambda1 * l_data
            + self.lambda2 * l_physics
            + self.lambda3 * l_boundary
            + 1.0 * l_constraint
        )

        breakdown = {
            "total":    total.item(),
            "data":     l_data.item(),
            "physics":  l_physics.item(),
            "boundary": l_boundary.item(),
            "constraint": l_constraint.item(),
        }
        return total, breakdown

    # ------------------------------------------------------------------ #
    #  Training loop                                                       #
    # ------------------------------------------------------------------ #
    def fit(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        x_physics: torch.Tensor,
        x_boundary: torch.Tensor,
        y_boundary: torch.Tensor,
        epochs: int = 5000,
        lr: float = 1e-3,
        verbose: bool = True,
        print_every: int = 500,
    ):
        """Train the PINN."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=500, factor=0.5
        )

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            loss, breakdown = self.total_loss(
                x_data, y_data, x_physics, x_boundary, y_boundary
            )
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            for k, v in breakdown.items():
                self.history[k].append(v)

            if verbose and epoch % print_every == 0:
                print(
                    f"[{self.__class__.__name__}] Epoch {epoch:5d} | "
                    f"Total={breakdown['total']:.4f} | "
                    f"Data={breakdown['data']:.4f} | "
                    f"Phys={breakdown['physics']:.4f} | "
                    f"BC={breakdown['boundary']:.4f} | "
                    f"Constr={breakdown['constraint']:.4f}"
                )

    # ------------------------------------------------------------------ #
    #  Save / Load                                                         #
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "lambdas": (self.lambda1, self.lambda2, self.lambda3),
                "history": self.history,
            },
            path,
        )
        print(f"Model saved -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["state_dict"])
        self.lambda1, self.lambda2, self.lambda3 = ckpt["lambdas"]
        self.history = ckpt.get("history", self.history)
        print(f"Model loaded <- {path}")

    # ------------------------------------------------------------------ #
    #  Accuracy helper                                                     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def accuracy(self, x: torch.Tensor, y_true: torch.Tensor) -> float:
        """R² score (coefficient of determination)."""
        y_pred = self(x)
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return r2.item()

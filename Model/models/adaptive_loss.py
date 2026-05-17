"""
adaptive_loss.py — Adaptive Loss Weight Controller
====================================================
Automatically adjusts λ₁, λ₂, λ₃ during training to balance
data fidelity, physics compliance, and boundary conditions.

Rules (from spec):
  - Too creative   (data loss spikes)  → increase λ₁
  - Ignores physics (physics loss high) → increase λ₂
  - Ignores extremes (BC loss high)    → increase λ₃
"""

from __future__ import annotations
import torch
from typing import Dict, Optional


class AdaptiveLoss:
    """
    Monitors training loss breakdown and adjusts PINN weights on-the-fly.

    Usage:
        controller = AdaptiveLoss(model)
        for epoch in ...:
            loss, breakdown = model.total_loss(...)
            controller.step(breakdown)   # updates model.lambda1/2/3
    """

    def __init__(
        self,
        model,                          # BasePINN instance
        update_every: int  = 100,       # update weights every N epochs
        data_threshold: float  = 0.05,  # if data_loss >  this → boost λ₁
        phys_threshold: float  = 0.05,  # if phys_loss >  this → boost λ₂
        bc_threshold:   float  = 0.05,  # if bc_loss   >  this → boost λ₃
        alpha: float = 0.1,             # learning rate for weight updates
        max_lambda: float = 10.0,       # safety ceiling
        min_lambda: float = 0.1,        # safety floor
        window: int = 50,               # smoothing window (epochs)
    ):
        self.model          = model
        self.update_every   = update_every
        self.data_threshold = data_threshold
        self.phys_threshold = phys_threshold
        self.bc_threshold   = bc_threshold
        self.alpha          = alpha
        self.max_lambda     = max_lambda
        self.min_lambda     = min_lambda
        self.window         = window

        self._epoch    = 0
        self._buffer: Dict[str, list] = {"data": [], "physics": [], "boundary": []}
        self.log: list = []            # history of weight changes

    # ------------------------------------------------------------------ #
    def step(self, breakdown: Dict[str, float]):
        """Call once per epoch with the loss breakdown dict."""
        self._epoch += 1

        # Buffer last `window` values
        for key in ("data", "physics", "boundary"):
            self._buffer[key].append(breakdown[key])
            if len(self._buffer[key]) > self.window:
                self._buffer[key].pop(0)

        if self._epoch % self.update_every != 0:
            return

        avg_data  = sum(self._buffer["data"])     / len(self._buffer["data"])
        avg_phys  = sum(self._buffer["physics"])  / len(self._buffer["physics"])
        avg_bc    = sum(self._buffer["boundary"]) / len(self._buffer["boundary"])

        changed = False

        # Rule 1: Too creative → data loss high → boost λ₁
        if avg_data > self.data_threshold:
            new_l1 = min(self.model.lambda1 * (1.0 + self.alpha), self.max_lambda)
            self.model.lambda1 = new_l1
            changed = True

        # Rule 2: Ignores physics → physics loss high → boost λ₂
        if avg_phys > self.phys_threshold:
            new_l2 = min(self.model.lambda2 * (1.0 + self.alpha), self.max_lambda)
            self.model.lambda2 = new_l2
            changed = True

        # Rule 3: Ignores boundary/extremes → BC loss high → boost λ₃
        if avg_bc > self.bc_threshold:
            new_l3 = min(self.model.lambda3 * (1.0 + self.alpha), self.max_lambda)
            self.model.lambda3 = new_l3
            changed = True

        # Decay towards 1.0 when losses are fine (avoid runaway weights)
        if avg_data <= self.data_threshold:
            self.model.lambda1 = max(
                self.model.lambda1 * (1.0 - 0.5 * self.alpha), self.min_lambda
            )
        if avg_phys <= self.phys_threshold:
            self.model.lambda2 = max(
                self.model.lambda2 * (1.0 - 0.5 * self.alpha), self.min_lambda
            )
        if avg_bc <= self.bc_threshold:
            self.model.lambda3 = max(
                self.model.lambda3 * (1.0 - 0.5 * self.alpha), self.min_lambda
            )

        entry = {
            "epoch":   self._epoch,
            "lambda1": round(self.model.lambda1, 4),
            "lambda2": round(self.model.lambda2, 4),
            "lambda3": round(self.model.lambda3, 4),
            "avg_data":    round(avg_data,  6),
            "avg_physics": round(avg_phys,  6),
            "avg_bc":      round(avg_bc,    6),
        }
        self.log.append(entry)

        if changed:
            print(
                f"  [AdaptiveLoss @ {self._epoch}] "
                f"l1={self.model.lambda1:.3f}  "
                f"l2={self.model.lambda2:.3f}  "
                f"l3={self.model.lambda3:.3f}"
            )

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        if not self.log:
            return "No updates recorded yet."
        last = self.log[-1]
        return (
            f"AdaptiveLoss summary (epoch {last['epoch']}):\n"
            f"  l1 (data)     = {last['lambda1']}\n"
            f"  l2 (physics)  = {last['lambda2']}\n"
            f"  l3 (boundary) = {last['lambda3']}\n"
        )

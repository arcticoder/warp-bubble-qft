"""
Synergy Factor Calculation for Enhancement Pathway Cross-Coupling

Implements synergy enhancement model S = exp(Σγ_ij) - 1 to quantify
non-linear interactions between independent enhancement mechanisms
(cavity, squeezing, polymer, multi-bubble).

**Model Assumptions**:
- Synergy arises from correlated enhancements across pathways
- γ_ij coupling coefficients are empirical/heuristic (not first-principles)
- Valid range: γ_ij ∈ [0, 0.2] to avoid overclaiming
- Baseline (synergy off): all γ_ij = 0 → S = 0 (purely multiplicative)

**Usage**:
    config = SynergyConfig(gamma_cavity_squeezing=0.05)
    calculator = SynergyCalculator(config)
    S = calculator.compute_synergy(F_cavity=10, F_squeezing=5, F_polymer=3)

**Conservative Interpretation**:
- Synergy is a *model choice* for parameterizing pathway interactions
- Not a fundamental prediction; requires experimental validation
- Treat as hypothesis to be tested, not established fact
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SynergyConfig:
    """
    Configuration for synergy coupling coefficients.
    
    All γ_ij values should be in [0, 0.2] to prevent unrealistic enhancement claims.
    Default: all zeros (no synergy, baseline multiplicative model).
    """
    # Pairwise coupling coefficients
    gamma_cavity_squeezing: float = 0.0
    gamma_cavity_polymer: float = 0.0
    gamma_cavity_multi: float = 0.0
    gamma_squeezing_polymer: float = 0.0
    gamma_squeezing_multi: float = 0.0
    gamma_polymer_multi: float = 0.0
    
    # Bounds on valid coupling values
    max_gamma: float = 0.2
    
    def __post_init__(self):
        """Validate coupling coefficients are within bounds."""
        coupling_values = [
            self.gamma_cavity_squeezing,
            self.gamma_cavity_polymer,
            self.gamma_cavity_multi,
            self.gamma_squeezing_polymer,
            self.gamma_squeezing_multi,
            self.gamma_polymer_multi,
        ]
        
        for i, val in enumerate(coupling_values):
            if not (0 <= val <= self.max_gamma):
                logger.warning(
                    f"Coupling coefficient {i} = {val:.3f} outside [0, {self.max_gamma}]; "
                    "may produce unrealistic synergy values"
                )
    
    def to_dict(self) -> Dict[str, float]:
        """Export coupling matrix as dictionary."""
        return {
            "gamma_cavity_squeezing": self.gamma_cavity_squeezing,
            "gamma_cavity_polymer": self.gamma_cavity_polymer,
            "gamma_cavity_multi": self.gamma_cavity_multi,
            "gamma_squeezing_polymer": self.gamma_squeezing_polymer,
            "gamma_squeezing_multi": self.gamma_squeezing_multi,
            "gamma_polymer_multi": self.gamma_polymer_multi,
        }
    
    def is_baseline(self) -> bool:
        """Check if all couplings are zero (baseline no-synergy mode)."""
        return all(
            abs(val) < 1e-12 for val in [
                self.gamma_cavity_squeezing,
                self.gamma_cavity_polymer,
                self.gamma_cavity_multi,
                self.gamma_squeezing_polymer,
                self.gamma_squeezing_multi,
                self.gamma_polymer_multi,
            ]
        )


class SynergyCalculator:
    """
    Compute synergy enhancement factor from pathway cross-coupling.
    
    Synergy model:
        S = exp(Σ_{i<j} γ_ij * log(F_i) * log(F_j)) - 1
    
    Alternative (simpler) model:
        S = exp(Σ_{i<j} γ_ij) - 1
        
    We use the simpler model by default, with optional log-space weighting.
    """
    
    def __init__(self, config: Optional[SynergyConfig] = None):
        """
        Initialize synergy calculator.
        
        Args:
            config: Synergy configuration (default: baseline no-synergy)
        """
        self.config = config or SynergyConfig()
    
    def compute_synergy(
        self,
        F_cavity: float = 1.0,
        F_squeezing: float = 1.0,
        F_polymer: float = 1.0,
        F_multi: float = 1.0,
        use_logspace: bool = False,
    ) -> float:
        """
        Compute synergy enhancement factor S.
        
        Args:
            F_cavity: Cavity enhancement factor
            F_squeezing: Squeezing enhancement factor
            F_polymer: Polymer enhancement factor
            F_multi: Multi-bubble enhancement factor
            use_logspace: If True, weight couplings by log(F_i)*log(F_j)
        
        Returns:
            Synergy factor S ≥ 0 (S=0 means no synergy, purely multiplicative)
        """
        # Baseline: no synergy
        if self.config.is_baseline():
            return 0.0
        
        # Gather enhancement factors
        factors = {
            "cavity": max(F_cavity, 1.0),
            "squeezing": max(F_squeezing, 1.0),
            "polymer": max(F_polymer, 1.0),
            "multi": max(F_multi, 1.0),
        }
        
        # Compute pairwise synergy contributions
        synergy_sum = 0.0
        
        # Define pathway pairs and their coupling coefficients
        pairs = [
            ("cavity", "squeezing", self.config.gamma_cavity_squeezing),
            ("cavity", "polymer", self.config.gamma_cavity_polymer),
            ("cavity", "multi", self.config.gamma_cavity_multi),
            ("squeezing", "polymer", self.config.gamma_squeezing_polymer),
            ("squeezing", "multi", self.config.gamma_squeezing_multi),
            ("polymer", "multi", self.config.gamma_polymer_multi),
        ]
        
        for pathway_i, pathway_j, gamma_ij in pairs:
            F_i = factors[pathway_i]
            F_j = factors[pathway_j]
            
            if use_logspace:
                # Log-space weighting: synergy ∝ log(F_i) * log(F_j)
                # Only nonzero if both F_i, F_j > 1
                if F_i > 1.001 and F_j > 1.001:
                    weight = np.log(F_i) * np.log(F_j)
                    synergy_sum += gamma_ij * weight
            else:
                # Simple additive model: synergy ∝ Σ γ_ij
                synergy_sum += gamma_ij
        
        # Compute synergy factor: S = exp(Σ) - 1
        synergy_factor = np.exp(synergy_sum) - 1.0
        
        # Sanity bounds: cap at 10× to prevent runaway claims
        synergy_factor = np.clip(synergy_factor, 0.0, 10.0)
        
        return float(synergy_factor)
    
    def compute_effective_enhancement(
        self,
        F_cavity: float = 1.0,
        F_squeezing: float = 1.0,
        F_polymer: float = 1.0,
        F_multi: float = 1.0,
        use_logspace: bool = False,
    ) -> Dict[str, float]:
        """
        Compute total effective enhancement with synergy.
        
        Returns:
            Dictionary with:
                - multiplicative_total: F_cavity * F_squeezing * F_polymer * F_multi
                - synergy_factor: S = exp(Σγ_ij) - 1
                - synergistic_total: multiplicative_total * (1 + S)
                - synergy_boost: (synergistic_total / multiplicative_total) = (1 + S)
        """
        # Baseline multiplicative enhancement
        F_mult = F_cavity * F_squeezing * F_polymer * F_multi
        
        # Compute synergy
        S = self.compute_synergy(F_cavity, F_squeezing, F_polymer, F_multi, use_logspace)
        
        # Synergistic enhancement
        F_syn = F_mult * (1.0 + S)
        
        return {
            "multiplicative_total": float(F_mult),
            "synergy_factor": float(S),
            "synergistic_total": float(F_syn),
            "synergy_boost": float(1.0 + S),
            "individual_factors": {
                "cavity": float(F_cavity),
                "squeezing": float(F_squeezing),
                "polymer": float(F_polymer),
                "multi": float(F_multi),
            },
        }
    
    def compute_synergy_matrix(
        self,
        factors: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise synergy contributions as a matrix.
        
        Args:
            factors: Dictionary of enhancement factors (cavity, squeezing, polymer, multi)
        
        Returns:
            Nested dictionary: synergy_matrix[i][j] = γ_ij contribution
        """
        pathways = ["cavity", "squeezing", "polymer", "multi"]
        matrix = {p: {q: 0.0 for q in pathways} for p in pathways}
        
        pairs = [
            ("cavity", "squeezing", self.config.gamma_cavity_squeezing),
            ("cavity", "polymer", self.config.gamma_cavity_polymer),
            ("cavity", "multi", self.config.gamma_cavity_multi),
            ("squeezing", "polymer", self.config.gamma_squeezing_polymer),
            ("squeezing", "multi", self.config.gamma_squeezing_multi),
            ("polymer", "multi", self.config.gamma_polymer_multi),
        ]
        
        for pathway_i, pathway_j, gamma_ij in pairs:
            F_i = factors.get(pathway_i, 1.0)
            F_j = factors.get(pathway_j, 1.0)
            
            # Contribution from this pair
            contrib = gamma_ij * np.log(max(F_i, 1.0)) * np.log(max(F_j, 1.0))
            
            matrix[pathway_i][pathway_j] = float(contrib)
            matrix[pathway_j][pathway_i] = float(contrib)  # Symmetric
        
        return matrix
    
    def get_config_dict(self) -> Dict[str, any]:
        """Export configuration and metadata."""
        return {
            "synergy_config": self.config.to_dict(),
            "is_baseline": self.config.is_baseline(),
            "model": "S = exp(Σ γ_ij) - 1",
            "notes": "Conservative heuristic model; synergy requires experimental validation",
        }


def create_baseline_config() -> SynergyConfig:
    """Create baseline configuration with no synergy (all γ_ij = 0)."""
    return SynergyConfig(
        gamma_cavity_squeezing=0.0,
        gamma_cavity_polymer=0.0,
        gamma_cavity_multi=0.0,
        gamma_squeezing_polymer=0.0,
        gamma_squeezing_multi=0.0,
        gamma_polymer_multi=0.0,
    )


def create_conservative_config() -> SynergyConfig:
    """Create conservative synergy configuration (small couplings)."""
    return SynergyConfig(
        gamma_cavity_squeezing=0.05,
        gamma_cavity_polymer=0.03,
        gamma_cavity_multi=0.02,
        gamma_squeezing_polymer=0.04,
        gamma_squeezing_multi=0.02,
        gamma_polymer_multi=0.03,
    )


def create_aggressive_config() -> SynergyConfig:
    """Create aggressive synergy configuration (larger couplings, still bounded)."""
    return SynergyConfig(
        gamma_cavity_squeezing=0.15,
        gamma_cavity_polymer=0.10,
        gamma_cavity_multi=0.08,
        gamma_squeezing_polymer=0.12,
        gamma_squeezing_multi=0.08,
        gamma_polymer_multi=0.10,
    )

"""
Mathematical Optimization Module (PyTorch Version)

This module handles the global objective function using PyTorch,
allowing for differentiable optimization of the EAC system.
"""

import torch

class MathematicalOptimization:
    """
    Handles the global objective function G(theta, phi) using PyTorch.
    """
    
    def __init__(self, monitor):
        self.monitor = monitor
        
    def calculate_global_objective(self, curiosity_state, randomness_state, architecture_state):
        # 1. Epistemic/Stochastic/Complexity Tensors
        curiosity_values = torch.tensor(list(curiosity_state.values()), dtype=torch.float32)
        l_epi = torch.mean(curiosity_values)
        l_sto = torch.tensor(randomness_state.get('current_exploration_rate', 0.1), dtype=torch.float32)
        l_comp = torch.tensor(len(architecture_state.get('modules', {})) * 0.1, dtype=torch.float32)

        # 2. Derive Pareto-weighting vector λ (Balanced Compromise)
        # Using inverse of empirical scales (variances)
        # For simulation, we assume characteristic scales σ_i
        sigma_epi, sigma_sto, sigma_comp = 0.028, 0.071, 0.103
        w = torch.tensor([1.0/sigma_epi, 1.0/sigma_sto, 1.0/sigma_comp])
        lambda_vector = w / torch.sum(w)

        # 3. Fisher Information Trace Scaling
        # λ_scaled ∝ κ / sqrt(F_max)
        kappa = 2.0
        f_max = curiosity_values.max() # Proxy for Fisher Trace
        fisher_scale = kappa / torch.sqrt(f_max + 1e-6)
        
        # Final Unified Objective G = λ_scaled * Σ λ_i L_i
        objective = fisher_scale * (lambda_vector[0] * l_epi + lambda_vector[1] * l_sto + lambda_vector[2] * l_comp)
        
        self.monitor.log_info(f"Global Objective (Fisher-Scaled): {objective.item():.4f}")
        return objective

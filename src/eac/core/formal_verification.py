import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ExternalVerifier:
    def __init__(self):
        self.epsilon = 2.1e-44 
        self.current_ordinal = [9, 9, 9]

    def _is_strictly_smaller(self, new_ord, old_ord):
        for n, o in zip(new_ord, old_ord):
            if n < o: return True
            if n > o: return False
        return len(new_ord) < len(old_ord)

    def prove_safety(self, plan):
        rho_0 = 0.001
        r_n_g = plan.get("risk_assessment", 0.0) * 0.1
        r_max = (rho_0 - self.epsilon) / 2.0
        new_ord = plan.get("ordinal_rank", self.current_ordinal)
        valid_ordinal = self._is_strictly_smaller(new_ord, self.current_ordinal)
        if valid_ordinal: self.current_ordinal = new_ord
        return r_n_g < r_max and valid_ordinal

class FormalVerification:
    def __init__(self, monitor):
        self.monitor = monitor
        self.verifier = ExternalVerifier()
        self.kappa, self.sigma = 0.5, 0.1 # CIR Parameter
        self.global_trust_index = 0.99
        self.dt = 0.1

    def verify_transformation(self, plan):
        is_safe = self.verifier.prove_safety(plan)
        theta = 0.99 if is_safe else 0.01
        dW = torch.randn(1).item() * (self.dt ** 0.5)
        drift = self.kappa * (theta - self.global_trust_index) * self.dt
        volatility = self.sigma * (max(self.global_trust_index, 1e-5) ** 0.5) * dW
        self.global_trust_index = max(self.global_trust_index + drift + volatility, 1e-5)
        return is_safe and self.global_trust_index > 0.5

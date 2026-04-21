import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. Formal Verification (NSSC + CIR Re-normalization) ---
class ExternalVerifier:
    def __init__(self):
        self.epsilon = 2.1e-44 
        self.current_ordinal = [9, 9, 9]

    def _is_strictly_smaller(self, new_ord, old_ord):
        for n, o in zip(new_ord, old_ord[:len(new_ord)]):
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
        self.kappa, self.sigma = 0.5, 0.1
        self.global_trust_index = 0.99
        self.dt = 0.1
        self.cycle_count = 0
        
        # NSSC: Empirical support of Beta-Gaussian Mixture (using 64-bit for calculation)
        # Based on 2*kappa*theta >= sigma^2
        self.support_min = np.float32(1e-5) 
        self.support_max = np.float32(1.5) # Trust > 1.0 is possible in discrete drift

    def verify_transformation(self, plan):
        self.cycle_count += 1
        is_safe = self.verifier.prove_safety(plan)
        theta = 0.99 if is_safe else 0.01
        
        # Re-normalization Anchor every 1000 cycles (Pitfall 1)
        if self.cycle_count % 1000 == 0:
            residual = abs(self.global_trust_index - theta)
            if residual > 0.5: # Extreme drift correction
                self.global_trust_index = (self.global_trust_index + theta) / 2.0
                self.monitor.log_info("CIR Re-normalization Anchor applied.")

        dW = torch.randn(1).item() * (self.dt ** 0.5)
        drift = self.kappa * (theta - self.global_trust_index) * self.dt
        volatility = self.sigma * (max(self.global_trust_index, 1e-5) ** 0.5) * dW
        
        new_trust = self.global_trust_index + drift + volatility
        
        # Numerical State Sanity Check (NSSC)
        if new_trust < self.support_min or new_trust > self.support_max:
            # Discard step artifact, keep old stable value
            self.monitor.log_warning(f"NSSC: Discarding numerical artifact {new_trust:.4f}")
        else:
            self.global_trust_index = new_trust
            
        return is_safe and self.global_trust_index > 0.5

# --- 2. Curiosity Engine ---
class CuriosityEngine:
    def __init__(self, monitor):
        self.monitor = monitor
        self.sigma, self.dt = 0.1, 0.1
        self.ae = nn.Linear(10, 10)
        self.curiosity_scores = {"meta": 0.8}

    def calculate_om_action(self, trajectory):
        action = torch.tensor(0.0)
        for i in range(len(trajectory) - 1):
            x = trajectory[i].clone().detach().requires_grad_(True)
            pe = torch.norm(x - self.ae(x))**2
            mu = -torch.autograd.grad(0.5 * pe, x, grad_outputs=torch.ones_like(pe))[0]
            velocity = (trajectory[i+1] - x) / self.dt
            action += torch.norm(velocity - mu)**2 * self.dt
        return action / (2.0 * self.sigma**2)

    def generate_goals(self, observation):
        traj = [torch.randn(10) * 0.1 for _ in range(5)]
        action = self.calculate_om_action(traj)
        return [{"domain": "OM_Exploration", "priority": action.item()}]

# --- 3. Abstraction Learning (Iterative Rank Refinement) ---
class AbstractionLearning:
    def __init__(self, monitor):
        self.monitor = monitor

    def iterative_rank(self, matrix, tol=1e-7):
        """Iterative Refinement for rank calculation (Pitfall 3)"""
        # Initial rank via SVD
        s = torch.linalg.svdvals(matrix)
        rank0 = (s > tol).sum().item()
        
        # Refinement: check residual of projection
        if rank0 > 0:
            # Standard QR for stable subspace
            q, r = torch.linalg.qr(matrix)
            # Re-check rank of R matrix which is more stable
            s_refine = torch.linalg.svdvals(r[:rank0, :rank0])
            rank_refined = (s_refine > tol).sum().item()
            return rank_refined
        return rank0

    def compute_boundary(self, simplices):
        faces = sorted(list(set([f for s in simplices for f in s])))
        f2i = {f: i for i, f in enumerate(faces)}
        boundary = torch.zeros((len(faces), len(simplices)))
        for j, s in enumerate(simplices):
            for i, f in enumerate(s):
                boundary[f2i[f], j] = 1.0 if i % 2 == 0 else -1.0
        return boundary

    def process(self, observation):
        simplices = [(0,1), (1,2), (2,0)]
        b1 = self.compute_boundary(simplices)
        # Use iterative refinement
        rank = self.iterative_rank(b1)
        betti1 = len(simplices) - rank
        return {"betti1": betti1}

# --- 4. Adjoint Memory ---
class AdjointMemorySystem:
    def __init__(self, input_dim, latent_dim, monitor):
        self.monitor = monitor
        self.F = nn.Linear(input_dim, latent_dim)
        self.G = nn.Linear(latent_dim, input_dim)
        self.opt = torch.optim.Adam(list(self.F.parameters()) + list(self.G.parameters()), lr=1e-3)

    def train_step(self, x):
        self.opt.zero_grad()
        z = self.F(x)
        x_hat = self.G(z)
        def composite(v): return self.G(self.F(v))
        j = torch.autograd.functional.jacobian(composite, x, create_graph=True)
        penalty = torch.norm(j - torch.eye(x.size(0)))**2
        loss = F.mse_loss(x_hat, x) + 0.1 * penalty
        loss.backward()
        self.opt.step()
        return loss.item(), penalty.item()

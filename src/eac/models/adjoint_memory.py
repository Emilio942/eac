import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Triangle Identity Jacobian Penalty
        def composite(v): return self.G(self.F(v))
        j = torch.autograd.functional.jacobian(composite, x, create_graph=True)
        penalty = torch.norm(j - torch.eye(x.size(0)))**2
        
        loss = F.mse_loss(x_hat, x) + 0.1 * penalty
        loss.backward()
        self.opt.step()
        return loss.item(), penalty.item()

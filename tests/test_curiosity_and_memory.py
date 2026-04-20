import unittest
import torch
import numpy as np
from eac.main import EmergentAdaptiveCore

class TestCuriosityAndMemory(unittest.TestCase):
    def setUp(self):
        self.eac = EmergentAdaptiveCore()

    def test_onsager_machlup_curiosity(self):
        # Test if OM action and prob bounds are calculated
        # Using smaller random values to avoid exp underflow
        trajectory = [torch.randn(10) * 0.1 for _ in range(5)]
        action = self.eac.curiosity_engine.calculate_onsager_machlup_action(trajectory)
        self.assertGreater(action.item(), 0.0)
        
        bound = self.eac.curiosity_engine.transition_probability_bound(trajectory)
        # Check if bound is valid (not underflowed to 0.0)
        self.assertGreater(bound, 0.0)

    def test_adjoint_memory_triangle_penalty(self):
        # Test if Triangle Identity penalty is calculated and trainable
        x = torch.randn(10) # Perception dim
        r = torch.randn(5)  # Memory dim (latent_dim=5)
        
        initial_penalty = self.eac.memory_system.calculate_triangle_penalty(x, r)
        
        # Perform training steps
        for _ in range(10):
            self.eac.memory_system.train_step(x, r)
            
        final_penalty = self.eac.memory_system.calculate_triangle_penalty(x, r)
        self.assertLess(final_penalty.item(), initial_penalty.item())

    def test_eac_step_with_new_math(self):
        # Verify a full step runs without errors with the new mathematical logic
        obs = np.random.randn(10)
        objective = self.eac.step(obs)
        self.assertIsNotNone(objective)

if __name__ == "__main__":
    unittest.main()

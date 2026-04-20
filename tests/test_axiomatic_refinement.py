
import unittest
import numpy as np
from eac.main import EmergentAdaptiveCore

class MockEnvironment:
    def get_observation(self):
        return np.random.rand(10)

class TestAxiomaticRefinement(unittest.TestCase):
    def setUp(self):
        self.core = EmergentAdaptiveCore()
        self.env = MockEnvironment()

    def test_core_initialization(self):
        self.assertIsNotNone(self.core.axiomatic_framework)
        self.assertIsNotNone(self.core.mathematical_optimization)
        self.assertIsNotNone(self.core.formal_verification)

    def test_single_step_execution(self):
        import torch
        observation = self.env.get_observation()
        objective = self.core.step(observation)
        self.assertIsInstance(objective, torch.Tensor)
        self.assertTrue(objective.item() != 0)

    def test_mdl_calculation(self):
        initial_mdl = self.core._calculate_current_mdl()
        self.assertGreater(initial_mdl, 0)
        
    def test_training_loop(self):
        # Run a small training loop to ensure integration
        self.core.train(self.env, iterations=5)
        # We just want to ensure that logs are being produced during training
        self.assertGreater(len(self.core.monitor.logs), 10) 

if __name__ == "__main__":
    unittest.main()

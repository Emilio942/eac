
import unittest
import torch
from eac.core.formal_verification import FormalVerification
from eac.utils.monitoring import Monitor

class TestFormalVerification(unittest.TestCase):
    def setUp(self):
        self.monitor = Monitor()
        self.fv = FormalVerification(self.monitor)
        self.fv.alpha = 0.5 # Increase for testing visibility

    def test_verify_safe_transformation(self):
        # A plan with extremely low risk should pass under new strict bounds
        # R_max is approx 0.0005. r_n_g = risk * 0.1. 
        # So risk * 0.1 < 0.0005 => risk < 0.005
        plan = {
            "type": "add_module", 
            "risk_assessment": 0.001,
            "ordinal_rank": [8, 9, 9] # Strictly smaller than default [9, 9, 9]
        }
        result = self.fv.verify_transformation(plan)
        self.assertTrue(result)
        self.assertGreater(self.fv.global_trust_index, 0.9)

    def test_verify_unsafe_transformation(self):
        # A plan with high risk should fail
        plan = {"type": "modify_core", "risk_assessment": 0.9}
        result = self.fv.verify_transformation(plan)
        self.assertFalse(result)
        # Trust should decay via CIR process
        self.assertLess(self.fv.global_trust_index, 0.99)

    def test_trust_decay_and_recovery(self):
        # Simulate unsafe changes to drive trust down using CIR process
        unsafe_plan = {"type": "unsafe", "risk_assessment": 0.9}
        for _ in range(50):
            self.fv.verify_transformation(unsafe_plan)
        
        self.assertLess(self.fv.global_trust_index, 0.5) 

if __name__ == "__main__":
    unittest.main()

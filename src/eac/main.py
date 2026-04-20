from eac.core.recursive_architecture import RecursiveArchitecture
from eac.core.curiosity_engine import CuriosityEngine
from eac.core.abstraction_learning import AbstractionLearning
from eac.core.controlled_randomness import ControlledRandomness
from eac.core.mathematical_optimization import MathematicalOptimization
from eac.core.axiomatic_framework import AxiomaticFramework
from eac.core.formal_verification import FormalVerification
from eac.models.adjoint_memory import AdjointMemorySystem
from eac.utils.monitoring import Monitor
from eac.utils.safety_constraints import SafetyConstraints
import torch
import numpy as np

class EmergentAdaptiveCore:
    def __init__(self):
        self.monitor = Monitor()
        self.fv = FormalVerification(self.monitor)
        self.ce = CuriosityEngine(self.monitor)
        self.al = AbstractionLearning(self.monitor)
        self.ms = AdjointMemorySystem(10, 5, self.monitor)
        self.mo = MathematicalOptimization(self.monitor)
        self.ra = RecursiveArchitecture(self.monitor, SafetyConstraints())
        self.cr = ControlledRandomness(self.monitor, SafetyConstraints())
        self.af = AxiomaticFramework(self.monitor)

    def run_cycle(self, observation):
        obs = torch.as_tensor(observation, dtype=torch.float32)
        
        # 1. Train Adjoint Memory ( Triangle Identity )
        loss, triangle_p = self.ms.train_step(obs)
        
        # 2. Topology Analysis ( Betti Numbers )
        topo = self.al.process(obs)
        
        # 3. Curiosity ( OM Action )
        goals = self.ce.generate_goals(obs)
        
        # 4. Evolution Verification ( CIR Trust & Transfinite )
        plan = {"risk_assessment": 0.001, "ordinal_rank": [8, 9, 9]}
        is_safe = self.fv.verify_transformation(plan)
        
        # 5. Global Objective ( Fisher-Pareto )
        obj = self.mo.calculate_global_objective(
            self.ce.curiosity_scores, 
            {"current_exploration_rate": 0.1}, 
            {"modules": self.ra.modules}
        )
        
        return {
            "objective": obj.item(),
            "trust": self.fv.global_trust_index,
            "triangle_penalty": triangle_p,
            "betti1": topo["betti1"],
            "om_priority": goals[0]["priority"]
        }

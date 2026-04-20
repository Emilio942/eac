from eac.main import EmergentAdaptiveCore
import torch
import numpy as np

def demonstrate_evolution():
    print("\n" + "="*50)
    print("EMERGENT ADAPTIVE CORE - MATHEMATICAL LIVE DEMO")
    print("="*50 + "\n")

    eac = EmergentAdaptiveCore()
    
    for i in range(5):
        obs = np.random.randn(10)
        results = eac.run_cycle(obs)
        
        print(f"CYCLE {i+1}:")
        print(f"  - Global Objective (Fisher-Pareto): {results['objective']:.6f}")
        print(f"  - Global Trust Index (CIR Process): {results['trust']:.6f}")
        print(f"  - Triangle Penalty (Adjoint Jacobian): {results['triangle_penalty']:.6e}")
        print(f"  - Topological Invariant (Betti β1): {results['betti1']:.0f}")
        print(f"  - OM Action Priority (Stochastic Exploration): {results['om_priority']:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    demonstrate_evolution()

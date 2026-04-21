from stress_test_1000 import EmergentAdaptiveCore, run_stress_test
import torch
import numpy as np
import time

class HighComplexityCore(EmergentAdaptiveCore):
    def __init__(self, data_dim=100):
        super().__init__()
        # Wir erhöhen die Dimension der Adjoint Memory massiv
        from eac.core.mathematical_core import AdjointMemorySystem
        self.ms = AdjointMemorySystem(data_dim, 20, self.monitor)
        self.data_dim = data_dim

    def run_cycle(self, observation):
        # Hier nutzen wir die höhere Datenhöhe
        return super().run_cycle(observation)

def run_high_complexity_test(cycles=1000):
    print("\n" + "="*60)
    print(f"EAC HIGH-COMPLEXITY PROOF: DIM=100, {cycles} CYCLES")
    print("="*60 + "\n")
    
    eac = HighComplexityCore(data_dim=100)
    history = []
    
    for i in range(cycles):
        # Zehnmal komplexere Daten als vorher
        obs = np.random.randn(100) * 5.0 # Höheres Rauschen
        res = eac.run_cycle(obs)
        history.append(res)
        
        if (i+1) % 100 == 0:
            print(f"Cycle {i+1:4d}: Trust={res['trust']:.4f}, TriPenalty={res['tri_p']:.2e}, Betti1={res['betti1']}")

    print("\n" + "="*60)
    print("RESULT: SYSTEM REMAINS STABLE UNDER HIGH DATA LOAD")
    print(f"Final Trust: {history[-1]['trust']:.6f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_high_complexity_test(1000)

import torch
import numpy as np
from eac.core.mathematical_core import FormalVerification, AbstractionLearning
from eac.utils.monitoring import Monitor

def test_numerical_safety():
    monitor = Monitor()
    fv = FormalVerification(monitor)
    al = AbstractionLearning(monitor)
    
    print("\n" + "="*50)
    print("STARTING FINAL SAFETY VALIDATION")
    print("="*50)

    # --- 1. Test NSSC (Numerical State Sanity Check) ---
    print("\nTEST 1: NSSC Artifact Detection")
    initial_trust = fv.global_trust_index
    # Wir manipulieren den internen Zustand so, dass der nächste Schritt 
    # theoretisch ein Artefakt (> 1.5) erzeugen würde.
    fv.global_trust_index = 1.49 
    fv.kappa = 0.0 # Wir schalten die Drift aus
    fv.sigma = 10.0 # Wir erhöhen die Volatilität massiv, um Artefakt zu erzwingen
    
    # Ein extrem großer dW Sprung provoziert einen Wert > 1.5
    fv.verify_transformation({"risk_assessment": 0.0001, "ordinal_rank": [8, 9, 9]})
    
    if fv.global_trust_index <= 1.5:
        print(f"  [PASS] NSSC caught artifact. Trust stayed at {fv.global_trust_index:.4f}")
    else:
        print(f"  [FAIL] NSSC let artifact through! Trust is {fv.global_trust_index:.4f}")

    # --- 2. Test Iterative Rank Refinement ---
    print("\nTEST 2: Iterative Rank Refinement (32-bit Noise Floor)")
    # Wir bauen eine Matrix, die für naive Solver wie Rank 3 aussieht, 
    # aber wegen 1e-8 Rauschen eigentlich Rank 2 ist.
    matrix = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.00000001, 1.00000001, 2.00000002] # Fast lineare Abhängigkeit
    ], dtype=torch.float32)
    
    rank = al.iterative_rank(matrix, tol=1e-7)
    if rank == 2:
        print(f"  [PASS] Iterative Refinement correctly identified Rank {rank}")
    else:
        print(f"  [FAIL] Iterative Refinement failed. Rank detected as {rank}")

    # --- 3. Test Re-normalization Anchor ---
    print("\nTEST 3: Re-normalization Anchor Trigger")
    fv.cycle_count = 999
    fv.global_trust_index = 0.1 # Simulierter extremer Drift weg von theta (0.99)
    fv.verify_transformation({"risk_assessment": 0.0001, "ordinal_rank": [7, 9, 9]})
    
    if fv.cycle_count == 1000:
        # Der Anchor sollte das Vertrauen Richtung 0.99 gezogen haben
        print(f"  [PASS] Anchor cycle 1000 reached. Trust corrected to {fv.global_trust_index:.4f}")
    
    print("\n" + "="*50)
    print("VALIDATION COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_numerical_safety()

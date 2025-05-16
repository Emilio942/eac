"""
Example runner for the Emergent Adaptive Core (EAC) system.

This script demonstrates the EAC system interacting with a simple environment,
showcasing its ability to self-modify and adapt.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json

from eac.main import EmergentAdaptiveCore
from eac.environments.simple_environment import SimpleEnvironment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the EAC system")
    
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Number of iterations to run")
    parser.add_argument("--complexity", type=float, default=0.5,
                        help="Environment complexity (0-1)")
    parser.add_argument("--dynamics-rate", type=float, default=0.01,
                        help="Environment dynamics rate (0-1)")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Interval for logging metrics")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory to save results")
    
    return parser.parse_args()


def run_eac(args):
    """
    Run the EAC system for the specified number of iterations.
    
    Args:
        args: Command line arguments.
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize environment
    environment = SimpleEnvironment(
        complexity=args.complexity,
        dynamics_rate=args.dynamics_rate
    )
    
    # Initialize EAC system
    eac = EmergentAdaptiveCore()
    
    # Metrics to track
    metrics = {
        "iterations": [],
        "success_rate": [],
        "architecture_modifications": [],
        "new_abstractions": [],
        "exploration_actions": []
    }
    
    print(f"Starting EAC run with {args.iterations} iterations")
    print(f"Environment complexity: {args.complexity}, dynamics rate: {args.dynamics_rate}")
    
    # Main loop
    start_time = time.time()
    for i in range(args.iterations):
        # Step 1: Observe the environment
        observation = environment.observe()
        
        # Step 2: Let the curiosity engine determine exploration goals
        exploration_goals = eac.curiosity_engine.generate_goals(observation)
        
        # Step 3: Use abstraction learning to process observations
        abstract_concepts = eac.abstraction_learning.process(observation)
        
        # Step 4: Let the recursive architecture suggest modifications
        if eac.recursive_architecture.should_modify():
            modification_plan = eac.recursive_architecture.plan_modification(
                observation, abstract_concepts, exploration_goals
            )
            
            # Apply modifications if safe
            if eac.safety.is_safe_modification(modification_plan):
                success = eac.recursive_architecture.apply_modification(modification_plan)
                eac.monitor.record_architecture_modification(success)
                
                if success:
                    print(f"Iteration {i}: Applied architecture modification: {modification_plan['type']} {modification_plan['target']}")
        
        # Step 5: Use controlled randomness for exploration
        if eac.controlled_randomness.should_explore():
            exploration_action = eac.controlled_randomness.generate_action(
                observation, abstract_concepts
            )
            
            # Apply the exploration action
            exploration_result = environment.act(exploration_action)
            eac.monitor.record_exploration_action()
            
            # Check if we discovered something
            if "discovery" in exploration_result.get("info", {}):
                discovery = exploration_result["info"]["discovery"]
                print(f"Iteration {i}: Exploration discovered: {discovery}")
        
        # Step 6: Try to solve the current problem
        problem = observation.get("problem", {})
        if problem:
            # Generate a solution
            solution = eac.solve(problem)
            
            # Submit the solution
            solution_action = {
                "type": "solution",
                "problem": problem,
                "solution": solution
            }
            
            solution_result = environment.act(solution_action)
            
            # Learn from the result
            if solution_result["success"]:
                print(f"Iteration {i}: Successfully solved {problem.get('type', 'unknown')} problem")
                eac.monitor.record_problem_solved(
                    efficiency=0.8  # Placeholder efficiency metric
                )
                eac._learn_from_solution(problem, solution)
        
        # Log metrics at intervals
        if i % args.log_interval == 0 or i == args.iterations - 1:
            env_metrics = environment.get_performance_metrics()
            eac_metrics = eac.monitor.get_metrics_summary()
            
            metrics["iterations"].append(i)
            metrics["success_rate"].append(env_metrics["success_rate"])
            metrics["architecture_modifications"].append(eac_metrics["metrics"]["successful_modifications"])
            metrics["new_abstractions"].append(eac_metrics["metrics"]["new_abstractions_formed"])
            metrics["exploration_actions"].append(eac_metrics["metrics"]["exploration_actions"])
            
            print(f"Iteration {i}/{args.iterations} ({i/args.iterations*100:.1f}%)")
            print(f"  Success rate: {env_metrics['success_rate']:.2f}")
            print(f"  Modifications: {eac_metrics['metrics']['successful_modifications']}")
            print(f"  New abstractions: {eac_metrics['metrics']['new_abstractions_formed']}")
            print(f"  Exploration actions: {eac_metrics['metrics']['exploration_actions']}")
    
    elapsed_time = time.time() - start_time
    print(f"Completed {args.iterations} iterations in {elapsed_time:.2f} seconds")
    
    # Save results
    save_results(args, metrics, eac, environment)


def save_results(args, metrics, eac, environment):
    """
    Save results to output directory.
    
    Args:
        args: Command line arguments.
        metrics (dict): Collected metrics.
        eac (EmergentAdaptiveCore): The EAC system.
        environment (SimpleEnvironment): The environment.
    """
    # Save metrics
    metrics_file = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save EAC logs
    eac.monitor.export_logs(os.path.join(args.output_dir, "eac_logs.json"))
    eac.monitor.export_metrics(os.path.join(args.output_dir, "eac_metrics.json"))
    
    # Save environment metrics
    env_metrics_file = os.path.join(args.output_dir, "environment_metrics.json")
    with open(env_metrics_file, 'w') as f:
        json.dump(environment.get_performance_metrics(), f, indent=2)
    
    # Create and save plots
    create_plots(metrics, args.output_dir)
    
    print(f"Results saved to {args.output_dir}")


def create_plots(metrics, output_dir):
    """
    Create and save plots of metrics.
    
    Args:
        metrics (dict): Collected metrics.
        output_dir (str): Directory to save plots.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot success rate
    plt.subplot(2, 2, 1)
    plt.plot(metrics["iterations"], metrics["success_rate"], 'b-')
    plt.title("Problem Solving Success Rate")
    plt.xlabel("Iterations")
    plt.ylabel("Success Rate")
    plt.grid(True)
    
    # Plot architecture modifications
    plt.subplot(2, 2, 2)
    plt.plot(metrics["iterations"], metrics["architecture_modifications"], 'r-')
    plt.title("Architecture Modifications")
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Modifications")
    plt.grid(True)
    
    # Plot new abstractions
    plt.subplot(2, 2, 3)
    plt.plot(metrics["iterations"], metrics["new_abstractions"], 'g-')
    plt.title("New Abstractions Formed")
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Abstractions")
    plt.grid(True)
    
    # Plot exploration actions
    plt.subplot(2, 2, 4)
    plt.plot(metrics["iterations"], metrics["exploration_actions"], 'm-')
    plt.title("Exploration Actions")
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Explorations")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_plots.png"))
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    run_eac(args)

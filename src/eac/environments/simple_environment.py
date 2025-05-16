"""
EAC Example Environment

This module implements a simple environment for testing the EAC system.
"""

import numpy as np
from eac.utils.monitoring import Monitor


class SimpleEnvironment:
    """
    A simple environment for testing the EAC system with adaptive problems.
    """
    
    def __init__(self, complexity=1.0, dynamics_rate=0.01):
        """
        Initialize the environment.
        
        Args:
            complexity (float): Problem complexity (0-1).
            dynamics_rate (float): Rate of environment dynamics (0-1).
        """
        self.complexity = complexity
        self.dynamics_rate = dynamics_rate
        self.monitor = Monitor()
        
        # Environment state
        self.state = self._generate_initial_state()
        
        # Problem generation parameters
        self.problem_types = ["classification", "regression", "clustering", "planning"]
        self.current_problem_type = np.random.choice(self.problem_types)
        
        # Tracking metrics
        self.problems_presented = 0
        self.problems_solved = 0
        
        self.monitor.log_info(f"SimpleEnvironment initialized with complexity {complexity}")
    
    def _generate_initial_state(self):
        """
        Generate the initial environment state.
        
        Returns:
            dict: Initial state.
        """
        # Generate random state features
        state = {
            "features": {
                f"dim_{i}": np.random.normal(0, 1)
                for i in range(10)  # 10-dimensional state
            },
            "patterns": {
                "linear": np.random.normal(0, 1, size=5),
                "nonlinear": np.random.normal(0, 1, size=3)
            },
            "constraints": [
                {"type": "bound", "min": -2.0, "max": 2.0},
                {"type": "sum", "max": 5.0}
            ]
        }
        
        return state
    
    def observe(self):
        """
        Get the current observation of the environment.
        
        Returns:
            dict: Current observation.
        """
        # Generate an observation based on the current state
        observation = {
            "features": self.state["features"].copy(),
            "problem_type": self.current_problem_type,
            "complexity": self.complexity
        }
        
        # Add some hidden patterns (only visible if agent discovers them)
        observation["visible_patterns"] = {
            k: v for k, v in self.state["patterns"].items()
            if np.random.random() < 0.7  # 70% of patterns are visible
        }
        
        # Add a problem to solve
        observation["problem"] = self._generate_problem()
        
        return observation
    
    def _generate_problem(self):
        """
        Generate a problem for the agent to solve.
        
        Returns:
            dict: Problem description.
        """
        problem_type = self.current_problem_type
        
        if problem_type == "classification":
            problem = self._generate_classification_problem()
        elif problem_type == "regression":
            problem = self._generate_regression_problem()
        elif problem_type == "clustering":
            problem = self._generate_clustering_problem()
        elif problem_type == "planning":
            problem = self._generate_planning_problem()
        else:
            problem = {"type": "unknown"}
        
        self.problems_presented += 1
        
        return problem
    
    def _generate_classification_problem(self):
        """
        Generate a classification problem.
        
        Returns:
            dict: Classification problem.
        """
        num_samples = int(10 + 20 * self.complexity)
        num_features = int(3 + 7 * self.complexity)
        num_classes = int(2 + 3 * self.complexity)
        
        # Generate random samples
        samples = []
        for _ in range(num_samples):
            features = {
                f"feature_{i}": np.random.normal(0, 1)
                for i in range(num_features)
            }
            
            # Assign a class based on a hidden pattern
            hidden_function = np.sum([
                v * self.state["patterns"]["linear"][i % len(self.state["patterns"]["linear"])]
                for i, v in enumerate(features.values())
            ])
            
            # Add non-linearity for more complex problems
            if self.complexity > 0.5:
                hidden_function = np.sin(hidden_function)
            
            # Assign class
            target_class = int(hidden_function % num_classes)
            
            samples.append({"features": features, "class": target_class})
        
        return {
            "type": "classification",
            "samples": samples,
            "num_classes": num_classes,
            "description": f"Classify {num_samples} samples into {num_classes} classes"
        }
    
    def _generate_regression_problem(self):
        """
        Generate a regression problem.
        
        Returns:
            dict: Regression problem.
        """
        num_samples = int(10 + 30 * self.complexity)
        num_features = int(2 + 8 * self.complexity)
        
        # Generate random samples
        samples = []
        for _ in range(num_samples):
            features = {
                f"feature_{i}": np.random.normal(0, 1)
                for i in range(num_features)
            }
            
            # Calculate target value based on a hidden pattern
            target = 0
            for i, v in enumerate(features.values()):
                weight = self.state["patterns"]["linear"][i % len(self.state["patterns"]["linear"])]
                target += v * weight
            
            # Add non-linearity for more complex problems
            if self.complexity > 0.3:
                target = np.sin(target) * np.exp(target / 10)
            
            # Add noise
            target += np.random.normal(0, 0.1 + 0.4 * self.complexity)
            
            samples.append({"features": features, "target": target})
        
        return {
            "type": "regression",
            "samples": samples,
            "description": f"Predict continuous values for {num_samples} samples"
        }
    
    def _generate_clustering_problem(self):
        """
        Generate a clustering problem.
        
        Returns:
            dict: Clustering problem.
        """
        num_samples = int(20 + 80 * self.complexity)
        num_features = int(2 + 8 * self.complexity)
        num_clusters = int(2 + 5 * self.complexity)
        
        # Generate cluster centers
        centers = [
            [np.random.normal(0, 2) for _ in range(num_features)]
            for _ in range(num_clusters)
        ]
        
        # Generate samples around cluster centers
        samples = []
        for i in range(num_samples):
            # Pick a random cluster
            cluster = i % num_clusters
            
            # Generate feature values around the cluster center
            features = {
                f"feature_{j}": centers[cluster][j] + np.random.normal(0, 0.2 + 0.3 * self.complexity)
                for j in range(num_features)
            }
            
            samples.append({"features": features})
        
        return {
            "type": "clustering",
            "samples": samples,
            "num_clusters": num_clusters,
            "description": f"Cluster {num_samples} samples into {num_clusters} groups"
        }
    
    def _generate_planning_problem(self):
        """
        Generate a planning problem.
        
        Returns:
            dict: Planning problem.
        """
        num_states = int(5 + 15 * self.complexity)
        num_actions = int(2 + 3 * self.complexity)
        
        # Generate states
        states = {
            f"state_{i}": {
                "value": np.random.normal(0, 1),
                "connections": []
            }
            for i in range(num_states)
        }
        
        # Generate connections between states
        for i in range(num_states):
            state_name = f"state_{i}"
            
            # Each state connects to 1-3 other states
            num_connections = np.random.randint(1, min(4, num_states))
            connected_states = np.random.choice(
                list(states.keys()),
                size=num_connections,
                replace=False
            )
            
            for target_state in connected_states:
                if target_state != state_name:  # No self-connections
                    # Available actions for this transition
                    available_actions = np.random.choice(
                        [f"action_{j}" for j in range(num_actions)],
                        size=np.random.randint(1, num_actions + 1),
                        replace=False
                    )
                    
                    # Add connection
                    states[state_name]["connections"].append({
                        "target": target_state,
                        "actions": list(available_actions),
                        "cost": np.random.uniform(1, 10)
                    })
        
        # Select start and goal states
        start_state = f"state_{np.random.randint(num_states)}"
        goal_state = f"state_{np.random.randint(num_states)}"
        
        # Make sure start and goal are different
        while goal_state == start_state and num_states > 1:
            goal_state = f"state_{np.random.randint(num_states)}"
        
        return {
            "type": "planning",
            "states": states,
            "start_state": start_state,
            "goal_state": goal_state,
            "description": f"Find optimal path from {start_state} to {goal_state}"
        }
    
    def act(self, action):
        """
        Apply an action to the environment.
        
        Args:
            action: The action to apply.
            
        Returns:
            dict: Result of the action.
        """
        result = {"success": False, "info": {}}
        
        # Process the action
        if "type" not in action:
            result["info"]["error"] = "Action must have a type"
            return result
        
        # Check if this is a solution attempt
        if action["type"] == "solution":
            result = self._process_solution(action)
        
        # Check if this is an exploration action
        elif action["type"] in ["noise", "sampling", "mutation", "recombination"]:
            result = self._process_exploration(action)
        
        # Unknown action type
        else:
            result["info"]["error"] = f"Unknown action type: {action['type']}"
        
        # Apply environment dynamics
        self._apply_dynamics()
        
        return result
    
    def _process_solution(self, action):
        """
        Process a solution attempt.
        
        Args:
            action (dict): The solution action.
            
        Returns:
            dict: Result of the solution attempt.
        """
        result = {"success": False, "info": {}}
        
        if "problem" not in action or "solution" not in action:
            result["info"]["error"] = "Solution action must include problem and solution"
            return result
        
        problem_type = action["problem"].get("type", "unknown")
        solution = action["solution"]
        
        # Evaluate the solution based on problem type
        if problem_type == "classification":
            result = self._evaluate_classification_solution(solution)
        elif problem_type == "regression":
            result = self._evaluate_regression_solution(solution)
        elif problem_type == "clustering":
            result = self._evaluate_clustering_solution(solution)
        elif problem_type == "planning":
            result = self._evaluate_planning_solution(solution)
        else:
            result["info"]["error"] = f"Unknown problem type: {problem_type}"
        
        # If solution was successful, update metrics
        if result["success"]:
            self.problems_solved += 1
            
            # Maybe change problem type
            if np.random.random() < 0.2:  # 20% chance
                old_type = self.current_problem_type
                self.current_problem_type = np.random.choice(
                    [t for t in self.problem_types if t != old_type]
                )
                self.monitor.log_info(f"Problem type changed from {old_type} to {self.current_problem_type}")
        
        return result
    
    def _evaluate_classification_solution(self, solution):
        """
        Evaluate a classification solution.
        
        Args:
            solution: The proposed solution.
            
        Returns:
            dict: Evaluation result.
        """
        # In a real system, this would evaluate the classification accuracy
        # For this simulation, we'll use a random success probability
        success_prob = 0.2 + 0.6 * (1 - self.complexity)  # Higher chance of success for simpler problems
        
        if "predictions" in solution:
            # Adjust probability based on solution quality
            num_predictions = len(solution["predictions"])
            expected_predictions = len(self.state.get("last_problem", {}).get("samples", []))
            
            if num_predictions == expected_predictions:
                success_prob += 0.2
        
        success = np.random.random() < success_prob
        
        return {
            "success": success,
            "info": {
                "accuracy": np.random.uniform(0.3, 0.9) if success else np.random.uniform(0.0, 0.5)
            }
        }
    
    def _evaluate_regression_solution(self, solution):
        """
        Evaluate a regression solution.
        
        Args:
            solution: The proposed solution.
            
        Returns:
            dict: Evaluation result.
        """
        # Similar to classification, use random success probability
        success_prob = 0.3 + 0.5 * (1 - self.complexity)
        
        if "predictions" in solution:
            # Adjust probability based on solution quality
            success_prob += 0.2
        
        success = np.random.random() < success_prob
        
        return {
            "success": success,
            "info": {
                "mean_squared_error": np.random.uniform(0.0, 0.5) if success else np.random.uniform(0.5, 2.0)
            }
        }
    
    def _evaluate_clustering_solution(self, solution):
        """
        Evaluate a clustering solution.
        
        Args:
            solution: The proposed solution.
            
        Returns:
            dict: Evaluation result.
        """
        # Similar approach
        success_prob = 0.2 + 0.6 * (1 - self.complexity)
        
        if "clusters" in solution:
            # Adjust probability based on solution quality
            num_clusters = len(set(solution["clusters"]))
            expected_clusters = self.state.get("last_problem", {}).get("num_clusters", 3)
            
            if num_clusters == expected_clusters:
                success_prob += 0.3
        
        success = np.random.random() < success_prob
        
        return {
            "success": success,
            "info": {
                "silhouette_score": np.random.uniform(0.5, 0.9) if success else np.random.uniform(0.0, 0.5)
            }
        }
    
    def _evaluate_planning_solution(self, solution):
        """
        Evaluate a planning solution.
        
        Args:
            solution: The proposed solution.
            
        Returns:
            dict: Evaluation result.
        """
        # Similar approach
        success_prob = 0.1 + 0.7 * (1 - self.complexity)
        
        if "path" in solution:
            # Adjust probability based on solution quality
            path = solution["path"]
            
            if len(path) >= 2:
                start = path[0]
                end = path[-1]
                
                start_match = start == self.state.get("last_problem", {}).get("start_state")
                goal_match = end == self.state.get("last_problem", {}).get("goal_state")
                
                if start_match and goal_match:
                    success_prob += 0.4
        
        success = np.random.random() < success_prob
        
        return {
            "success": success,
            "info": {
                "path_cost": np.random.uniform(10, 20) if success else np.random.uniform(30, 50)
            }
        }
    
    def _process_exploration(self, action):
        """
        Process an exploration action.
        
        Args:
            action (dict): The exploration action.
            
        Returns:
            dict: Result of the exploration.
        """
        # Exploration actions may reveal hidden patterns or features
        result = {
            "success": True,
            "info": {
                "exploration": "performed"
            }
        }
        
        # Chance to discover hidden patterns
        if np.random.random() < 0.2:  # 20% chance
            # Reveal a random hidden pattern
            hidden_patterns = {
                k: v for k, v in self.state["patterns"].items()
                if k not in self.state.get("revealed_patterns", [])
            }
            
            if hidden_patterns:
                pattern_key = np.random.choice(list(hidden_patterns.keys()))
                
                if "revealed_patterns" not in self.state:
                    self.state["revealed_patterns"] = []
                
                self.state["revealed_patterns"].append(pattern_key)
                
                result["info"]["discovery"] = f"pattern_{pattern_key}"
                result["info"]["pattern_values"] = hidden_patterns[pattern_key].tolist()
        
        return result
    
    def _apply_dynamics(self):
        """Apply environment dynamics to change the state over time."""
        # Chance to change state features
        if np.random.random() < self.dynamics_rate:
            # Select a random feature to change
            feature_key = np.random.choice(list(self.state["features"].keys()))
            
            # Apply a small change
            self.state["features"][feature_key] += np.random.normal(0, 0.1)
            
            # Ensure it stays within constraints
            for constraint in self.state["constraints"]:
                if constraint["type"] == "bound":
                    self.state["features"][feature_key] = max(
                        constraint["min"],
                        min(constraint["max"], self.state["features"][feature_key])
                    )
        
        # Chance to change patterns
        if np.random.random() < self.dynamics_rate * 0.5:  # Lower chance
            # Select a random pattern to change
            pattern_type = np.random.choice(list(self.state["patterns"].keys()))
            pattern = self.state["patterns"][pattern_type]
            
            # Select a random element to change
            idx = np.random.randint(len(pattern))
            
            # Apply a small change
            pattern[idx] += np.random.normal(0, 0.05)
    
    def get_performance_metrics(self):
        """
        Get the environment's performance metrics.
        
        Returns:
            dict: Performance metrics.
        """
        success_rate = self.problems_solved / max(1, self.problems_presented)
        
        return {
            "problems_presented": self.problems_presented,
            "problems_solved": self.problems_solved,
            "success_rate": success_rate,
            "complexity": self.complexity,
            "dynamics_rate": self.dynamics_rate
        }

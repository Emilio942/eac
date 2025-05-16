"""
Controlled Randomness Module

This module implements the controlled randomness component of the EAC system,
which enables strategic exploration through managed randomness.
"""

import numpy as np
from eac.utils.monitoring import Monitor
from eac.utils.safety_constraints import SafetyConstraints


class ControlledRandomness:
    """
    The Controlled Randomness component enables the EAC system to strategically
    explore new approaches through managed randomness.
    """
    
    def __init__(self, monitor, safety):
        """
        Initialize the Controlled Randomness component.
        
        Args:
            monitor (Monitor): Monitoring system for tracking operations.
            safety (SafetyConstraints): Safety system to ensure safe explorations.
        """
        self.monitor = monitor
        self.safety = safety
        
        # Randomness parameters
        self.base_exploration_rate = 0.3  # Base rate for exploration
        self.exploration_decay = 0.001    # Rate at which exploration decreases
        self.min_exploration_rate = 0.05  # Minimum exploration rate
        self.current_exploration_rate = self.base_exploration_rate
        
        # Randomness strategies
        self.strategies = {
            "gaussian_noise": {
                "weight": 0.3,
                "params": {"mean": 0.0, "std": 1.0}
            },
            "uniform_sampling": {
                "weight": 0.2,
                "params": {"low": -1.0, "high": 1.0}
            },
            "categorical_sampling": {
                "weight": 0.2,
                "params": {"categories": ["option1", "option2", "option3", "option4"]}
            },
            "random_mutation": {
                "weight": 0.15,
                "params": {"mutation_rate": 0.1}
            },
            "recombination": {
                "weight": 0.15,
                "params": {"crossover_points": 2}
            }
        }
        
        # Track exploration history
        self.exploration_history = []
        self.success_history = []
        
        self.monitor.log_info("ControlledRandomness initialized")
    
    def should_explore(self):
        """
        Determine if the system should explore using randomness.
        
        Returns:
            bool: True if exploration should be performed, False otherwise.
        """
        # Generate a random value and compare to the current exploration rate
        return np.random.random() < self.current_exploration_rate
    
    def generate_action(self, observation, abstract_concepts):
        """
        Generate an exploratory action using controlled randomness.
        
        Args:
            observation: The current observation.
            abstract_concepts: Abstract concepts derived from observations.
            
        Returns:
            dict: The generated exploratory action.
        """
        # Select a randomness strategy based on weights
        strategy = self._select_strategy()
        
        # Generate a random action using the selected strategy
        action = self._apply_strategy(strategy, observation, abstract_concepts)
        
        # Ensure action is within safety constraints
        if not self.safety.is_safe_action(action):
            self.monitor.log_warning(f"Generated unsafe action using {strategy}, generating fallback")
            action = self._generate_safe_fallback(observation)
        
        # Record the exploration for learning
        self._record_exploration(strategy, action)
        
        self.monitor.log_info(f"Generated exploratory action using {strategy} strategy")
        
        return action
    
    def _select_strategy(self):
        """
        Select a randomness strategy based on weights.
        
        Returns:
            str: The selected strategy name.
        """
        # Extract strategy names and weights
        strategies = list(self.strategies.keys())
        weights = [self.strategies[s]["weight"] for s in strategies]
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Select strategy
        return np.random.choice(strategies, p=normalized_weights)
    
    def _apply_strategy(self, strategy_name, observation, abstract_concepts):
        """
        Apply the selected randomness strategy.
        
        Args:
            strategy_name (str): The name of the strategy to apply.
            observation: The current observation.
            abstract_concepts: Abstract concepts derived from observations.
            
        Returns:
            dict: The generated action.
        """
        strategy = self.strategies[strategy_name]
        params = strategy["params"]
        
        if strategy_name == "gaussian_noise":
            # Add Gaussian noise to some aspect of the observation
            action = self._apply_gaussian_noise(observation, params["mean"], params["std"])
        
        elif strategy_name == "uniform_sampling":
            # Generate a uniformly random value in a range
            action = self._apply_uniform_sampling(params["low"], params["high"])
        
        elif strategy_name == "categorical_sampling":
            # Randomly select from categorical options
            action = self._apply_categorical_sampling(params["categories"])
        
        elif strategy_name == "random_mutation":
            # Mutate parts of the observation or abstract concepts
            action = self._apply_random_mutation(observation, abstract_concepts, params["mutation_rate"])
        
        elif strategy_name == "recombination":
            # Recombine parts of the abstract concepts
            action = self._apply_recombination(abstract_concepts, params["crossover_points"])
        
        else:
            self.monitor.log_warning(f"Unknown strategy: {strategy_name}")
            action = {"type": "default", "value": 0.0}
        
        # Add metadata to the action
        action["strategy"] = strategy_name
        action["exploration_rate"] = self.current_exploration_rate
        
        return action
    
    def _apply_gaussian_noise(self, observation, mean, std):
        """
        Apply Gaussian noise to observation.
        
        Args:
            observation: The current observation.
            mean (float): Mean of the Gaussian distribution.
            std (float): Standard deviation of the Gaussian distribution.
            
        Returns:
            dict: Action with Gaussian noise.
        """
        # In a real system, this would add noise to specific aspects of the observation
        # For this simulation, we'll generate random values
        
        # Generate random values
        noise_values = {}
        for i in range(5):  # Generate 5 noise values
            noise_values[f"dim_{i}"] = np.random.normal(mean, std)
        
        return {
            "type": "noise",
            "subtype": "gaussian",
            "values": noise_values,
            "magnitude": np.linalg.norm(list(noise_values.values()))
        }
    
    def _apply_uniform_sampling(self, low, high):
        """
        Apply uniform sampling.
        
        Args:
            low (float): Lower bound.
            high (float): Upper bound.
            
        Returns:
            dict: Action with uniform samples.
        """
        # Generate random uniform values
        sample_values = {}
        for i in range(5):  # Generate 5 uniform samples
            sample_values[f"dim_{i}"] = np.random.uniform(low, high)
        
        return {
            "type": "sampling",
            "subtype": "uniform",
            "values": sample_values,
            "range": [low, high]
        }
    
    def _apply_categorical_sampling(self, categories):
        """
        Apply categorical sampling.
        
        Args:
            categories (list): List of category options.
            
        Returns:
            dict: Action with categorical samples.
        """
        # Randomly select categories
        selections = {}
        for i in range(3):  # Make 3 selections
            selections[f"choice_{i}"] = np.random.choice(categories)
        
        return {
            "type": "sampling",
            "subtype": "categorical",
            "selections": selections,
            "categories": categories
        }
    
    def _apply_random_mutation(self, observation, abstract_concepts, mutation_rate):
        """
        Apply random mutations to observation or abstract concepts.
        
        Args:
            observation: The current observation.
            abstract_concepts: Abstract concepts derived from observations.
            mutation_rate (float): Rate of mutation.
            
        Returns:
            dict: Action with mutations.
        """
        # In a real system, this would mutate specific aspects of the observation or concepts
        # For this simulation, we'll generate random mutations
        
        mutations = {}
        
        # Select random targets for mutation
        if len(abstract_concepts) > 0 and np.random.random() < 0.7:
            # Mutate abstract concepts
            target_concept = np.random.choice(abstract_concepts)
            target_type = "concept"
            
            # Select properties to mutate
            if isinstance(target_concept, dict):
                properties = list(target_concept.keys())
                target_props = np.random.choice(
                    properties, 
                    size=max(1, int(len(properties) * mutation_rate)),
                    replace=False
                )
                
                for prop in target_props:
                    if prop not in ["name", "level", "type"]:  # Don't mutate metadata
                        mutations[prop] = np.random.random()
            
        else:
            # Mutate observation
            target_type = "observation"
            
            # Generate random mutations
            for i in range(int(5 * mutation_rate)):
                mutations[f"prop_{i}"] = np.random.random()
        
        return {
            "type": "mutation",
            "target_type": target_type,
            "mutations": mutations,
            "mutation_rate": mutation_rate
        }
    
    def _apply_recombination(self, abstract_concepts, crossover_points):
        """
        Apply recombination to abstract concepts.
        
        Args:
            abstract_concepts: Abstract concepts derived from observations.
            crossover_points (int): Number of crossover points.
            
        Returns:
            dict: Action with recombinations.
        """
        # In a real system, this would recombine aspects of different concepts
        # For this simulation, we'll simulate recombinations
        
        recombinations = {}
        
        if len(abstract_concepts) >= 2:
            # Need at least 2 concepts to recombine
            concept1, concept2 = np.random.choice(abstract_concepts, size=2, replace=False)
            
            # Simulate recombination
            recombinations["parent1"] = str(concept1.get("name", "unknown"))
            recombinations["parent2"] = str(concept2.get("name", "unknown"))
            recombinations["crossover_points"] = crossover_points
            
            # Create a fictional recombined result
            recombinations["result"] = f"recombination_of_{concept1.get('name', 'c1')}_{concept2.get('name', 'c2')}"
        
        return {
            "type": "recombination",
            "recombinations": recombinations,
            "crossover_points": crossover_points
        }
    
    def _generate_safe_fallback(self, observation):
        """
        Generate a safe fallback action when the main action is unsafe.
        
        Args:
            observation: The current observation.
            
        Returns:
            dict: A safe fallback action.
        """
        # This would generate a guaranteed safe action in a real system
        # For this simulation, we'll use a simple fallback
        
        return {
            "type": "fallback",
            "value": 0.0,
            "reason": "safety_constraint_violation"
        }
    
    def _record_exploration(self, strategy, action):
        """
        Record an exploration for later learning.
        
        Args:
            strategy (str): The strategy used.
            action (dict): The action generated.
        """
        self.exploration_history.append({
            "timestamp": self.monitor.get_timestamp(),
            "strategy": strategy,
            "action": action,
            "success": None  # Will be updated later
        })
        
        # Limit history size
        if len(self.exploration_history) > 100:
            self.exploration_history.pop(0)
    
    def update(self, iteration, observation, abstract_concepts):
        """
        Update the controlled randomness system based on new information.
        
        Args:
            iteration (int): Current iteration number.
            observation: The current observation.
            abstract_concepts: Processed abstract concepts.
        """
        # Update exploration rate using decay
        self.current_exploration_rate = max(
            self.min_exploration_rate,
            self.current_exploration_rate - self.exploration_decay
        )
        
        # If we have success history, update strategy weights
        if len(self.success_history) >= 10:
            self._update_strategy_weights()
            self.success_history = []  # Reset after updating
    
    def _update_strategy_weights(self):
        """Update strategy weights based on success history."""
        strategy_success = {}
        
        # Calculate success rate for each strategy
        for entry in self.success_history:
            strategy = entry["strategy"]
            success = entry["success"]
            
            if strategy not in strategy_success:
                strategy_success[strategy] = {"successes": 0, "total": 0}
            
            strategy_success[strategy]["total"] += 1
            if success:
                strategy_success[strategy]["successes"] += 1
        
        # Update weights based on success rates
        for strategy, stats in strategy_success.items():
            if stats["total"] > 0:
                success_rate = stats["successes"] / stats["total"]
                
                # Adjust weight (increase for successful strategies)
                current_weight = self.strategies[strategy]["weight"]
                new_weight = current_weight * (1.0 + 0.2 * (success_rate - 0.5))
                
                # Ensure weight stays reasonable
                self.strategies[strategy]["weight"] = max(0.05, min(0.5, new_weight))
        
        # Normalize weights
        total_weight = sum(s["weight"] for s in self.strategies.values())
        for strategy in self.strategies:
            self.strategies[strategy]["weight"] /= total_weight
    
    def learn_from_solution(self, problem, solution):
        """
        Learn from a successful solution.
        
        Args:
            problem: The problem that was solved.
            solution: The solution that was found.
        """
        # Check if any recent explorations contributed to the solution
        # This would involve sophisticated analysis in a real system
        # For this simulation, we'll randomly mark some explorations as successful
        
        for i, entry in enumerate(reversed(self.exploration_history[:10])):
            # Mark as successful with decreasing probability based on recency
            success = np.random.random() < (0.8 - 0.07 * i)
            
            # Update the success status
            self.exploration_history[-(i+1)]["success"] = success
            
            # Add to success history for weight updates
            self.success_history.append({
                "strategy": entry["strategy"],
                "success": success
            })

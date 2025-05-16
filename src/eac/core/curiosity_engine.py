"""
Curiosity Engine Module

This module implements the internal curiosity engine component of the EAC system,
which drives exploration and self-improvement beyond explicit task objectives.
"""

import numpy as np
from eac.utils.monitoring import Monitor


class CuriosityEngine:
    """
    The Curiosity Engine component drives the system's intrinsic motivation,
    exploration of new approaches, and self-improvement initiatives.
    """
    
    def __init__(self, monitor):
        """
        Initialize the Curiosity Engine component.
        
        Args:
            monitor (Monitor): Monitoring system for tracking operations.
        """
        self.monitor = monitor
        
        # Curiosity parameters
        self.novelty_threshold = 0.6  # Threshold for considering something novel
        self.exploration_rate = 0.3   # Base rate of exploration
        self.curiosity_decay = 0.01   # Rate at which curiosity decays for familiar things
        
        # References to other components (will be set via register_components)
        self.recursive_architecture = None
        self.abstraction_learning = None
        self.controlled_randomness = None
        
        # Knowledge state
        self.known_patterns = {}     # Patterns the system has recognized
        self.exploration_history = []  # History of exploration attempts
        self.curiosity_scores = {}   # Curiosity scores for different domains
        
        # Initialize curiosity for different domains
        self._initialize_curiosity_domains()
        
        self.monitor.log_info("CuriosityEngine initialized")
    
    def _initialize_curiosity_domains(self):
        """Initialize curiosity scores for different domains."""
        domains = [
            "problem_structure",
            "solution_approaches",
            "abstraction_levels",
            "architecture_modifications",
            "reasoning_methods",
            "data_patterns",
            "causal_relationships"
        ]
        
        for domain in domains:
            # Higher value means more curiosity about this domain
            self.curiosity_scores[domain] = np.random.uniform(0.5, 1.0)
    
    def register_components(self, recursive_architecture, abstraction_learning, controlled_randomness):
        """
        Register other EAC components for cross-component interactions.
        
        Args:
            recursive_architecture: The recursive architecture component.
            abstraction_learning: The abstraction learning component.
            controlled_randomness: The controlled randomness component.
        """
        self.recursive_architecture = recursive_architecture
        self.abstraction_learning = abstraction_learning
        self.controlled_randomness = controlled_randomness
        
        self.monitor.log_info("Components registered with CuriosityEngine")
    
    def generate_goals(self, observation):
        """
        Generate curiosity-driven exploration goals based on the current observation.
        
        Args:
            observation: The current observation.
            
        Returns:
            list: Exploration goals.
        """
        # Assess the novelty of the observation
        novelty = self._assess_novelty(observation)
        
        # Update curiosity based on novelty
        self._update_curiosity(novelty)
        
        # Generate exploration goals based on current curiosity
        goals = self._create_exploration_goals()
        
        self.monitor.log_info(f"Generated {len(goals)} exploration goals")
        
        return goals
    
    def _assess_novelty(self, observation):
        """
        Assess the novelty of an observation.
        
        Args:
            observation: The current observation.
            
        Returns:
            dict: Novelty scores for different aspects of the observation.
        """
        # In a real system, this would involve comparing the observation to known patterns
        # For this simulation, we'll use random values
        novelty = {}
        for domain in self.curiosity_scores:
            # Higher value means more novel
            novelty[domain] = np.random.beta(2, 5)  # Biased toward lower values
            
            # Occasionally generate high novelty to simulate discovery
            if np.random.random() < 0.05:  # 5% chance
                novelty[domain] = np.random.uniform(0.8, 1.0)
        
        return novelty
    
    def _update_curiosity(self, novelty):
        """
        Update curiosity scores based on novelty assessment.
        
        Args:
            novelty (dict): Novelty scores for different domains.
        """
        for domain, novelty_score in novelty.items():
            if novelty_score > self.novelty_threshold:
                # Increase curiosity for novel domains
                self.curiosity_scores[domain] = min(
                    1.0, 
                    self.curiosity_scores[domain] + (novelty_score - self.novelty_threshold)
                )
            else:
                # Gradually decrease curiosity for familiar domains
                self.curiosity_scores[domain] = max(
                    0.1,
                    self.curiosity_scores[domain] - self.curiosity_decay
                )
    
    def _create_exploration_goals(self):
        """
        Create specific exploration goals based on current curiosity.
        
        Returns:
            list: Exploration goals.
        """
        goals = []
        
        # For each domain with high curiosity, generate a specific goal
        for domain, curiosity_score in self.curiosity_scores.items():
            if curiosity_score > self.exploration_rate:
                # Create a goal for this domain
                goal = self._generate_domain_goal(domain, curiosity_score)
                goals.append(goal)
        
        return goals
    
    def _generate_domain_goal(self, domain, curiosity_score):
        """
        Generate a specific goal for a domain.
        
        Args:
            domain (str): The domain to generate a goal for.
            curiosity_score (float): The curiosity score for the domain.
            
        Returns:
            dict: A specific exploration goal.
        """
        # Each domain has different types of goals
        if domain == "problem_structure":
            goal_types = [
                "identify_hidden_variables",
                "discover_problem_categories",
                "map_constraint_relationships"
            ]
        elif domain == "solution_approaches":
            goal_types = [
                "test_alternative_algorithm",
                "combine_existing_methods",
                "invent_new_approach"
            ]
        elif domain == "architecture_modifications":
            goal_types = [
                "optimize_module_connections",
                "add_specialized_component",
                "reorganize_information_flow"
            ]
        else:
            goal_types = [
                "explore_edge_cases",
                "find_counterexamples",
                "identify_patterns"
            ]
        
        goal_type = np.random.choice(goal_types)
        
        return {
            "domain": domain,
            "type": goal_type,
            "priority": curiosity_score,
            "description": f"{goal_type.replace('_', ' ').capitalize()} in {domain.replace('_', ' ')}"
        }
    
    def update(self, iteration, observation, abstract_concepts):
        """
        Update the curiosity engine based on new information.
        
        Args:
            iteration (int): Current iteration number.
            observation: The current observation.
            abstract_concepts: Processed abstract concepts.
        """
        # Periodically decay all curiosity slightly to prevent fixation
        if iteration % 10 == 0:
            for domain in self.curiosity_scores:
                self.curiosity_scores[domain] = max(
                    0.1,
                    self.curiosity_scores[domain] - 0.05
                )
        
        # Update knowledge state based on abstract concepts
        for concept in abstract_concepts:
            concept_key = str(concept)  # Convert to string for dict key
            if concept_key not in self.known_patterns:
                self.known_patterns[concept_key] = {
                    "first_seen": iteration,
                    "occurrences": 1
                }
            else:
                self.known_patterns[concept_key]["occurrences"] += 1
    
    def learn_from_solution(self, problem, solution):
        """
        Learn from a successful solution.
        
        Args:
            problem: The problem that was solved.
            solution: The solution that was found.
        """
        # Analyze the solution to update the curiosity model
        # Record what worked and what didn't to guide future exploration
        
        # In a real system, this would involve detailed analysis
        # For this simulation, we'll just record the event
        self.exploration_history.append({
            "problem": str(problem),
            "solution": str(solution),
            "timestamp": self.monitor.get_timestamp()
        })

"""
Abstraction Learning Module

This module implements the multi-level abstraction learning component of the EAC system,
which enables hierarchical concept formation and cross-domain knowledge transfer.
"""

import numpy as np
from collections import defaultdict
from eac.utils.monitoring import Monitor


class AbstractionLearning:
    """
    The Abstraction Learning component enables the EAC system to form and
    manipulate concepts at multiple levels of abstraction simultaneously.
    """
    
    def __init__(self, monitor):
        """
        Initialize the Abstraction Learning component.
        
        Args:
            monitor (Monitor): Monitoring system for tracking operations.
        """
        self.monitor = monitor
        
        # Abstraction parameters
        self.num_abstraction_levels = 5  # Number of abstraction levels
        self.abstraction_threshold = 0.65  # Threshold for forming new abstractions
        
        # Concept storage structure (organized by abstraction level)
        self.concepts = defaultdict(dict)
        
        # Relationships between concepts (across and within levels)
        self.concept_relationships = defaultdict(list)
        
        # Performance metrics
        self.abstraction_success_rate = 0.0
        
        self.monitor.log_info("AbstractionLearning initialized")
    
    def process(self, observation):
        """
        Process an observation into abstract concepts at multiple levels.
        
        Args:
            observation: The observation to process.
            
        Returns:
            list: Abstracted concepts derived from the observation.
        """
        # Extract features from the observation
        features = self._extract_features(observation)
        
        # Form concrete concepts (level 0)
        concrete_concepts = self._form_concrete_concepts(features)
        
        # Generate higher level abstractions
        abstractions = [concrete_concepts]  # Start with concrete concepts
        
        # For each additional abstraction level
        for level in range(1, self.num_abstraction_levels):
            # Generate abstractions based on the previous level
            level_abstractions = self._generate_abstractions(abstractions[-1], level)
            abstractions.append(level_abstractions)
        
        # Identify relationships between concepts at different levels
        self._identify_relationships(abstractions)
        
        # Flatten the abstractions for return
        all_abstractions = []
        for level_abstractions in abstractions:
            all_abstractions.extend(level_abstractions)
        
        self.monitor.log_info(f"Processed observation into {len(all_abstractions)} abstract concepts")
        
        return all_abstractions
    
    def _extract_features(self, observation):
        """
        Extract relevant features from an observation.
        
        Args:
            observation: The observation to extract features from.
            
        Returns:
            dict: Extracted features.
        """
        # In a real system, this would involve sophisticated feature extraction
        # For this simulation, we'll treat the observation as already having features
        # If it's a string or simple value, we'll create some dummy features
        
        if isinstance(observation, (str, int, float, bool)):
            # Create dummy features for simple types
            return {
                "type": str(type(observation)),
                "value": str(observation),
                "length": len(str(observation)) if hasattr(observation, "__len__") else 0,
                "is_numeric": isinstance(observation, (int, float))
            }
        
        # If it's already a dict or object with attributes, use it directly
        # but convert to a dict if needed
        if hasattr(observation, "__dict__"):
            return vars(observation)
        
        return observation
    
    def _form_concrete_concepts(self, features):
        """
        Form concrete concepts (level 0) from features.
        
        Args:
            features (dict): Extracted features from the observation.
            
        Returns:
            list: Concrete concepts.
        """
        concrete_concepts = []
        
        # Convert each feature into a concrete concept
        for feature_name, feature_value in features.items():
            concept = {
                "name": f"concept_{feature_name}",
                "level": 0,
                "type": "concrete",
                "feature": feature_name,
                "value": feature_value,
                "confidence": 1.0
            }
            
            # Add to concepts storage
            self.concepts[0][concept["name"]] = concept
            
            concrete_concepts.append(concept)
        
        return concrete_concepts
    
    def _generate_abstractions(self, lower_concepts, level):
        """
        Generate abstractions at a given level based on lower-level concepts.
        
        Args:
            lower_concepts (list): Concepts from the lower abstraction level.
            level (int): The current abstraction level.
            
        Returns:
            list: Concepts at the current abstraction level.
        """
        abstractions = []
        
        # Group lower concepts by compatible features
        concept_groups = self._group_compatible_concepts(lower_concepts)
        
        # For each group, try to form an abstraction
        for group in concept_groups:
            if len(group) < 2:
                continue  # Need at least 2 concepts to form a meaningful abstraction
            
            # Try to identify a common pattern or relationship
            abstraction = self._abstract_from_group(group, level)
            
            if abstraction and abstraction.get("confidence", 0) > self.abstraction_threshold:
                # Add to concepts storage
                self.concepts[level][abstraction["name"]] = abstraction
                abstractions.append(abstraction)
        
        # Sometimes create novel abstractions through recombination
        if np.random.random() < 0.2:  # 20% chance
            novel_abstraction = self._create_novel_abstraction(lower_concepts, level)
            if novel_abstraction:
                self.concepts[level][novel_abstraction["name"]] = novel_abstraction
                abstractions.append(novel_abstraction)
        
        return abstractions
    
    def _group_compatible_concepts(self, concepts):
        """
        Group concepts that might form meaningful abstractions together.
        
        Args:
            concepts (list): Concepts to group.
            
        Returns:
            list: Groups of compatible concepts.
        """
        # In a real system, this would involve sophisticated compatibility analysis
        # For this simulation, we'll use a simple approach based on feature types
        
        # Group by concept type as a simple approach
        groups = defaultdict(list)
        for concept in concepts:
            key = concept.get("type", "unknown")
            groups[key].append(concept)
        
        # Convert to list of groups
        return list(groups.values())
    
    def _abstract_from_group(self, concept_group, level):
        """
        Form an abstraction from a group of related concepts.
        
        Args:
            concept_group (list): Group of related concepts.
            level (int): Current abstraction level.
            
        Returns:
            dict: The formed abstraction, or None if no abstraction could be formed.
        """
        # In a real system, this would involve pattern recognition and generalization
        # For this simulation, we'll use a simple approach
        
        # Extract common features
        common_features = {}
        for feature_name in concept_group[0]:
            if feature_name in ["name", "level", "confidence"]:
                continue  # Skip metadata
            
            # Check if this feature is common to all concepts in the group
            feature_values = [c.get(feature_name) for c in concept_group if feature_name in c]
            if len(feature_values) == len(concept_group):
                # Feature exists in all concepts
                # If the values are the same, it's a common feature
                if len(set(str(v) for v in feature_values)) == 1:
                    common_features[feature_name] = feature_values[0]
        
        if not common_features:
            return None  # No common features found
        
        # Create an abstraction based on common features
        abstraction_name = f"abstract_{level}_{np.random.randint(1000)}"
        
        abstraction = {
            "name": abstraction_name,
            "level": level,
            "type": "abstract",
            "common_features": common_features,
            "source_concepts": [c["name"] for c in concept_group],
            "confidence": np.random.uniform(0.6, 0.9)  # Random confidence level
        }
        
        return abstraction
    
    def _create_novel_abstraction(self, concepts, level):
        """
        Create a novel abstraction through recombination or transformation.
        
        Args:
            concepts (list): Available concepts.
            level (int): Current abstraction level.
            
        Returns:
            dict: The novel abstraction, or None if no abstraction could be created.
        """
        if not concepts:
            return None
        
        # Select a few random concepts
        selected_concepts = np.random.choice(
            concepts, 
            size=min(3, len(concepts)), 
            replace=False
        )
        
        # Create a "what-if" abstraction by modifying some features
        modified_features = {}
        for concept in selected_concepts:
            for feature_name, feature_value in concept.items():
                if feature_name not in ["name", "level", "type", "confidence"]:
                    # Add with some random modification
                    if isinstance(feature_value, (int, float)):
                        modified_features[feature_name] = feature_value * np.random.uniform(0.5, 1.5)
                    elif isinstance(feature_value, str):
                        modified_features[feature_name] = f"novel_{feature_value}"
                    else:
                        modified_features[feature_name] = feature_value
        
        # Create the novel abstraction
        abstraction_name = f"novel_{level}_{np.random.randint(1000)}"
        
        abstraction = {
            "name": abstraction_name,
            "level": level,
            "type": "novel",
            "features": modified_features,
            "inspiration_concepts": [c["name"] for c in selected_concepts],
            "confidence": np.random.uniform(0.4, 0.7)  # Lower confidence for novel abstractions
        }
        
        return abstraction
    
    def _identify_relationships(self, all_level_abstractions):
        """
        Identify relationships between concepts at different abstraction levels.
        
        Args:
            all_level_abstractions (list): Lists of abstractions at each level.
        """
        # This would be a sophisticated process in a real system
        # For this simulation, we'll focus on hierarchical relationships
        
        # For each abstraction level (except the lowest)
        for level in range(1, len(all_level_abstractions)):
            higher_concepts = all_level_abstractions[level]
            lower_concepts = all_level_abstractions[level - 1]
            
            # For each higher-level concept
            for higher in higher_concepts:
                # Look for lower-level concepts that contributed to it
                for lower in lower_concepts:
                    # Check if the lower concept contributed to the higher one
                    if "source_concepts" in higher and lower["name"] in higher["source_concepts"]:
                        # Add a hierarchical relationship
                        self.concept_relationships["hierarchical"].append({
                            "higher": higher["name"],
                            "lower": lower["name"],
                            "type": "composition",
                            "strength": np.random.uniform(0.7, 1.0)
                        })
        
        # Also look for lateral relationships within the same level
        for level_concepts in all_level_abstractions:
            for i, concept1 in enumerate(level_concepts):
                for concept2 in level_concepts[i+1:]:
                    # Check for potential lateral relationships
                    if np.random.random() < 0.3:  # 30% chance of relationship
                        relationship_type = np.random.choice([
                            "similarity", "contrast", "association"
                        ])
                        
                        self.concept_relationships["lateral"].append({
                            "concept1": concept1["name"],
                            "concept2": concept2["name"],
                            "type": relationship_type,
                            "strength": np.random.uniform(0.5, 0.9)
                        })
    
    def update(self, iteration, observation, abstract_concepts):
        """
        Update the abstraction learning system based on new information.
        
        Args:
            iteration (int): Current iteration number.
            observation: The current observation.
            abstract_concepts: Processed abstract concepts.
        """
        # In a real system, this would involve updating concept models
        # For this simulation, we'll just track performance
        
        # Occasionally adjust abstraction parameters based on performance
        if iteration % 10 == 0:
            # Adjust abstraction threshold based on success rate
            if self.abstraction_success_rate > 0.8:
                # If we're doing well, increase the threshold to form more precise abstractions
                self.abstraction_threshold = min(0.9, self.abstraction_threshold + 0.02)
            elif self.abstraction_success_rate < 0.4:
                # If we're doing poorly, decrease the threshold to form more abstractions
                self.abstraction_threshold = max(0.4, self.abstraction_threshold - 0.02)
    
    def learn_from_solution(self, problem, solution):
        """
        Learn from a successful solution by updating abstraction models.
        
        Args:
            problem: The problem that was solved.
            solution: The solution that was found.
        """
        # Extract abstractions from the problem and solution
        problem_abstractions = self.process(problem)
        solution_abstractions = self.process(solution)
        
        # Learn relationships between problem and solution abstractions
        for p_abs in problem_abstractions:
            for s_abs in solution_abstractions:
                if np.random.random() < 0.4:  # 40% chance of relationship
                    self.concept_relationships["problem_solution"].append({
                        "problem_concept": p_abs["name"],
                        "solution_concept": s_abs["name"],
                        "type": "solution_mapping",
                        "strength": np.random.uniform(0.6, 0.95)
                    })
        
        # Update success rate
        self.abstraction_success_rate = 0.9 * self.abstraction_success_rate + 0.1

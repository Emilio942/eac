"""
Safety Constraints Module

This module implements safety constraints for the EAC architecture,
ensuring safe operation during self-modification and exploration.
"""

import numpy as np


class SafetyConstraints:
    """
    The SafetyConstraints class enforces safety constraints on the EAC system,
    preventing dangerous actions and modifications.
    """
    
    def __init__(self, safety_level="high"):
        """
        Initialize SafetyConstraints.
        
        Args:
            safety_level (str): Level of safety enforcement ("low", "medium", "high").
        """
        self.safety_level = safety_level
        
        # Set threshold based on safety level
        self.thresholds = {
            "low": 0.7,
            "medium": 0.8,
            "high": 0.9
        }
        
        # Initialize safety constraint checkers
        self._init_constraint_checkers()
        
        # Safety violation counter
        self.violation_counts = {}
    
    def _init_constraint_checkers(self):
        """Initialize the constraint checking functions."""
        self.modification_constraints = [
            self._check_architectural_integrity,
            self._check_operational_stability,
            self._check_modification_scope,
            self._check_rollback_capability
        ]
        
        self.action_constraints = [
            self._check_action_safety,
            self._check_action_reversibility,
            self._check_resource_usage
        ]
    
    def is_safe_modification(self, modification_plan):
        """
        Check if an architectural modification is safe.
        
        Args:
            modification_plan (dict): The modification plan to evaluate.
            
        Returns:
            bool: True if the modification is safe, False otherwise.
        """
        safety_scores = []
        
        # Run all constraint checks
        for constraint_check in self.modification_constraints:
            score = constraint_check(modification_plan)
            safety_scores.append(score)
        
        # Calculate aggregate safety score
        if not safety_scores:
            return False
        
        aggregate_score = np.mean(safety_scores)
        threshold = self.thresholds.get(self.safety_level, 0.9)
        
        # Record violation if unsafe
        is_safe = aggregate_score >= threshold
        if not is_safe:
            self._record_violation("modification", modification_plan.get("type", "unknown"))
        
        return is_safe
    
    def is_safe_action(self, action):
        """
        Check if an action is safe.
        
        Args:
            action (dict): The action to evaluate.
            
        Returns:
            bool: True if the action is safe, False otherwise.
        """
        safety_scores = []
        
        # Run all constraint checks
        for constraint_check in self.action_constraints:
            score = constraint_check(action)
            safety_scores.append(score)
        
        # Calculate aggregate safety score
        if not safety_scores:
            return False
        
        aggregate_score = np.mean(safety_scores)
        threshold = self.thresholds.get(self.safety_level, 0.9)
        
        # Record violation if unsafe
        is_safe = aggregate_score >= threshold
        if not is_safe:
            self._record_violation("action", action.get("type", "unknown"))
        
        return is_safe
    
    def _record_violation(self, category, violation_type):
        """
        Record a safety violation for analysis.
        
        Args:
            category (str): Violation category.
            violation_type (str): Type of violation.
        """
        key = f"{category}_{violation_type}"
        
        if key not in self.violation_counts:
            self.violation_counts[key] = 0
        
        self.violation_counts[key] += 1
    
    def _check_architectural_integrity(self, modification_plan):
        """
        Check if a modification preserves architectural integrity.
        
        Args:
            modification_plan (dict): The modification plan to evaluate.
            
        Returns:
            float: Safety score (0-1).
        """
        # In a real system, this would involve sophisticated integrity checking
        # For this simulation, we'll use a heuristic
        
        mod_type = modification_plan.get("type", "unknown")
        target = modification_plan.get("target", "unknown")
        risk = modification_plan.get("risk_assessment", 0.5)
        
        if mod_type == "add":
            # Adding is generally safer than modifying or removing
            integrity_score = 0.9 - (0.3 * risk)
        elif mod_type == "improve":
            # Improving existing modules has moderate risk
            integrity_score = 0.8 - (0.4 * risk)
        elif mod_type == "reorganize":
            # Reorganizing has higher risk
            integrity_score = 0.7 - (0.5 * risk)
        else:
            # Unknown modifications are risky
            integrity_score = 0.5
        
        return max(0.0, min(1.0, integrity_score))
    
    def _check_operational_stability(self, modification_plan):
        """
        Check if a modification maintains operational stability.
        
        Args:
            modification_plan (dict): The modification plan to evaluate.
            
        Returns:
            float: Safety score (0-1).
        """
        # This would check if core functionality remains stable
        
        # For simulation, we'll use a heuristic
        expected_impact = modification_plan.get("expected_impact", {})
        performance_impact = expected_impact.get("performance", 0.0)
        
        # Higher impact means potentially lower stability
        stability_score = 1.0 - (performance_impact * 0.5)
        
        return max(0.0, min(1.0, stability_score))
    
    def _check_modification_scope(self, modification_plan):
        """
        Check if the modification scope is reasonable.
        
        Args:
            modification_plan (dict): The modification plan to evaluate.
            
        Returns:
            float: Safety score (0-1).
        """
        # This would check if the modification affects too many components
        
        # For simulation, we'll use a simple heuristic
        target = modification_plan.get("target", "unknown")
        
        if target == "all":
            # Modifying all modules is risky
            scope_score = 0.6
        else:
            # Modifying a single module is safer
            scope_score = 0.9
        
        # Adjust based on risk assessment
        risk = modification_plan.get("risk_assessment", 0.5)
        scope_score -= (risk * 0.3)
        
        return max(0.0, min(1.0, scope_score))
    
    def _check_rollback_capability(self, modification_plan):
        """
        Check if the modification can be rolled back if needed.
        
        Args:
            modification_plan (dict): The modification plan to evaluate.
            
        Returns:
            float: Safety score (0-1).
        """
        # This would check if we can roll back the modification
        
        # For simulation, assume most modifications can be rolled back
        mod_type = modification_plan.get("type", "unknown")
        
        if mod_type == "reorganize":
            # Reorganizations might be harder to roll back
            rollback_score = 0.7
        else:
            # Other modifications are easier to roll back
            rollback_score = 0.9
        
        return rollback_score
    
    def _check_action_safety(self, action):
        """
        Check if an action is inherently safe.
        
        Args:
            action (dict): The action to evaluate.
            
        Returns:
            float: Safety score (0-1).
        """
        # This would check if the action has inherently safe properties
        
        # For simulation, we'll use a heuristic based on action type
        action_type = action.get("type", "unknown")
        
        if action_type == "fallback":
            # Fallback actions are designed to be safe
            return 0.95
        elif action_type == "noise":
            # Adding noise has moderate risk
            magnitude = action.get("magnitude", 0.5)
            return 0.9 - (magnitude * 0.3)
        elif action_type == "sampling":
            # Sampling has moderate risk
            return 0.85
        elif action_type == "mutation":
            # Mutations have higher risk
            mutation_rate = action.get("mutation_rate", 0.5)
            return 0.8 - (mutation_rate * 0.4)
        elif action_type == "recombination":
            # Recombinations have higher risk
            return 0.75
        else:
            # Unknown action types are treated as moderate risk
            return 0.7
    
    def _check_action_reversibility(self, action):
        """
        Check if an action can be reversed if needed.
        
        Args:
            action (dict): The action to evaluate.
            
        Returns:
            float: Safety score (0-1).
        """
        # This would check if the action can be undone
        
        # For simulation, assume most actions can be reversed
        action_type = action.get("type", "unknown")
        
        if action_type in ["noise", "sampling", "fallback"]:
            # These actions are easy to reverse (just don't apply them again)
            return 0.95
        elif action_type in ["mutation", "recombination"]:
            # These actions might be harder to reverse
            return 0.8
        else:
            # Unknown action types are treated as moderately reversible
            return 0.85
    
    def _check_resource_usage(self, action):
        """
        Check if the action uses resources responsibly.
        
        Args:
            action (dict): The action to evaluate.
            
        Returns:
            float: Safety score (0-1).
        """
        # This would check if the action uses excessive resources
        
        # For simulation, assume most actions use reasonable resources
        return 0.9
    
    def update_safety_level(self, level):
        """
        Update the safety level.
        
        Args:
            level (str): New safety level ("low", "medium", "high").
            
        Returns:
            bool: True if updated successfully, False otherwise.
        """
        if level in self.thresholds:
            self.safety_level = level
            return True
        return False
    
    def get_violation_summary(self):
        """
        Get a summary of safety violations.
        
        Returns:
            dict: Violation summary.
        """
        return self.violation_counts.copy()

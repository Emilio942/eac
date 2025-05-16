"""
Monitoring Module

This module implements the monitoring system for the EAC architecture,
tracking operations, logging events, and maintaining performance metrics.
"""

import time
import json
from datetime import datetime


class Monitor:
    """
    The Monitor class handles logging, metrics tracking, and event recording
    for the EAC system.
    """
    
    def __init__(self, log_level="INFO"):
        """
        Initialize the Monitor.
        
        Args:
            log_level (str): Minimum log level to record.
        """
        self.log_level = log_level
        self.log_levels = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4
        }
        
        # Initialize logs
        self.logs = []
        
        # Initialize metrics
        self.metrics = {
            "architecture_modifications": 0,
            "successful_modifications": 0,
            "failed_modifications": 0,
            "new_abstractions_formed": 0,
            "exploration_actions": 0,
            "problems_solved": 0,
            "solution_efficiency": 0.0
        }
        
        # Initialize event listeners
        self.event_listeners = {}
        
        self.log_info("Monitor initialized")
    
    def log(self, level, message):
        """
        Log a message at the specified level.
        
        Args:
            level (str): Log level.
            message (str): Message to log.
        """
        if self.log_levels.get(level, 0) >= self.log_levels.get(self.log_level, 0):
            log_entry = {
                "timestamp": self.get_timestamp(),
                "level": level,
                "message": message
            }
            
            self.logs.append(log_entry)
            
            # Keep log size manageable
            if len(self.logs) > 1000:
                self.logs = self.logs[-1000:]
            
            # Print to console (would be more sophisticated in a real system)
            print(f"[{log_entry['timestamp']}] {level}: {message}")
    
    def log_debug(self, message):
        """Log a debug message."""
        self.log("DEBUG", message)
    
    def log_info(self, message):
        """Log an info message."""
        self.log("INFO", message)
    
    def log_warning(self, message):
        """Log a warning message."""
        self.log("WARNING", message)
    
    def log_error(self, message):
        """Log an error message."""
        self.log("ERROR", message)
    
    def log_critical(self, message):
        """Log a critical message."""
        self.log("CRITICAL", message)
    
    def get_timestamp(self):
        """
        Get a formatted timestamp.
        
        Returns:
            str: Formatted timestamp.
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def update_metrics(self, iteration, observation, abstract_concepts):
        """
        Update performance metrics.
        
        Args:
            iteration (int): Current iteration number.
            observation: The current observation.
            abstract_concepts: Processed abstract concepts.
        """
        # Update iteration-based metrics
        # In a real system, this would involve sophisticated metric calculations
        
        # Count new abstractions formed
        self.metrics["new_abstractions_formed"] += len([
            c for c in abstract_concepts if c.get("timestamp", "") == self.get_timestamp()
        ])
    
    def record_architecture_modification(self, success):
        """
        Record an architecture modification.
        
        Args:
            success (bool): Whether the modification was successful.
        """
        self.metrics["architecture_modifications"] += 1
        
        if success:
            self.metrics["successful_modifications"] += 1
        else:
            self.metrics["failed_modifications"] += 1
    
    def record_exploration_action(self):
        """Record an exploration action."""
        self.metrics["exploration_actions"] += 1
    
    def record_problem_solved(self, efficiency):
        """
        Record a solved problem.
        
        Args:
            efficiency (float): Efficiency of the solution (0-1).
        """
        self.metrics["problems_solved"] += 1
        
        # Update rolling average of solution efficiency
        self.metrics["solution_efficiency"] = (
            0.9 * self.metrics["solution_efficiency"] + 0.1 * efficiency
        )
    
    def register_event_listener(self, event_type, callback):
        """
        Register a listener for a specific event type.
        
        Args:
            event_type (str): Type of event to listen for.
            callback (function): Function to call when event occurs.
        """
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        
        self.event_listeners[event_type].append(callback)
        self.log_debug(f"Registered listener for event type: {event_type}")
    
    def emit_event(self, event_type, data):
        """
        Emit an event to registered listeners.
        
        Args:
            event_type (str): Type of event.
            data: Event data.
        """
        if event_type in self.event_listeners:
            for callback in self.event_listeners[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    self.log_error(f"Error in event listener for {event_type}: {e}")
    
    def get_metrics_summary(self):
        """
        Get a summary of current metrics.
        
        Returns:
            dict: Metrics summary.
        """
        return {
            "timestamp": self.get_timestamp(),
            "metrics": self.metrics.copy()
        }
    
    def get_recent_logs(self, level=None, limit=100):
        """
        Get recent logs, optionally filtered by level.
        
        Args:
            level (str, optional): Log level to filter by.
            limit (int): Maximum number of logs to return.
            
        Returns:
            list: Recent logs.
        """
        if level:
            filtered_logs = [log for log in self.logs if log["level"] == level]
            return filtered_logs[-limit:]
        
        return self.logs[-limit:]
    
    def export_logs(self, file_path):
        """
        Export logs to a file.
        
        Args:
            file_path (str): Path to export logs to.
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.logs, f, indent=2)
            
            self.log_info(f"Logs exported to {file_path}")
            return True
        except Exception as e:
            self.log_error(f"Failed to export logs: {e}")
            return False
    
    def export_metrics(self, file_path):
        """
        Export metrics to a file.
        
        Args:
            file_path (str): Path to export metrics to.
        """
        try:
            metrics_data = {
                "timestamp": self.get_timestamp(),
                "metrics": self.metrics
            }
            
            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self.log_info(f"Metrics exported to {file_path}")
            return True
        except Exception as e:
            self.log_error(f"Failed to export metrics: {e}")
            return False

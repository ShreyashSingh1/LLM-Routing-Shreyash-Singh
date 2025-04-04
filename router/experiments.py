"""Experiments module for A/B testing different routing strategies."""

import random
import time
from typing import Dict, Any, List, Callable, Optional
import pandas as pd
import numpy as np
from collections import defaultdict

from config import ROUTING_CONFIG


class ExperimentManager:
    """Manages A/B testing experiments for comparing routing strategies."""
    
    def __init__(self):
        """Initialize the experiment manager."""
        self.experiments = {}
        self.active_experiment = None
        self.metrics = defaultdict(list)
        self.strategy_registry = {}
        
    def register_strategy(self, name: str, strategy_fn: Callable) -> None:
        """Register a routing strategy function.
        
        Args:
            name: Unique name for the strategy
            strategy_fn: Function that implements the routing strategy
        """
        self.strategy_registry[name] = strategy_fn
        
    def create_experiment(self, name: str, strategies: List[str], 
                         traffic_split: Optional[Dict[str, float]] = None) -> None:
        """Create a new A/B testing experiment.
        
        Args:
            name: Name of the experiment
            strategies: List of strategy names to compare
            traffic_split: Optional dictionary mapping strategy names to traffic percentages
                          (must sum to 1.0). If None, traffic will be split evenly.
        """
        # Validate strategies
        for strategy in strategies:
            if strategy not in self.strategy_registry:
                raise ValueError(f"Strategy '{strategy}' is not registered")
        
        # Create default even split if not provided
        if traffic_split is None:
            split_value = 1.0 / len(strategies)
            traffic_split = {strategy: split_value for strategy in strategies}
        
        # Validate traffic split
        if sum(traffic_split.values()) != 1.0:
            raise ValueError("Traffic split must sum to 1.0")
        
        # Create experiment
        self.experiments[name] = {
            "strategies": strategies,
            "traffic_split": traffic_split,
            "created_at": time.time(),
            "status": "created"
        }
        
    def start_experiment(self, name: str) -> None:
        """Start an experiment.
        
        Args:
            name: Name of the experiment to start
        """
        if name not in self.experiments:
            raise ValueError(f"Experiment '{name}' does not exist")
        
        self.experiments[name]["status"] = "active"
        self.experiments[name]["started_at"] = time.time()
        self.active_experiment = name
        
    def stop_experiment(self, name: str) -> None:
        """Stop an experiment.
        
        Args:
            name: Name of the experiment to stop
        """
        if name not in self.experiments:
            raise ValueError(f"Experiment '{name}' does not exist")
        
        self.experiments[name]["status"] = "completed"
        self.experiments[name]["stopped_at"] = time.time()
        
        if self.active_experiment == name:
            self.active_experiment = None
    
    def get_strategy(self, query: Dict[str, Any]) -> str:
        """Get the strategy to use for a given query based on active experiment.
        
        Args:
            query: The query to route
            
        Returns:
            Name of the selected strategy
        """
        if not self.active_experiment:
            # Return default strategy if no active experiment
            return list(self.strategy_registry.keys())[0] if self.strategy_registry else None
        
        experiment = self.experiments[self.active_experiment]
        strategies = experiment["strategies"]
        traffic_split = experiment["traffic_split"]
        
        # Deterministic assignment based on query hash to ensure consistent routing
        # for the same query during an experiment
        query_str = str(query.get("query", ""))
        query_hash = hash(query_str) % 1000 / 1000.0
        
        # Select strategy based on traffic split
        cumulative = 0.0
        for strategy, percentage in traffic_split.items():
            cumulative += percentage
            if query_hash < cumulative:
                return strategy
        
        # Fallback to last strategy
        return strategies[-1]
    
    def record_metrics(self, strategy: str, query: str, metrics: Dict[str, Any]) -> None:
        """Record performance metrics for a strategy.
        
        Args:
            strategy: Name of the strategy used
            query: The query that was processed
            metrics: Dictionary of metrics to record
        """
        record = {
            "strategy": strategy,
            "query": query,
            "timestamp": time.time(),
            **metrics
        }
        
        if self.active_experiment:
            record["experiment"] = self.active_experiment
        
        self.metrics[strategy].append(record)
    
    def get_experiment_results(self, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """Get results for an experiment.
        
        Args:
            experiment_name: Name of the experiment to get results for.
                           If None, returns results for the active experiment.
                           
        Returns:
            Dictionary with experiment results
        """
        name = experiment_name or self.active_experiment
        
        if not name:
            return {"error": "No active experiment"}
        
        if name not in self.experiments:
            return {"error": f"Experiment '{name}' does not exist"}
        
        experiment = self.experiments[name]
        strategies = experiment["strategies"]
        
        # Filter metrics for this experiment
        experiment_metrics = []
        for strategy_metrics in self.metrics.values():
            for metric in strategy_metrics:
                if metric.get("experiment") == name:
                    experiment_metrics.append(metric)
        
        # Convert to DataFrame for easier analysis
        if not experiment_metrics:
            return {
                "experiment": name,
                "status": experiment["status"],
                "strategies": strategies,
                "metrics": {},
                "message": "No metrics recorded yet"
            }
        
        df = pd.DataFrame(experiment_metrics)
        
        # Calculate summary statistics by strategy
        results = {}
        for strategy in strategies:
            strategy_df = df[df["strategy"] == strategy]
            
            if len(strategy_df) == 0:
                results[strategy] = {"message": "No data recorded for this strategy"}
                continue
            
            # Calculate metrics
            metrics = {
                "query_count": len(strategy_df),
                "avg_latency": strategy_df.get("latency", pd.Series([0])).mean(),
                "avg_user_rating": strategy_df.get("user_rating", pd.Series([0])).mean(),
                "success_rate": strategy_df.get("success", pd.Series([False])).mean(),
            }
            
            # Add any other numeric metrics that might be present
            for col in strategy_df.columns:
                if col not in ["strategy", "query", "timestamp", "experiment", "latency", "user_rating", "success"] and \
                   pd.api.types.is_numeric_dtype(strategy_df[col]):
                    metrics[f"avg_{col}"] = strategy_df[col].mean()
            
            results[strategy] = metrics
        
        return {
            "experiment": name,
            "status": experiment["status"],
            "duration": experiment.get("stopped_at", time.time()) - experiment.get("started_at", time.time()),
            "total_queries": len(df),
            "strategies": strategies,
            "metrics": results
        }
    
    def get_all_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all experiments.
        
        Returns:
            Dictionary mapping experiment names to their information
        """
        return {name: {
            "status": exp["status"],
            "strategies": exp["strategies"],
            "created_at": exp["created_at"],
            "started_at": exp.get("started_at"),
            "stopped_at": exp.get("stopped_at")
        } for name, exp in self.experiments.items()}

    def register_default_strategies(self) -> None:
        """Register default routing strategies for experiments."""
        # Standard strategy is already registered in the router initialization
        
        # Register a domain-focused strategy that prioritizes domain expertise
        self.register_strategy("domain_focused", self._domain_focused_strategy)
        
        # Register a cost-efficient strategy that prioritizes cost
        self.register_strategy("cost_efficient", self._cost_efficient_strategy)
        
        # Register a performance-focused strategy that prioritizes historical performance
        self.register_strategy("performance_focused", self._performance_focused_strategy)
    
    def _domain_focused_strategy(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Domain-focused routing strategy implementation.
        
        This strategy prioritizes domain expertise over other factors.
        
        Args:
            query_data: Dictionary containing query and analysis information
            
        Returns:
            Response from the selected provider
        """
        from router.selector import ModelSelector
        from config import ROUTING_CONFIG
        
        query = query_data["query"]
        analysis = query_data["analysis"]
        
        # Create a custom selector with domain match weighted higher
        custom_weights = ROUTING_CONFIG["weights"].copy()
        custom_weights["domain_match"] = 0.6  # Increase domain weight
        custom_weights["query_complexity"] = 0.2  # Reduce complexity weight
        custom_weights["performance_history"] = 0.1  # Reduce performance weight
        custom_weights["cost"] = 0.05  # Reduce cost weight
        custom_weights["response_time"] = 0.05  # Reduce response time weight
        
        # Normalize weights to ensure they sum to 1.0
        total = sum(custom_weights.values())
        for key in custom_weights:
            custom_weights[key] /= total
        
        # Create a temporary selector with custom weights
        temp_selector = ModelSelector(weights=custom_weights)
        
        # Get performance history from the router's feedback system
        from router import LLMRouter
        router = next((obj for obj in globals().values() if isinstance(obj, LLMRouter)), None)
        performance_history = router.feedback.get_recent_performance() if router else {}
        
        # Select provider using the custom selector
        provider_name = temp_selector.select_provider(analysis, performance_history)
        
        # Use the router to get or create the provider and generate response
        if router:
            if provider_name not in router.providers:
                from models import create_provider
                router.providers[provider_name] = create_provider(provider_name)
            
            provider = router.providers[provider_name]
            response = provider.generate_with_fallback(query)
            
            # Add metadata
            if "metadata" not in response:
                response["metadata"] = {}
            response["metadata"]["strategy"] = "domain_focused"
            response["metadata"]["provider"] = provider_name
            
            return response
        
        # Fallback to standard routing if router not found
        return query_data
    
    def _cost_efficient_strategy(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cost-efficient routing strategy implementation.
        
        This strategy prioritizes cost efficiency over other factors.
        
        Args:
            query_data: Dictionary containing query and analysis information
            
        Returns:
            Response from the selected provider
        """
        from router.selector import ModelSelector
        from config import ROUTING_CONFIG
        
        query = query_data["query"]
        analysis = query_data["analysis"]
        
        # Create a custom selector with cost weighted higher
        custom_weights = ROUTING_CONFIG["weights"].copy()
        custom_weights["cost"] = 0.5  # Increase cost weight
        custom_weights["domain_match"] = 0.2  # Reduce domain weight
        custom_weights["query_complexity"] = 0.15  # Reduce complexity weight
        custom_weights["performance_history"] = 0.1  # Reduce performance weight
        custom_weights["response_time"] = 0.05  # Reduce response time weight
        
        # Normalize weights to ensure they sum to 1.0
        total = sum(custom_weights.values())
        for key in custom_weights:
            custom_weights[key] /= total
        
        # Create a temporary selector with custom weights
        temp_selector = ModelSelector(weights=custom_weights)
        
        # Get performance history from the router's feedback system
        from router import LLMRouter
        router = next((obj for obj in globals().values() if isinstance(obj, LLMRouter)), None)
        performance_history = router.feedback.get_recent_performance() if router else {}
        
        # Select provider using the custom selector
        provider_name = temp_selector.select_provider(analysis, performance_history)
        
        # Use the router to get or create the provider and generate response
        if router:
            if provider_name not in router.providers:
                from models import create_provider
                router.providers[provider_name] = create_provider(provider_name)
            
            provider = router.providers[provider_name]
            response = provider.generate_with_fallback(query)
            
            # Add metadata
            if "metadata" not in response:
                response["metadata"] = {}
            response["metadata"]["strategy"] = "cost_efficient"
            response["metadata"]["provider"] = provider_name
            
            return response
        
        # Fallback to standard routing if router not found
        return query_data
    
    def _performance_focused_strategy(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Performance-focused routing strategy implementation.
        
        This strategy prioritizes historical performance over other factors.
        
        Args:
            query_data: Dictionary containing query and analysis information
            
        Returns:
            Response from the selected provider
        """
        from router.selector import ModelSelector
        from config import ROUTING_CONFIG
        
        query = query_data["query"]
        analysis = query_data["analysis"]
        
        # Create a custom selector with performance weighted higher
        custom_weights = ROUTING_CONFIG["weights"].copy()
        custom_weights["performance_history"] = 0.5  # Increase performance weight
        custom_weights["domain_match"] = 0.2  # Reduce domain weight
        custom_weights["query_complexity"] = 0.15  # Reduce complexity weight
        custom_weights["cost"] = 0.1  # Reduce cost weight
        custom_weights["response_time"] = 0.05  # Reduce response time weight
        
        # Normalize weights to ensure they sum to 1.0
        total = sum(custom_weights.values())
        for key in custom_weights:
            custom_weights[key] /= total
        
        # Create a temporary selector with custom weights
        temp_selector = ModelSelector(weights=custom_weights)
        
        # Get performance history from the router's feedback system
        from router import LLMRouter
        router = next((obj for obj in globals().values() if isinstance(obj, LLMRouter)), None)
        performance_history = router.feedback.get_recent_performance() if router else {}
        
        # Select provider using the custom selector
        provider_name = temp_selector.select_provider(analysis, performance_history)
        
        # Use the router to get or create the provider and generate response
        if router:
            if provider_name not in router.providers:
                from models import create_provider
                router.providers[provider_name] = create_provider(provider_name)
            
            provider = router.providers[provider_name]
            response = provider.generate_with_fallback(query)
            
            # Add metadata
            if "metadata" not in response:
                response["metadata"] = {}
            response["metadata"]["strategy"] = "performance_focused"
            response["metadata"]["provider"] = provider_name
            
            return response
        
        # Fallback to standard routing if router not found
        return query_data
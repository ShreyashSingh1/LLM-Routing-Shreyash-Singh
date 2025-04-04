"""Router package for the Dynamic LLM Router."""

from typing import Dict, Any, Optional
import time

from router.analyzer import QueryAnalyzer
from router.selector import ModelSelector
from router.feedback import FeedbackSystem
from router.discovery import ModelDiscovery
from router.experiments import ExperimentManager
from router.cache import QueryCache
from models import create_provider
from config import CACHE_CONFIG, EXPERIMENT_CONFIG


def create_router():
    """Create and initialize the LLM router.
    
    Returns:
        Initialized router instance
    """
    return LLMRouter()


class LLMRouter:
    """Main router class that orchestrates query analysis and model selection."""
    
    def __init__(self):
        """Initialize the router components."""
        self.analyzer = QueryAnalyzer()
        self.selector = ModelSelector()
        self.feedback = FeedbackSystem()
        self.discovery = ModelDiscovery()
        self.providers = {}
        
        # Initialize cache if enabled
        self.cache_enabled = CACHE_CONFIG.get("enabled", False)
        if self.cache_enabled:
            self.cache = QueryCache(
                max_size=CACHE_CONFIG.get("max_size", 1000),
                default_ttl=CACHE_CONFIG.get("default_ttl", 3600),
                strategy=CACHE_CONFIG.get("strategy", "ttl")
            )
        
        # Initialize experiment manager if enabled
        self.experiments_enabled = EXPERIMENT_CONFIG.get("enabled", False)
        if self.experiments_enabled:
            self.experiments = ExperimentManager()
            # Register default routing strategy
            self.experiments.register_strategy("standard", self._standard_routing_strategy)
            # Register additional strategies
            self.experiments.register_default_strategies()
    
    def route(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Route a query to the most appropriate LLM provider.
        
        Args:
            query: The input query string
            context: Optional context information
            
        Returns:
            Response from the selected provider
        """
        # Check cache first if enabled
        if self.cache_enabled:
            cached_response = self.cache.get(query)
            if cached_response:
                return cached_response
        
        # Analyze the query
        analysis = self.analyzer.analyze(query)
        
        # Prepare query data for experiment routing
        query_data = {
            "query": query,
            "analysis": analysis,
            "context": context or {}
        }
        
        # Use experiment manager to determine routing strategy if enabled
        if self.experiments_enabled and self.experiments.active_experiment:
            strategy_name = self.experiments.get_strategy(query_data)
            strategy_fn = self.experiments.strategy_registry.get(strategy_name)
            if strategy_fn:
                start_time = time.time()
                response = strategy_fn(query_data)
                end_time = time.time()
                
                # Record metrics for the experiment
                self.experiments.record_metrics(
                    strategy=strategy_name,
                    query=query,
                    metrics={
                        "latency": end_time - start_time,
                        "success": True,  # Assuming success unless error handling indicates otherwise
                        "token_usage": response.get("metadata", {}).get("token_usage", {}),
                    }
                )
                
                return response
        
        # Fall back to standard routing if no experiment is active or strategy not found
        return self._standard_routing_strategy(query_data)
    
    def _standard_routing_strategy(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standard routing strategy implementation.
        
        Args:
            query_data: Dictionary containing query and analysis information
            
        Returns:
            Response from the selected provider
        """
        query = query_data["query"]
        analysis = query_data["analysis"]
        
        # Get performance history from feedback system
        performance_history = self.feedback.get_recent_performance()
        
        # Select the most appropriate provider
        provider_name = self.selector.select_provider(analysis, performance_history)
        
        # Get or create the provider instance
        if provider_name not in self.providers:
            self.providers[provider_name] = create_provider(provider_name)
        
        provider = self.providers[provider_name]
        
        # Generate the response with error handling and fallback
        start_time = time.time()
        response = provider.generate_with_fallback(query)  # Using enhanced error handling
        end_time = time.time()
        
        # Record feedback
        self.feedback.record_query(
            query=query,
            provider=provider_name,
            response=response,
            latency=end_time - start_time
        )
        
        # Add metadata to the response
        if "metadata" not in response:
            response["metadata"] = {}
            
        response["metadata"]["provider"] = provider_name
        response["metadata"]["query_analysis"] = analysis
        response["metadata"]["latency"] = end_time - start_time
        
        # Cache the response if caching is enabled
        if self.cache_enabled and not response.get("error"):
            # Get domain-specific TTL if available
            domain = analysis.get("primary_domain", "general_knowledge")
            ttl = CACHE_CONFIG.get("ttl_by_domain", {}).get(domain, CACHE_CONFIG.get("default_ttl"))
            
            self.cache.set(query, response, provider_name, ttl)
        
        return response
    
    def provide_feedback(self, query_id: str, feedback: Dict[str, Any]) -> None:
        """Provide feedback for a previous query.
        
        Args:
            query_id: Identifier for the query
            feedback: Feedback information
        """
        self.feedback.add_feedback(query_id, feedback)
        
        # Update experiment metrics if applicable
        if self.experiments_enabled and self.experiments.active_experiment:
            # Find the query in experiment metrics
            for strategy, metrics_list in self.experiments.metrics.items():
                for metric in metrics_list:
                    if metric.get("query") == query_id:
                        # Update with user feedback
                        metric["user_rating"] = feedback.get("rating")
                        metric["success"] = feedback.get("success", True)
                        break
    
    def discover_models(self, criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Discover models based on specified criteria.
        
        Args:
            criteria: Optional criteria for model discovery
            
        Returns:
            Dictionary with discovered models
        """
        return self.discovery.discover_models(criteria)
    
    def create_experiment(self, name: str, strategies: list, traffic_split: Optional[Dict[str, float]] = None) -> None:
        """Create a new A/B testing experiment.
        
        Args:
            name: Name of the experiment
            strategies: List of strategy names to compare
            traffic_split: Optional dictionary mapping strategy names to traffic percentages
        """
        if not self.experiments_enabled:
            raise ValueError("Experiments are not enabled in the configuration")
        
        self.experiments.create_experiment(name, strategies, traffic_split)
    
    def start_experiment(self, name: str) -> None:
        """Start an experiment.
        
        Args:
            name: Name of the experiment to start
        """
        if not self.experiments_enabled:
            raise ValueError("Experiments are not enabled in the configuration")
        
        self.experiments.start_experiment(name)
    
    def stop_experiment(self, name: str) -> None:
        """Stop an experiment.
        
        Args:
            name: Name of the experiment to stop
        """
        if not self.experiments_enabled:
            raise ValueError("Experiments are not enabled in the configuration")
        
        self.experiments.stop_experiment(name)
    
    def get_experiment_results(self, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """Get results for an experiment.
        
        Args:
            experiment_name: Name of the experiment to get results for
            
        Returns:
            Dictionary with experiment results
        """
        if not self.experiments_enabled:
            raise ValueError("Experiments are not enabled in the configuration")
        
        return self.experiments.get_experiment_results(experiment_name)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_enabled:
            raise ValueError("Cache is not enabled in the configuration")
        
        return self.cache.get_stats()
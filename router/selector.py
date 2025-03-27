"""Model selector component for the Dynamic LLM Router."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from config import MODEL_CONFIGS, ROUTING_CONFIG


class ModelSelector:
    """Selects the most appropriate LLM provider based on query characteristics."""
    
    def __init__(self):
        """Initialize the model selector."""
        self.model_configs = MODEL_CONFIGS
        self.routing_config = ROUTING_CONFIG
        self.weights = self.routing_config["weights"]
        self.domains = self.routing_config["domains"]
        self.default_provider = self.routing_config["default_provider"]
    
    def select_provider(self, query_analysis: Dict[str, Any], 
                        performance_history: Optional[Dict[str, Any]] = None) -> str:
        """Select the most appropriate provider based on query analysis.
        
        Args:
            query_analysis: Analysis of the query from QueryAnalyzer
            performance_history: Optional historical performance data
            
        Returns:
            Provider name string
        """
        # Calculate scores for each provider
        provider_scores = {}
        
        for provider_name in self.model_configs.keys():
            # Calculate individual factor scores
            complexity_score = self._score_complexity(provider_name, query_analysis)
            domain_score = self._score_domain_match(provider_name, query_analysis)
            performance_score = self._score_performance(provider_name, performance_history)
            cost_score = self._score_cost(provider_name, query_analysis)
            response_time_score = self._score_response_time(provider_name, query_analysis)
            
            # Combine scores using weights
            weighted_score = (
                self.weights["query_complexity"] * complexity_score +
                self.weights["domain_match"] * domain_score +
                self.weights["performance_history"] * performance_score +
                self.weights["cost"] * cost_score +
                self.weights["response_time"] * response_time_score
            )
            
            provider_scores[provider_name] = weighted_score
        
        # Select the provider with the highest score
        if provider_scores:
            selected_provider = max(provider_scores.items(), key=lambda x: x[1])[0]
            return selected_provider
        else:
            return self.default_provider
    
    def _score_complexity(self, provider_name: str, query_analysis: Dict[str, Any]) -> float:
        """Score provider based on query complexity.
        
        Args:
            provider_name: Name of the provider
            query_analysis: Analysis of the query
            
        Returns:
            Complexity match score between 0 and 1
        """
        complexity_category = query_analysis.get("complexity_category", "medium")
        
        # Map providers to their suitability for different complexity levels
        complexity_suitability = {
            "low": {"mistral": 1.0, "google": 0.9, "openai": 0.8, "anthropic": 0.7},
            "medium": {"openai": 1.0, "google": 0.9, "mistral": 0.8, "anthropic": 0.8},
            "high": {"anthropic": 1.0, "openai": 0.9, "google": 0.7, "mistral": 0.6},
        }
        
        return complexity_suitability.get(complexity_category, {}).get(provider_name, 0.5)
    
    def _score_domain_match(self, provider_name: str, query_analysis: Dict[str, Any]) -> float:
        """Score provider based on domain match.
        
        Args:
            provider_name: Name of the provider
            query_analysis: Analysis of the query
            
        Returns:
            Domain match score between 0 and 1
        """
        primary_domain = query_analysis.get("primary_domain", "general_knowledge")
        domain_scores = query_analysis.get("domain_scores", {})
        
        # Check if this provider is recommended for the primary domain
        domain_providers = self.domains.get(primary_domain, [])
        
        if not domain_providers:
            return 0.5  # Neutral score if no providers specified for domain
        
        # Calculate score based on provider's position in the domain list
        if provider_name in domain_providers:
            position = domain_providers.index(provider_name)
            return 1.0 - (position * 0.2)  # Higher score for earlier positions
        else:
            return 0.2  # Low score if not in the list
    
    def _score_performance(self, provider_name: str, 
                           performance_history: Optional[Dict[str, Any]]) -> float:
        """Score provider based on historical performance.
        
        Args:
            provider_name: Name of the provider
            performance_history: Historical performance data
            
        Returns:
            Performance score between 0 and 1
        """
        if not performance_history or provider_name not in performance_history:
            return 0.5  # Neutral score if no history
        
        provider_history = performance_history.get(provider_name, {})
        success_rate = provider_history.get("success_rate", 0.5)
        avg_latency = provider_history.get("avg_latency", 1.0)
        
        # Normalize latency score (lower is better)
        latency_score = 1.0 - min(avg_latency / 5.0, 1.0)  # Assuming 5 seconds is the worst acceptable
        
        # Combine success rate and latency
        return 0.7 * success_rate + 0.3 * latency_score
    
    def _score_cost(self, provider_name: str, query_analysis: Dict[str, Any]) -> float:
        """Score provider based on cost considerations.
        
        Args:
            provider_name: Name of the provider
            query_analysis: Analysis of the query
            
        Returns:
            Cost score between 0 and 1 (higher is better, meaning lower cost)
        """
        estimated_tokens = query_analysis.get("estimated_tokens", 100)
        cost_per_1k = self.model_configs.get(provider_name, {}).get("cost_per_1k_tokens", 0.01)
        
        # Calculate estimated cost
        estimated_cost = (estimated_tokens / 1000) * cost_per_1k
        
        # Normalize cost score (lower cost is better)
        # Assuming 0.05 USD is the highest acceptable cost for a query
        cost_score = 1.0 - min(estimated_cost / 0.05, 1.0)
        
        return cost_score
    
    def _score_response_time(self, provider_name: str, query_analysis: Dict[str, Any]) -> float:
        """Score provider based on response time needs.
        
        Args:
            provider_name: Name of the provider
            query_analysis: Analysis of the query
            
        Returns:
            Response time score between 0 and 1
        """
        # Map response time categories to scores (higher is better)
        response_time_scores = {
            "fast": 1.0,
            "medium": 0.7,
            "slow": 0.4,
        }
        
        provider_response_time = self.model_configs.get(provider_name, {}).get("response_time", "medium")
        return response_time_scores.get(provider_response_time, 0.7)
    
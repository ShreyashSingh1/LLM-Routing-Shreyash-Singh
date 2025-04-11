"""Feedback system for tracking LLM provider performance."""

from typing import Dict, Any, List, Optional, Tuple
import time
from collections import deque

from config import FEEDBACK_CONFIG


class FeedbackSystem:
    """Tracks performance of LLM providers to improve routing decisions."""
    
    def __init__(self):
        """Initialize the feedback system."""
        self.history_size = FEEDBACK_CONFIG["history_size"]
        self.performance_window = FEEDBACK_CONFIG["performance_window"]
        self.min_feedback_samples = FEEDBACK_CONFIG["min_feedback_samples"]
        
        # Initialize performance history for each provider
        self.query_history = deque(maxlen=self.history_size)
        self.provider_stats = {}
    
    def record_query(self, query: str, provider: str, response: Dict[str, Any], 
                    latency: float, feedback: Optional[Dict[str, Any]] = None) -> None:
        """Record a query and its response for performance tracking.
        
        Args:
            query: The original query string
            provider: The provider that handled the query
            response: The response from the provider
            latency: The time taken to generate the response
            feedback: Optional user or system feedback
        """
        # Create a record of this query
        record = {
            "query": query,
            "provider": provider,
            "timestamp": time.time(),
            "latency": latency,
            "token_usage": response.get("metadata", {}).get("token_usage", {}),
            "feedback": feedback or {},
        }
        
        # Add to history
        self.query_history.append(record)
        
        # Update provider stats
        if provider not in self.provider_stats:
            self.provider_stats[provider] = {
                "query_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "total_latency": 0.0,
                "avg_latency": 0.0,
                "success_rate": 0.5,  # Start with neutral success rate
            }
        
        # Update stats
        stats = self.provider_stats[provider]
        stats["query_count"] += 1
        stats["total_latency"] += latency
        stats["avg_latency"] = stats["total_latency"] / stats["query_count"]
        
        # Update success/failure counts if feedback is provided
        if feedback and "success" in feedback:
            if feedback["success"]:
                stats["success_count"] += 1
            else:
                stats["failure_count"] += 1
                
            # Recalculate success rate
            total_feedback = stats["success_count"] + stats["failure_count"]
            if total_feedback > 0:
                stats["success_rate"] = stats["success_count"] / total_feedback
    
    def get_performance_history(self) -> Dict[str, Any]:
        """Get performance history for all providers.
        
        Returns:
            Dictionary with performance metrics for each provider
        """
        # Return a copy of the provider stats
        return {provider: stats.copy() for provider, stats in self.provider_stats.items()}
    
    def get_recent_performance(self) -> Dict[str, Any]:
        """Get recent performance metrics based on the performance window.
        
        Returns:
            Dictionary with recent performance metrics for each provider
        """
        # Calculate recent performance based on the last N queries
        recent_queries = list(self.query_history)[-self.performance_window:]
        
        # Group by provider
        provider_recent = {}
        
        for record in recent_queries:
            provider = record["provider"]
            
            if provider not in provider_recent:
                provider_recent[provider] = {
                    "query_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "total_latency": 0.0,
                    "avg_latency": 0.0,
                    "success_rate": 0.5,  # Start with neutral success rate
                }
            
            stats = provider_recent[provider]
            stats["query_count"] += 1
            stats["total_latency"] += record["latency"]
            
            # Update success/failure if feedback is available
            feedback = record.get("feedback", {})
            if "success" in feedback:
                if feedback["success"]:
                    stats["success_count"] += 1
                else:
                    stats["failure_count"] += 1
        
        # Calculate averages and rates
        for provider, stats in provider_recent.items():
            if stats["query_count"] > 0:
                stats["avg_latency"] = stats["total_latency"] / stats["query_count"]
                
            total_feedback = stats["success_count"] + stats["failure_count"]
            if total_feedback >= self.min_feedback_samples:
                stats["success_rate"] = stats["success_count"] / total_feedback
        
        return provider_recent
    
    def should_consider_feedback(self, provider: str) -> bool:
        """Determine if we have enough feedback data to consider for routing.
        
        Args:
            provider: The provider to check
            
        Returns:
            True if we have enough feedback data, False otherwise
        """
        if provider not in self.provider_stats:
            return False
            
        stats = self.provider_stats[provider]
        total_feedback = stats["success_count"] + stats["failure_count"]
        
        return total_feedback >= self.min_feedback_samples

    def record_feedback(self, query_id: str, rating: int, comments: Optional[str] = None) -> bool:
        """Record feedback for a specific query.

        Args:
            query_id: The unique identifier of the query.
            rating: The feedback rating (e.g., 1-5).
            comments: Optional comments about the feedback.

        Returns:
            True if the feedback was recorded successfully, False otherwise.
        """
        # Search for the query in the history
        for record in self.query_history:
            if record.get("query_id") == query_id:
                # Update the feedback in the record
                record["feedback"] = {
                    "rating": rating,
                    "comments": comments,
                    "success": rating >= 3  # Consider ratings 3 and above as successful
                }
                return True

        # Query ID not found
        return False
"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import random
import time


class LLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    All LLM provider implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LLM provider with configuration.
        
        Args:
            config: Configuration dictionary for the provider
        """
        self.config = config
        self.default_model = config.get("default_model")
        self.fallback_model = config.get("fallback_model")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt/query
            **kwargs: Additional parameters for the generation
            
        Returns:
            Dictionary containing the response and metadata
        """
        pass
    
    @property
    def strengths(self) -> List[str]:
        """Get the strengths of this provider.
        
        Returns:
            List of strength categories
        """
        return self.config.get("strengths", [])
    
    @property
    def cost_per_1k_tokens(self) -> float:
        """Get the cost per 1k tokens for this provider.
        
        Returns:
            Cost per 1k tokens
        """
        return self.config.get("cost_per_1k_tokens", 0.01)
        
    @property
    def response_time(self) -> str:
        """Get the response time category for this provider.
        
        Returns:
            Response time category (fast, medium, slow)
        """
        return self.config.get("cost_per_1k_tokens", 0.0)
    
    @property
    def response_time(self) -> str:
        """Get the typical response time category for this provider.
        
        Returns:
            Response time category (fast, medium, slow)
        """
        return self.config.get("response_time", "medium")
    
    def get_model(self, use_fallback: bool = False) -> str:
        """Get the model to use, with fallback option.
        
        Args:
            use_fallback: Whether to use the fallback model
            
        Returns:
            Model identifier string
        """
        if use_fallback and self.fallback_model:
            return self.fallback_model
        return self.default_model
        
    def mock_generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a mock response without requiring API keys.
        
        Args:
            prompt: The input prompt/query
            **kwargs: Additional parameters for the generation
            
        Returns:
            Dictionary containing the mock response and metadata
        """
        # Simulate processing time based on provider's response_time property
        response_times = {"fast": 0.5, "medium": 1.0, "slow": 1.5}
        delay = response_times.get(self.response_time, 1.0)
        delay *= random.uniform(0.8, 1.2)  # Add some randomness
        time.sleep(delay)
        
        # Get model name
        model = kwargs.get("model", self.get_model(kwargs.get("use_fallback", False)))
        
        # Generate mock response based on provider strengths
        response_prefix = f"[{self.__class__.__name__} - {model}] "
        
        # Simulate different response styles based on provider strengths
        if "coding" in self.strengths and any(kw in prompt.lower() for kw in ["code", "function", "program"]):
            response = response_prefix + "Here's a code implementation that addresses your request."
        elif "creative_writing" in self.strengths and any(kw in prompt.lower() for kw in ["write", "story", "poem"]):
            response = response_prefix + "Here's a creative response to your writing request."
        elif "reasoning" in self.strengths and any(kw in prompt.lower() for kw in ["explain", "why", "how"]):
            response = response_prefix + "Here's a detailed explanation with reasoning."
        else:
            response = response_prefix + "Here's a general response to your query."
            
        # Simulate token usage based on prompt length
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(response) // 4
        total_tokens = prompt_tokens + completion_tokens
        
        return {
            "content": response,
            "model": model,
            "provider": self.__class__.__name__,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens
            },
            "cost": (total_tokens / 1000) * self.cost_per_1k_tokens,
            "latency": delay
        }
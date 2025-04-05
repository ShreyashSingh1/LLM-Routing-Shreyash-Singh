"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
import random
import time
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from langchain_groq import ChatGroq  # Add this import

# Debugging the import
try:
    assert ChatGroq is not None
    logger.info("ChatGroq successfully imported.")
except Exception as e:
    logger.error(f"Failed to import ChatGroq: {e}")


class LLMProviderError(Exception):
    """Base exception class for LLM provider errors."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.details = details or {}


class RateLimitError(LLMProviderError):
    """Exception raised when provider rate limits are exceeded."""
    pass


class AuthenticationError(LLMProviderError):
    """Exception raised when authentication fails."""
    pass


class ProviderAPIError(LLMProviderError):
    """Exception raised when the provider API returns an error."""
    pass


class ContextLengthExceededError(LLMProviderError):
    """Exception raised when the input exceeds the model's context length."""
    pass


class TimeoutError(LLMProviderError):
    """Exception raised when a request to the provider times out."""
    pass


class NetworkError(LLMProviderError):
    """Exception raised when there are network connectivity issues."""
    pass


class InvalidRequestError(LLMProviderError):
    """Exception raised when the request to the provider is invalid."""
    pass


class ModelNotAvailableError(LLMProviderError):
    """Exception raised when the requested model is not available."""
    pass


class ContentFilterError(LLMProviderError):
    """Exception raised when content is filtered by the provider's safety systems."""
    pass


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
        
        # Error handling configuration
        self.max_retries = config.get("max_retries", 3)
        self.initial_backoff = config.get("initial_backoff", 1.0)  # seconds
        self.backoff_factor = config.get("backoff_factor", 2.0)
        self.jitter = config.get("jitter", 0.1)  # random jitter factor
        
        # Fallback chain configuration
        self.fallback_providers = config.get("fallback_providers", [])
        
    def generate_with_fallback(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response with fallback to other providers if this one fails.
        
        Args:
            prompt: The input prompt/query
            **kwargs: Additional parameters for the generation
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Try with the current provider first
        try:
            return self.generate_with_retry(prompt, **kwargs)
        except LLMProviderError as e:
            logger.warning(f"Provider {self.__class__.__name__} failed: {str(e)}")
            
            # Try fallback providers if available
            if self.fallback_providers:
                from models import create_provider
                
                for provider_name in self.fallback_providers:
                    try:
                        logger.info(f"Attempting fallback to provider: {provider_name}")
                        
                        # Create the fallback provider instance
                        fallback_provider = create_provider(provider_name)
                        
                        # Call generate on the fallback provider
                        fallback_response = fallback_provider.generate_with_retry(prompt, **kwargs)
                        
                        # Add fallback metadata
                        if "metadata" not in fallback_response:
                            fallback_response["metadata"] = {}
                            
                        fallback_response["metadata"]["fallback"] = True
                        fallback_response["metadata"]["original_provider"] = self.__class__.__name__
                        fallback_response["metadata"]["original_error"] = {
                            "type": e.__class__.__name__,
                            "message": str(e),
                            "details": getattr(e, "details", {})
                        }
                        
                        return fallback_response
                        
                    except Exception as fallback_error:
                        logger.warning(f"Fallback provider {provider_name} also failed: {str(fallback_error)}")
            
            # If we get here, all fallbacks failed or none were available
            # Return a structured error response
            return {
                "text": "Sorry, I encountered an error processing your request.",
                "error": {
                    "type": e.__class__.__name__,
                    "message": str(e),
                    "details": getattr(e, "details", {})
                },
                "metadata": {
                    "success": False,
                    "fallback_attempted": bool(self.fallback_providers)
                }
            }
    
    def generate_with_retry(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response with automatic retry logic.
        
        Args:
            prompt: The input prompt/query
            **kwargs: Additional parameters for the generation
            
        Returns:
            Dictionary containing the response and metadata
            
        Raises:
            LLMProviderError: If all retries fail
        """
        retries = 0
        last_exception = None
        
        # Define which errors are retriable vs. non-retriable
        retriable_errors = (RateLimitError, ProviderAPIError, NetworkError, TimeoutError)
        non_retriable_errors = (AuthenticationError, ContextLengthExceededError, 
                               InvalidRequestError, ContentFilterError)
        
        while retries <= self.max_retries:
            try:
                # If this isn't the first attempt, try with fallback model
                if retries > 0 and self.fallback_model:
                    logger.info(f"Retry {retries}/{self.max_retries} with fallback model: {self.fallback_model}")
                    kwargs["model"] = self.fallback_model
                
                # Adjust parameters for retry attempts
                if retries > 0:
                    # For rate limit errors, we might want to reduce the request complexity
                    if isinstance(last_exception, RateLimitError):
                        # Reduce token count for subsequent attempts
                        if "max_tokens" in kwargs and kwargs["max_tokens"] > 100:
                            kwargs["max_tokens"] = int(kwargs["max_tokens"] * 0.8)  # Reduce by 20%
                    
                    # For timeout errors, we might want to simplify the request
                    if isinstance(last_exception, TimeoutError):
                        # Increase timeout for subsequent attempts if supported by the provider
                        kwargs["timeout"] = kwargs.get("timeout", 30) * 1.5
                
                start_time = time.time()
                response = self.generate(prompt, **kwargs)
                end_time = time.time()
                
                # Add retry information to metadata
                if "metadata" not in response:
                    response["metadata"] = {}
                
                response["metadata"]["retries"] = retries
                response["metadata"]["latency"] = end_time - start_time
                if retries > 0 and "model" in kwargs:
                    response["metadata"]["fallback_model_used"] = kwargs["model"]
                
                return response
                
            except retriable_errors as e:
                last_exception = e
                retries += 1
                
                if retries <= self.max_retries:
                    # Calculate backoff time with exponential backoff and jitter
                    backoff_time = self.initial_backoff * (self.backoff_factor ** (retries - 1))
                    jitter_amount = backoff_time * self.jitter * random.uniform(-1, 1)
                    sleep_time = backoff_time + jitter_amount
                    
                    logger.warning(f"Attempt {retries} failed: {str(e)}. Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All {self.max_retries} retries failed for provider {self.__class__.__name__}")
            
            except non_retriable_errors as e:
                # Don't retry these errors as they're unlikely to be resolved by retrying
                last_exception = e
                logger.error(f"Non-retriable error: {str(e)}")
                break
            
            except Exception as e:
                # For unexpected errors, log and treat as non-retriable
                last_exception = ProviderAPIError(f"Unexpected error: {str(e)}", {"original_error": str(e)})
                logger.error(f"Unexpected error during generation: {str(e)}")
                break
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise LLMProviderError("Failed to generate response after retries")
    
    def process_response(self, response):
        """Process the response from the LLM."""
        try:
            # Safely access response_metadata if it exists
            response_metadata = getattr(response, "response_metadata", None)
            if response_metadata is None:
                logger.warning("Response metadata is missing in the AIMessage object.")
                response_metadata = {}  # Provide a default empty dictionary

            # Process the response content
            content = response.content
            logger.info(f"Processed response content: {content}")
            return {"content": content, "metadata": response_metadata}

        except AttributeError as e:
            logger.error(f"Unexpected error during response processing: {e}")
            raise

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response using LangChain's ChatGroq model.
        
        Args:
            prompt: The input prompt/query
            **kwargs: Additional parameters for the generation
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Initialize ChatGroq model
        chat_model = ChatGroq(
            model="Llama3-8b-8192",
            groq_api_key="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"
        )

        try:
            # Generate response
            response = chat_model.invoke(prompt)
            logger.info(f"Raw response from ChatGroq: {response}")

            # Validate and extract response text
            if hasattr(response, "content") and isinstance(response.content, str):
                response_text = response.content.strip()
            else:
                logger.error(f"Unexpected response format: {response}")
                raise ProviderAPIError("Unexpected response format from ChatGroq.")

            # Safely extract token usage and other metadata
            response_metadata = getattr(response, "response_metadata", {})
            if not isinstance(response_metadata, dict):
                logger.warning("Response metadata is not a dictionary. Defaulting to empty metadata.")
                response_metadata = {}

            token_usage = response_metadata.get("token_usage", {})
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)

            # Correctly format the return dictionary
            return {
                "content": response_text,
                "model": response_metadata.get("model_name", self.default_model),
                "provider": self.__class__.__name__,
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens
                },
                "cost": (total_tokens / 1000) * self.cost_per_1k_tokens,
                "latency": token_usage.get("total_time", 0)  # Use total_time as latency if available
            }
        except ProviderAPIError as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during generation: {str(e)}")
            raise ProviderAPIError(f"Unexpected error: {str(e)}")
    
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
            
        Raises:
            Various LLMProviderError subclasses to simulate real API errors
        """
        # Randomly simulate errors to test error handling (10% chance)
        if random.random() < 0.1:
            error_types = [
                (RateLimitError, "Rate limit exceeded. Please try again later."),
                (AuthenticationError, "Invalid API key or authentication token."),
                (ProviderAPIError, "The provider API returned an error."),
                (ContextLengthExceededError, "Input exceeds maximum context length."),
                (TimeoutError, "Request timed out. Please try again."),
                (NetworkError, "Network connectivity issues detected."),
                (InvalidRequestError, "The request contains invalid parameters."),
                (ModelNotAvailableError, "The requested model is currently unavailable."),
                (ContentFilterError, "Content was filtered due to safety concerns.")
            ]
            error_class, error_msg = random.choice(error_types)
            raise error_class(error_msg)
            
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
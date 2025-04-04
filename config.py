"""Configuration settings for the Dynamic LLM Router."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Mock API Keys (no real keys needed)
OPENAI_API_KEY = "mock-openai-key"
ANTHROPIC_API_KEY = "mock-anthropic-key"
MISTRAL_API_KEY = "mock-mistral-key"
GOOGLE_API_KEY = "mock-google-key"

# Model configurations
MODEL_CONFIGS = {
    "openai": {
        "default_model": "gpt-4o",
        "fallback_model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000,
        "cost_per_1k_tokens": 0.01,  # Simplified cost metric
        "strengths": ["general_knowledge", "coding", "reasoning"],
        "response_time": "medium",
    },
    "anthropic": {
        "default_model": "claude-3-opus",
        "fallback_model": "claude-3-sonnet",
        "temperature": 0.7,
        "max_tokens": 1000,
        "cost_per_1k_tokens": 0.015,
        "strengths": ["creative_writing", "reasoning", "safety"],
        "response_time": "slow",
    },
    "mistral": {
        "default_model": "mistral-large",
        "fallback_model": "mistral-medium",
        "temperature": 0.7,
        "max_tokens": 1000,
        "cost_per_1k_tokens": 0.008,
        "strengths": ["efficiency", "general_knowledge"],
        "response_time": "fast",
    },
    "google": {
        "default_model": "gemini-pro",
        "fallback_model": "gemini-flash",
        "temperature": 0.7,
        "max_tokens": 1000,
        "cost_per_1k_tokens": 0.0075,
        "strengths": ["general_knowledge", "multimodal"],
        "response_time": "medium",
    },
}

# Routing policy configuration
ROUTING_CONFIG = {
    # Weights for different routing factors (must sum to 1.0)
    "weights": {
        "query_complexity": 0.3,
        "domain_match": 0.3,
        "performance_history": 0.2,
        "cost": 0.1,
        "response_time": 0.1,
    },
    
    # Domain expertise mappings
    "domains": {
        "coding": ["openai", "anthropic"],
        "creative_writing": ["anthropic", "openai"],
        "math": ["openai", "google"],
        "science": ["google", "openai"],
        "general_knowledge": ["openai", "google", "mistral", "anthropic"],
        "reasoning": ["anthropic", "openai"],
        "safety": ["anthropic"],
        "multimodal": ["google", "openai"],
    },
    
    # Complexity thresholds
    "complexity": {
        "low": 0.3,  # Below this is considered low complexity
        "high": 0.7,  # Above this is considered high complexity
    },
    
    # Default provider if no clear winner
    "default_provider": "openai",
}

# Feedback configuration
FEEDBACK_CONFIG = {
    "history_size": 100,  # Number of queries to keep in history
    "performance_window": 10,  # Number of recent queries to consider for performance
    "min_feedback_samples": 5,  # Minimum samples needed before considering feedback
}

# Cache configuration
CACHE_CONFIG = {
    "enabled": True,
    "max_size": 1000,  # Maximum number of entries in the cache
    "default_ttl": 3600,  # Default time-to-live in seconds (1 hour)
    "strategy": "ttl",  # Cache invalidation strategy ('ttl', 'lru', or 'lfu')
    "ttl_by_domain": {  # Custom TTL by domain (in seconds)
        "general_knowledge": 86400,  # 24 hours for general knowledge
        "news": 1800,  # 30 minutes for news-related queries
        "weather": 900,  # 15 minutes for weather-related queries
        "coding": 604800,  # 1 week for coding-related queries
    },
}

# Experiment configuration
EXPERIMENT_CONFIG = {
    "enabled": True,
    "default_traffic_split": {
        "standard": 0.7,  # 70% of traffic to standard routing
        "experimental": 0.3,  # 30% of traffic to experimental routing
    },
    "metrics": [
        "latency",
        "user_rating",
        "success_rate",
        "token_usage",
        "cost",
    ],
}

# Enhanced analyzer configuration
ANALYZER_CONFIG = {
    "sentiment_analysis": {
        "enabled": True,
        "urgency_keywords": [
            "urgent", "emergency", "immediately", "asap", "critical", "now", "hurry"
        ],
    },
    "language_detection": {
        "enabled": True,
        "confidence_threshold": 0.5,  # Minimum confidence for language detection
    },
    "topic_classification": {
        "enabled": True,
        "n_topics": 10,  # Number of topics for classification
    },
    "complexity_estimation": {
        "enabled": True,
        "weights": {
            "word_count": 0.3,
            "sentence_length": 0.2,
            "vocabulary_richness": 0.2,
            "question_count": 0.1,
            "nested_structure": 0.1,
            "multi_part": 0.1,
        },
    },
}
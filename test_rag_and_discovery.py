"""Test script for RAG application and model discovery features."""

import time
import json
from typing import Dict, Any, List

from models.mistral import MistralProvider
from models.knowledge_base import KnowledgeBase
from models.model_discovery import ModelDiscovery
from router.discovery import RouterModelDiscovery
from config import MODEL_CONFIGS


def test_rag_fallback():
    """Test the RAG-style fallback mechanism."""
    print("\n=== Testing RAG-style Fallback ===\n")
    
    # Initialize Mistral provider
    mistral_config = MODEL_CONFIGS["mistral"]
    mistral = MistralProvider(mistral_config)
    
    # Test queries for different domains
    test_queries = {
        "general_knowledge": "What is the capital of France?",
        "coding": "Explain what a function is in programming.",
        "math": "What is the Pythagorean theorem?",
        "science": "Explain the theory of relativity."
    }
    
    for domain, query in test_queries.items():
        print(f"\nDomain: {domain}")
        print(f"Query: {query}")
        
        # Generate response with fallback (RAG)
        response = mistral.generate(query, use_fallback=True)
        
        print(f"Response: {response['content']}")
        print(f"RAG Info: {json.dumps(response.get('rag_info', {}), indent=2)}")
        print("-" * 50)


def test_model_discovery():
    """Test the model discovery feature."""
    print("\n=== Testing Model Discovery ===\n")
    
    # Initialize model discovery
    model_discovery = ModelDiscovery()
    
    # Search for LLaMA models
    print("Searching for LLaMA models...")
    llama_models = model_discovery.search_models(category="llama")
    
    print(f"Found {len(llama_models)} LLaMA models:")
    for i, model in enumerate(llama_models):
        print(f"\nModel {i+1}: {model['name']}")
        print(f"Provider: {model['provider']}")
        print(f"Size: {model['size']}")
        print(f"Strengths: {', '.join(model['strengths'])}")
        print(f"Cost per 1k tokens: ${model['cost_per_1k_tokens']}")
        print(f"Response time: {model['response_time']}")
    
    # Evaluate the first model
    if llama_models:
        print("\nEvaluating the first model...")
        model_to_evaluate = llama_models[0]
        evaluation_scores = model_discovery.evaluate_model(model_to_evaluate)
        
        print(f"Evaluation scores for {model_to_evaluate['name']}:")
        for metric, score in evaluation_scores.items():
            print(f"  {metric}: {score:.2f}")
        
        # Check if the model is promising
        is_promising = model_discovery.is_promising(evaluation_scores)
        print(f"\nIs the model promising? {is_promising}")
        
        # Add to config if promising
        if is_promising:
            print("\nAdding model to configuration...")
            updated_config = model_discovery.add_to_config(model_to_evaluate, "meta")
            print(f"Updated configuration for 'meta' provider:")
            print(json.dumps(updated_config.get("meta", {}), indent=2))


def test_router_model_discovery():
    """Test the router model discovery integration."""
    print("\n=== Testing Router Model Discovery Integration ===\n")
    
    # Initialize router model discovery
    router_discovery = RouterModelDiscovery()
    
    # Discover LLaMA variants
    print("Discovering LLaMA variants...")
    promising_models, updated_config = router_discovery.discover_llama_variants()
    
    print(f"Found {len(promising_models)} promising LLaMA variants")
    if promising_models:
        print("\nFirst promising model:")
        print(json.dumps(promising_models[0], indent=2))
    
    if updated_config:
        print("\nConfiguration updated successfully")
    else:
        print("\nNo configuration updates were made")
    
    # Discover models for a specific domain
    domain = "coding"
    print(f"\nDiscovering models for domain: {domain}")
    domain_models, domain_config = router_discovery.discover_models_for_domain(domain)
    
    print(f"Found {len(domain_models)} promising models for {domain}")
    if domain_models:
        print("\nFirst domain-specific model:")
        print(json.dumps(domain_models[0], indent=2))
    
    # Get discovery log
    discovery_log = router_discovery.get_discovery_log()
    print(f"\nDiscovery log entries: {len(discovery_log)}")
    
    # Display the discovery log entries
    if discovery_log:
        print("\nDiscovery Log:")
        for i, entry in enumerate(discovery_log):
            print(f"\nEntry {i+1}:")
            for key, value in entry.items():
                print(f"  {key}: {value}")
    else:
        print("\nNo discovery log entries found.")


def test_mistral_model_discovery():
    """Test the Mistral provider's model discovery capabilities."""
    print("\n=== Testing Mistral Provider Model Discovery ===\n")
    
    # Initialize Mistral provider
    mistral_config = MODEL_CONFIGS["mistral"]
    mistral = MistralProvider(mistral_config)
    
    # Discover LLaMA models
    print("Discovering LLaMA models through Mistral provider...")
    discovered_models = mistral.discover_llama_models()
    
    print(f"Discovered {len(discovered_models)} LLaMA models")
    if discovered_models:
        # Evaluate and add the first model
        model_to_add = discovered_models[0]
        print(f"\nEvaluating and potentially adding model: {model_to_add['name']}")
        
        updated_config = mistral.evaluate_and_add_model(model_to_add)
        
        if updated_config:
            print("Model was promising and added to configuration")
            print(f"Updated Mistral configuration:")
            print(json.dumps(updated_config.get("mistral", {}), indent=2))
        else:
            print("Model was not promising enough to add to configuration")


if __name__ == "__main__":
    # Run the tests
    test_rag_fallback()
    time.sleep(1)  # Add a small delay between tests
    
    test_model_discovery()
    time.sleep(1)
    
    test_router_model_discovery()
    time.sleep(1)
    
    test_mistral_model_discovery()
    
    print("\nAll tests completed!")
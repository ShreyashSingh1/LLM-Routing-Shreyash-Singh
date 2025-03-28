"""Model discovery integration for the Dynamic LLM Router."""

from typing import Dict, Any, List, Optional, Tuple
import time
import logging

from models.model_discovery import ModelDiscovery
from config import MODEL_CONFIGS, ROUTING_CONFIG


class RouterModelDiscovery:
    """Model discovery integration for the Dynamic LLM Router.
    
    This class integrates the model discovery functionality with the router,
    allowing it to discover, evaluate, and add new models to the configuration.
    """
    
    def __init__(self):
        """Initialize the router model discovery integration."""
        self.model_discovery = ModelDiscovery()
        self.model_configs = MODEL_CONFIGS
        self.routing_config = ROUTING_CONFIG
        self.discovery_log = []
    
    def discover_llama_variants(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Discover and evaluate new LLaMA variants.
        
        Returns:
            Tuple containing (list of promising models, updated configuration or None)
        """
        # Log discovery attempt
        self.discovery_log.append({
            "timestamp": time.time(),
            "action": "discover_llama_variants",
            "status": "started"
        })
        
        # Discover and evaluate LLaMA variants
        promising_models, updated_config = self.model_discovery.discover_and_evaluate(
            category="llama",
            provider_name="meta"
        )
        
        # Log discovery results
        self.discovery_log.append({
            "timestamp": time.time(),
            "action": "discover_llama_variants",
            "status": "completed",
            "promising_models_count": len(promising_models),
            "config_updated": updated_config is not None
        })
        
        return promising_models, updated_config
    
    def discover_models_for_domain(self, domain: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Discover models that are good for a specific domain.
        
        Args:
            domain: The domain to find models for (e.g., "coding", "math")
            
        Returns:
            Tuple containing (list of promising models, updated configuration or None)
        """
        # Log discovery attempt
        self.discovery_log.append({
            "timestamp": time.time(),
            "action": "discover_models_for_domain",
            "domain": domain,
            "status": "started"
        })
        
        # Discover and evaluate models from different categories
        all_promising_models = []
        updated_config = None
        
        # Try LLaMA models first
        llama_models, llama_config = self.model_discovery.discover_and_evaluate(
            category="llama",
            provider_name="meta"
        )
        
        # Then try other models
        other_models, other_config = self.model_discovery.discover_and_evaluate(
            category="other",
            provider_name="other"
        )
        
        # Combine promising models
        all_promising_models = llama_models + other_models
        
        # Filter models that have the requested domain in their strengths
        domain_specific_models = [
            model for model in all_promising_models
            if domain in model.get("strengths", [])
        ]
        
        # Use the first config update that was found
        if llama_config is not None:
            updated_config = llama_config
        elif other_config is not None:
            updated_config = other_config
        
        # Log discovery results
        self.discovery_log.append({
            "timestamp": time.time(),
            "action": "discover_models_for_domain",
            "domain": domain,
            "status": "completed",
            "promising_models_count": len(domain_specific_models),
            "config_updated": updated_config is not None
        })
        
        return domain_specific_models, updated_config
    
    def update_routing_config(self, updated_model_configs: Dict[str, Any]) -> Dict[str, Any]:
        """Update the routing configuration based on new model configurations.
        
        Args:
            updated_model_configs: Updated model configurations
            
        Returns:
            Updated routing configuration
        """
        # Create a copy of the current routing config
        updated_routing_config = self.routing_config.copy()
        
        # Update domain mappings based on new model strengths
        domains = updated_routing_config["domains"].copy()
        
        # For each provider in the updated config
        for provider_name, provider_config in updated_model_configs.items():
            # Get the provider's strengths
            strengths = provider_config.get("strengths", [])
            
            # For each strength/domain
            for domain in strengths:
                # If the domain exists in the routing config
                if domain in domains:
                    # Add the provider to the domain if not already present
                    if provider_name not in domains[domain]:
                        domains[domain].append(provider_name)
                else:
                    # Create a new domain entry
                    domains[domain] = [provider_name]
        
        # Update the domains in the routing config
        updated_routing_config["domains"] = domains
        
        return updated_routing_config
    
    def get_discovery_log(self) -> List[Dict[str, Any]]:
        """Get the discovery log.
        
        Returns:
            List of discovery log entries
        """
        return self.discovery_log
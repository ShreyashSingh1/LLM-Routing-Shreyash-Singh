# Dynamic LLM Router Documentation

This document provides an overview of the Dynamic LLM Router system, its components, and how to extend it.

## System Overview

The Dynamic LLM Router is an intelligent routing system that directs queries to the most appropriate Large Language Model (LLM) based on query characteristics, model capabilities, and historical performance. The system includes features like:

- Query analysis to extract key characteristics
- Dynamic model selection based on multiple criteria
- Performance tracking and feedback loop
- Extensible architecture for adding new models
- Configurable routing policies
- RAG-style knowledge base for fallback responses
- Model discovery system to find and evaluate new LLMs
- Automatic configuration updates for promising models

## Architecture Components

### Router Components
- analyzer.py: Analyzes incoming queries to determine their characteristics
- selector.py: Selects the most appropriate model based on query analysis
- feedback.py: Tracks model performance and updates routing policies
- discovery.py: Integrates with model discovery system
- graph.py: Defines the LangGraph workflow for routing

### Model Components
- base.py: Defines the base interface for all models
- Various model implementations (openai.py, anthropic.py, mistral.py, google.py)
- knowledge_base.py: Implements RAG-style knowledge retrieval
- model_discovery.py: Discovers and evaluates new LLM models

## Extending the System

### Adding a New LLM Provider
1. Create a new model implementation in the `models/` directory
2. Update the model selection logic in `router/selector.py`
3. Add any new routing criteria to `router/analyzer.py`

### Customizing Routing Logic
The routing logic can be customized by modifying the query analysis in `router/analyzer.py` and the model selection criteria in `router/selector.py`.

### Improving the Knowledge Base
The RAG-style knowledge base can be extended by adding new documents and improving the retrieval mechanism in `models/knowledge_base.py`.

## Usage Notes

The system is designed to be used as a component in larger applications that require intelligent routing between multiple LLM providers. It can be integrated via the main application entry point in `app.py`.
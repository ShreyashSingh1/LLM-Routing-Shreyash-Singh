# Dynamic LLM Router

A dynamic routing system that intelligently directs queries to the most appropriate Large Language Model (LLM) provider based on query characteristics, using LangGraph for workflow orchestration.

## Overview

This project implements a smart routing mechanism that analyzes incoming queries and selects the optimal LLM provider based on factors such as:

- Query complexity
- Required expertise domain
- Performance requirements
- Cost considerations
- Historical performance

The system uses LangGraph to create a flexible, maintainable workflow that can be easily extended with additional models and routing criteria. It also includes a RAG (Retrieval-Augmented Generation) system for knowledge-based fallbacks and a model discovery feature to dynamically find and evaluate new LLM models.

## Features

- Query analysis to extract key characteristics
- Dynamic model selection based on multiple criteria
- Performance tracking and feedback loop
- Extensible architecture for adding new models
- Configurable routing policies
- RAG-style knowledge base for fallback responses
- Model discovery system to find and evaluate new LLMs
- Automatic configuration updates for promising models

## Project Structure

```
├── README.md
├── requirements.txt
├── config.py            # Configuration settings
├── router/
│   ├── __init__.py
│   ├── analyzer.py      # Query analysis component
│   ├── selector.py      # Model selection logic
│   ├── feedback.py      # Performance tracking
│   ├── discovery.py     # Router model discovery integration
│   └── graph.py         # LangGraph workflow definition
├── models/
│   ├── __init__.py
│   ├── base.py          # Base model interface
│   ├── openai.py        # OpenAI model implementation
│   ├── anthropic.py     # Anthropic model implementation
│   ├── mistral.py       # Mistral model implementation
│   ├── google.py        # Google model implementation
│   ├── knowledge_base.py # RAG-style knowledge base
│   └── model_discovery.py # Model discovery system
├── test_rag_and_discovery.py # Test script for RAG and discovery features
└── app.py               # Main application entry point
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from router import create_router

# Initialize the router
llm_router = create_router()

# Route a query to the best LLM
response = llm_router.route("What is the capital of France?")
print(response)
```

### Using the RAG System

```python
from models.mistral import MistralProvider
from config import MODEL_CONFIGS

# Initialize Mistral provider
mistral = MistralProvider(MODEL_CONFIGS["mistral"])

# Generate response with RAG fallback
response = mistral.generate("What is the Pythagorean theorem?", use_fallback=True)
print(response["content"])
print(response.get("rag_info", {}))
```

### Using Model Discovery

```python
from models.model_discovery import ModelDiscovery
from router.discovery import RouterModelDiscovery

# Initialize model discovery
model_discovery = ModelDiscovery()

# Search for LLaMA models
llama_models = model_discovery.search_models(category="llama")

# Evaluate a model
model_to_evaluate = llama_models[0]
evaluation_scores = model_discovery.evaluate_model(model_to_evaluate)

# Check if the model is promising and add to config if it is
if model_discovery.is_promising(evaluation_scores):
    updated_config = model_discovery.add_to_config(model_to_evaluate, "meta")

# Or use the router discovery integration
router_discovery = RouterModelDiscovery()
promising_models, updated_config = router_discovery.discover_llama_variants()
```

## Extending

### Adding a New LLM Provider

1. Create a new model implementation in the `models/` directory
2. Update the model selection logic in `router/selector.py`
3. Add any new routing criteria to `router/analyzer.py`

### Enhancing the RAG System

1. Modify the `KnowledgeBase` class in `models/knowledge_base.py`
2. Add new knowledge domains or connect to a vector database
3. Implement more sophisticated retrieval methods

### Improving Model Discovery

1. Extend the `ModelDiscovery` class in `models/model_discovery.py`
2. Add connections to real model repositories or APIs
3. Implement more rigorous evaluation methods

## License

MIT
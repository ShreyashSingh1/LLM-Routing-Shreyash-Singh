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

### Core Routing
- Query analysis to extract key characteristics
- Dynamic model selection based on multiple criteria
- Performance tracking and feedback loop
- Extensible architecture for adding new models
- Configurable routing policies

### RAG System
- Knowledge base integration for enhanced responses
- Fallback mechanism when models lack specific knowledge
- Domain-specific document retrieval
- Contextual augmentation of model responses

### Model Discovery
- Automatic discovery of new LLM models
- Performance evaluation against benchmarks
- Integration of promising models into the routing system
- Provider-specific model variant discovery
- Configuration updates for new models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Dynamic-LLM-Router.git
cd Dynamic-LLM-Router

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (for production use)
echo "OPENAI_API_KEY=your_openai_key" > .env
echo "ANTHROPIC_API_KEY=your_anthropic_key" >> .env
echo "MISTRAL_API_KEY=your_mistral_key" >> .env
echo "GOOGLE_API_KEY=your_google_key" >> .env
```

## Usage

### Basic Usage

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
if llama_models:
    model_to_evaluate = llama_models[0]
    evaluation_scores = model_discovery.evaluate_model(model_to_evaluate)
    
    # Check if the model is promising
    is_promising = model_discovery.is_promising(evaluation_scores)
    
    # Add to config if promising
    if is_promising:
        updated_config = model_discovery.add_to_config(model_to_evaluate, "meta")
```

### Running the Web Interface

The project includes a Streamlit web interface for testing the router, RAG system, and model discovery features:

```bash
# Run the Streamlit app
streamlit run app.py
```

This will launch a web interface where you can:
- Submit queries to the router
- View query analysis and provider selection
- Test the RAG system with various domains
- Explore model discovery features

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

## Testing

The project includes a test script for the RAG system and model discovery features:

```bash
# Run the test script
python test_rag_and_discovery.py
```

This will test:
- RAG-style fallback for different domains
- Model discovery for LLaMA variants
- Model evaluation and configuration updates

## Extending the System

### Adding a New LLM Provider

1. Create a new model implementation in the `models/` directory:
   ```python
   from models.base import BaseProvider
   
   class NewProvider(BaseProvider):
       def __init__(self, config):
           super().__init__(config)
           # Initialize provider-specific components
       
       def generate(self, query, use_fallback=False):
           # Implement generation logic
           pass
   ```

2. Update the model selection logic in `router/selector.py`
3. Add any new routing criteria to `router/analyzer.py`

### Customizing Routing Logic

The routing logic can be customized by modifying:
- Query analysis in `router/analyzer.py`
- Model selection criteria in `router/selector.py`
- The LangGraph workflow in `router/graph.py`

### Improving the Knowledge Base

The RAG-style knowledge base can be extended by:
- Adding new documents to the knowledge base
- Improving the retrieval mechanism in `models/knowledge_base.py`
- Customizing the document processing pipeline

## Dependencies

- langchain (>= 0.1.0)
- langchain-core (>= 0.1.0)
- langgraph (>= 0.0.15)
- streamlit (>= 1.28.0)
- pydantic (>= 2.0.0)
- python-dotenv (>= 1.0.0)
- numpy (>= 1.24.0)
- scikit-learn (>= 1.3.0)

## License

MIT
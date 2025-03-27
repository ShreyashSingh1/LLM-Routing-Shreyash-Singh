# Dynamic LLM Router

A dynamic routing system that intelligently directs queries to the most appropriate Large Language Model (LLM) provider based on query characteristics, using LangGraph for workflow orchestration.

## Overview

This project implements a smart routing mechanism that analyzes incoming queries and selects the optimal LLM provider based on factors such as:

- Query complexity
- Required expertise domain
- Performance requirements
- Cost considerations
- Historical performance

The system uses LangGraph to create a flexible, maintainable workflow that can be easily extended with additional models and routing criteria.

## Features

- Query analysis to extract key characteristics
- Dynamic model selection based on multiple criteria
- Performance tracking and feedback loop
- Extensible architecture for adding new models
- Configurable routing policies

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
│   └── graph.py         # LangGraph workflow definition
├── models/
│   ├── __init__.py
│   ├── base.py          # Base model interface
│   ├── openai.py        # OpenAI model implementation
│   ├── anthropic.py     # Anthropic model implementation
│   ├── mistral.py       # Mistral model implementation
│   └── google.py        # Google model implementation
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

## Extending

To add a new LLM provider:

1. Create a new model implementation in the `models/` directory
2. Update the model selection logic in `router/selector.py`
3. Add any new routing criteria to `router/analyzer.py`

## License

MIT
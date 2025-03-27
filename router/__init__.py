"""Router package for dynamically routing queries to appropriate LLM providers."""

from router.graph import create_router

__all__ = ["create_router"]
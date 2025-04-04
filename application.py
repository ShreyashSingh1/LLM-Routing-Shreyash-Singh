"""FastAPI application for the Dynamic LLM Router."""

from typing import Dict, Any, Optional
import time
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
import os

from router import create_router
from models.base import LLMProviderError, RateLimitError, AuthenticationError, ProviderAPIError, \
    ContextLengthExceededError, TimeoutError, NetworkError, InvalidRequestError, \
    ModelNotAvailableError, ContentFilterError

# Load environment variables
load_dotenv()

# Set mock API keys if not present
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "mock-openai-key"
if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = "mock-anthropic-key"
if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = "mock-mistral-key"
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "mock-google-key"

# Create FastAPI app
app = FastAPI(
    title="Dynamic LLM Router API",
    description="API for routing queries to the most appropriate LLM provider",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create router instance
llm_router = create_router()

# Define request models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The input query/prompt to process")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context information")
    provider_override: Optional[str] = Field(None, description="Optional provider override")
    
class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="ID of the query to provide feedback for")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comments: Optional[str] = Field(None, description="Optional feedback comments")

# Error handling
@app.exception_handler(LLMProviderError)
def llm_provider_error_handler(request: Request, exc: LLMProviderError):
    status_code = 500
    
    # Map specific errors to appropriate status codes
    if isinstance(exc, RateLimitError):
        status_code = 429
    elif isinstance(exc, AuthenticationError):
        status_code = 401
    elif isinstance(exc, InvalidRequestError):
        status_code = 400
    elif isinstance(exc, ContextLengthExceededError):
        status_code = 413
    elif isinstance(exc, ModelNotAvailableError):
        status_code = 503
    elif isinstance(exc, TimeoutError) or isinstance(exc, NetworkError):
        status_code = 504
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "details": getattr(exc, "details", {})
            }
        }
    )

# API endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the Dynamic LLM Router API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/query")
def process_query(request: QueryRequest):
    """Process a query through the LLM router."""
    start_time = time.time()
    
    # Route the query
    response = llm_router.route(request.query, request.context)
    
    # Add timing information
    end_time = time.time()
    if "metadata" not in response:
        response["metadata"] = {}
    response["metadata"]["api_latency"] = end_time - start_time
    
    return response

@app.get("/providers")
def list_providers():
    """List available LLM providers and their configurations."""
    from config import MODEL_CONFIGS
    
    providers = {}
    for provider_name, config in MODEL_CONFIGS.items():
        # Filter out sensitive information
        safe_config = {
            "default_model": config.get("default_model"),
            "fallback_model": config.get("fallback_model"),
            "strengths": config.get("strengths", []),
            "response_time": config.get("response_time"),
            "cost_per_1k_tokens": config.get("cost_per_1k_tokens"),
        }
        providers[provider_name] = safe_config
    
    return {"providers": providers}

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for a previous query."""
    # Record the feedback
    success = llm_router.feedback.record_feedback(
        query_id=feedback.query_id,
        rating=feedback.rating,
        comments=feedback.comments
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Query ID not found")
    
    return {"message": "Feedback recorded successfully"}

@app.get("/cache/stats")
def get_cache_stats():
    """Get cache statistics."""
    if not llm_router.cache_enabled:
        return {"enabled": False}
    
    return {
        "enabled": True,
        "stats": llm_router.cache.get_stats()
    }

@app.post("/cache/clear")
def clear_cache():
    """Clear the query cache."""
    if not llm_router.cache_enabled:
        return {"enabled": False}
    
    llm_router.cache.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/experiments")
def get_experiments():
    """Get information about active experiments."""
    if not llm_router.experiments_enabled:
        return {"enabled": False}
    
    return {
        "enabled": True,
        "active_experiment": llm_router.experiments.active_experiment,
        "strategies": list(llm_router.experiments.strategy_registry.keys())
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run("application:app", host="0.0.0.0", port=8000, reload=True)
"""Flask application for the Dynamic LLM Router."""

from flask import Flask, jsonify, request, abort
from dotenv import load_dotenv
import os
import time
from router import create_router
from models.base import LLMProviderError

# Load environment variables
load_dotenv()

# Set mock API keys if not present
os.environ.setdefault("OPENAI_API_KEY", "mock-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "mock-anthropic-key")
os.environ.setdefault("MISTRAL_API_KEY", "mock-mistral-key")
os.environ.setdefault("GOOGLE_API_KEY", "mock-google-key")

# Create Flask app
app = Flask(__name__)

# Create router instance
llm_router = create_router()

# Error handling
@app.errorhandler(LLMProviderError)
def handle_llm_provider_error(error):
    status_code = 500
    error_type = error.__class__.__name__

    if error_type == "RateLimitError":
        status_code = 429
    elif error_type == "AuthenticationError":
        status_code = 401
    elif error_type == "InvalidRequestError":
        status_code = 400
    elif error_type == "ContextLengthExceededError":
        status_code = 413
    elif error_type == "ModelNotAvailableError":
        status_code = 503
    elif error_type in ["TimeoutError", "NetworkError"]:
        status_code = 504

    return jsonify({
        "error": {
            "type": error_type,
            "message": str(error),
            "details": getattr(error, "details", {})
        }
    }), status_code

# API endpoints
@app.route("/", methods=["GET"])
def read_root():
    return jsonify({"message": "Welcome to the Dynamic LLM Router API"})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})

@app.route("/query", methods=["POST"])
def process_query():
    """Process a query through the LLM router."""
    data = request.json
    if not data or "query" not in data:
        abort(400, description="Missing 'query' in request body")

    query = data["query"]
    context = data.get("context", None)

    start_time = time.time()
    response = llm_router.route(query, context)
    end_time = time.time()

    if "metadata" not in response:
        response["metadata"] = {}
    response["metadata"]["api_latency"] = end_time - start_time

    return jsonify(response)

@app.route("/providers", methods=["GET"])
def list_providers():
    """List available LLM providers and their configurations."""
    from config import MODEL_CONFIGS

    providers = {}
    for provider_name, config in MODEL_CONFIGS.items():
        safe_config = {
            "default_model": config.get("default_model"),
            "fallback_model": config.get("fallback_model"),
            "strengths": config.get("strengths", []),
            "response_time": config.get("response_time"),
            "cost_per_1k_tokens": config.get("cost_per_1k_tokens"),
        }
        providers[provider_name] = safe_config

    return jsonify({"providers": providers})

@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """Submit feedback for a previous query."""
    data = request.json
    if not data or "query_id" not in data or "rating" not in data:
        abort(400, description="Missing 'query_id' or 'rating' in request body")

    query_id = data["query_id"]
    rating = data["rating"]
    comments = data.get("comments", None)

    success = llm_router.feedback.record_feedback(query_id, rating, comments)
    if not success:
        abort(404, description="Query ID not found")

    return jsonify({"message": "Feedback recorded successfully"})

@app.route("/cache/stats", methods=["GET"])
def get_cache_stats():
    """Get cache statistics."""
    if not llm_router.cache_enabled:
        return jsonify({"enabled": False})

    return jsonify({
        "enabled": True,
        "stats": llm_router.cache.get_stats()
    })

@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    """Clear the query cache."""
    if not llm_router.cache_enabled:
        return jsonify({"enabled": False})

    llm_router.cache.clear()
    return jsonify({"message": "Cache cleared successfully"})

@app.route("/experiments", methods=["GET"])
def get_experiments():
    """Get information about active experiments."""
    if not llm_router.experiments_enabled:
        return jsonify({"enabled": False})

    return jsonify({
        "enabled": True,
        "active_experiment": llm_router.experiments.active_experiment,
        "strategies": list(llm_router.experiments.strategy_registry.keys())
    })

# Run the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
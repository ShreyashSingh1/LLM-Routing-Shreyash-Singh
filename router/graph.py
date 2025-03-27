"""LangGraph workflow definition for the Dynamic LLM Router."""

from typing import Dict, Any, List, Optional, Tuple, TypedDict, Annotated, Literal
import time
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from config import MODEL_CONFIGS
from router.analyzer import QueryAnalyzer
from router.selector import ModelSelector
from router.feedback import FeedbackSystem
from models import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    MistralProvider,
    GoogleProvider,
)


# Define the state schema
class RouterState(TypedDict):
    """State for the LLM Router graph."""
    query: str
    query_analysis: Optional[Dict[str, Any]]
    selected_provider: Optional[str]
    response: Optional[Dict[str, Any]]
    start_time: Optional[float]
    end_time: Optional[float]
    error: Optional[str]


def create_router():
    """Create and configure the LLM router graph.
    
    Returns:
        Configured router instance
    """
    # Initialize components
    analyzer = QueryAnalyzer()
    selector = ModelSelector()
    feedback_system = FeedbackSystem()
    
    # Initialize provider instances
    providers = {
        "openai": OpenAIProvider(MODEL_CONFIGS["openai"]),
        "anthropic": AnthropicProvider(MODEL_CONFIGS["anthropic"]),
        "mistral": MistralProvider(MODEL_CONFIGS["mistral"]),
        "google": GoogleProvider(MODEL_CONFIGS["google"]),
    }
    
    # Define the graph
    graph = StateGraph(RouterState)
    
    # Define the nodes
    
    # 1. Analyze the query
    def analyze_query(state: RouterState) -> RouterState:
        """Analyze the query to extract characteristics."""
        query = state["query"]
        query_analysis = analyzer.analyze(query)
        return {"query_analysis": query_analysis}
    
    # 2. Select the provider
    def select_provider(state: RouterState) -> RouterState:
        """Select the most appropriate provider based on query analysis."""
        query_analysis = state["query_analysis"]
        performance_history = feedback_system.get_recent_performance()
        selected_provider = selector.select_provider(query_analysis, performance_history)
        return {"selected_provider": selected_provider, "start_time": time.time()}
    
    # 3. Generate response
    def generate_response(state: RouterState) -> RouterState:
        """Generate a response using the selected provider."""
        query = state["query"]
        provider_name = state["selected_provider"]
        
        try:
            provider = providers[provider_name]
            response = provider.generate(query)
            end_time = time.time()
            
            # Record feedback
            latency = end_time - state["start_time"]
            feedback_system.record_query(query, provider_name, response, latency)
            
            return {"response": response, "end_time": end_time}
        except Exception as e:
            return {"error": str(e), "end_time": time.time()}
    
    # Add nodes to the graph
    graph.add_node("analyze", analyze_query)
    graph.add_node("select", select_provider)
    graph.add_node("generate", generate_response)
    
    # Define the edges
    graph.add_edge("analyze", "select")
    graph.add_edge("select", "generate")
    graph.add_edge("generate", END)
    
    # Set the entry point
    graph.set_entry_point("analyze")
    
    # Compile the graph
    app = graph.compile()
    
    # Create a router class with a simple interface
    class Router:
        """Dynamic LLM Router that selects the best provider for each query."""
        
        def __init__(self, graph_app):
            """Initialize the router."""
            self.graph = graph_app
            self.providers = providers
            self.analyzer = analyzer
            self.selector = selector
            self.feedback_system = feedback_system
        
        def route(self, query: str) -> Dict[str, Any]:
            """Route a query to the best LLM provider and return the response.
            
            Args:
                query: The input query string
                
            Returns:
                Response from the selected provider
            """
            print("\n===== ROUTING DECISION PROCESS =====\n")
            print(f"Input Query: {query[:100]}..." if len(query) > 100 else f"Input Query: {query}")
            
            # Step 1: Analyze the query
            print("\n1. QUERY ANALYSIS")
            query_analysis = self.analyzer.analyze(query)
            
            # Print key analysis results
            print(f"   - Complexity Score: {query_analysis['complexity_score']:.2f} ({query_analysis['complexity_category']} complexity)")
            print(f"   - Primary Domain: {query_analysis['primary_domain']}")
            print(f"   - Contains Code: {'Yes' if query_analysis['has_code'] else 'No'}")
            print(f"   - Estimated Tokens: {int(query_analysis['estimated_tokens'])}")
            
            # Print domain scores
            print("   - Domain Relevance:")
            for domain, score in sorted(query_analysis['domain_scores'].items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"     * {domain}: {score:.2f}")
            
            # Step 2: Select provider
            print("\n2. PROVIDER SELECTION")
            performance_history = self.feedback_system.get_recent_performance()
            
            # Calculate scores for each provider
            provider_scores = {}
            for provider_name in self.providers.keys():
                # Calculate individual factor scores
                complexity_score = self.selector._score_complexity(provider_name, query_analysis)
                domain_score = self.selector._score_domain_match(provider_name, query_analysis)
                performance_score = self.selector._score_performance(provider_name, performance_history)
                cost_score = self.selector._score_cost(provider_name, query_analysis)
                response_time_score = self.selector._score_response_time(provider_name, query_analysis)
                
                # Combine scores using weights
                weighted_score = (
                    self.selector.weights["query_complexity"] * complexity_score +
                    self.selector.weights["domain_match"] * domain_score +
                    self.selector.weights["performance_history"] * performance_score +
                    self.selector.weights["cost"] * cost_score +
                    self.selector.weights["response_time"] * response_time_score
                )
                
                provider_scores[provider_name] = {
                    "total": weighted_score,
                    "complexity": complexity_score,
                    "domain": domain_score,
                    "performance": performance_score,
                    "cost": cost_score,
                    "response_time": response_time_score
                }
            
            # Print scores for each provider
            for provider, scores in sorted(provider_scores.items(), key=lambda x: x[1]["total"], reverse=True):
                print(f"   - {provider}: {scores['total']:.2f} total score")
                print(f"     * Complexity match: {scores['complexity']:.2f}")
                print(f"     * Domain match: {scores['domain']:.2f}")
                print(f"     * Performance: {scores['performance']:.2f}")
                print(f"     * Cost efficiency: {scores['cost']:.2f}")
                print(f"     * Response time: {scores['response_time']:.2f}")
            
            # Select the provider with the highest score
            selected_provider = max(provider_scores.items(), key=lambda x: x[1]["total"])[0]
            print(f"\n   Selected Provider: {selected_provider}")
            print(f"   Selected Model: {self.providers[selected_provider].default_model}")
            
            # Initialize the state
            initial_state = {"query": query}
            
            # Run the graph
            print("\n3. GENERATING RESPONSE")
            start_time = time.time()
            final_state = self.graph.invoke(initial_state)
            end_time = time.time()
            
            # Check for errors
            if "error" in final_state and final_state["error"]:
                print(f"   Error: {final_state['error']}")
                return {"error": final_state["error"]}
            
            # Print token usage
            response = final_state["response"]
            print(f"   - Token Usage:")
            print(f"     * Prompt tokens: {response['tokens']['prompt']}")
            print(f"     * Completion tokens: {response['tokens']['completion']}")
            print(f"     * Total tokens: {response['tokens']['total']}")
            print(f"   - Estimated cost: ${response['cost']:.6f}")
            print(f"   - Response time: {end_time - start_time:.2f} seconds")
            print("\n===== END OF ROUTING PROCESS =====\n")
            
            # Return the response
            return final_state["response"]
        
        def add_feedback(self, query: str, provider: str, success: bool, 
                        feedback_text: Optional[str] = None) -> None:
            """Add feedback for a previous query to improve future routing.
            
            Args:
                query: The original query
                provider: The provider that handled the query
                success: Whether the response was successful
                feedback_text: Optional feedback text
            """
            feedback = {
                "success": success,
                "feedback_text": feedback_text,
                "timestamp": time.time(),
            }
            
            # Find the query in history and update its feedback
            for record in self.feedback_system.query_history:
                if record["query"] == query and record["provider"] == provider:
                    record["feedback"] = feedback
                    break
            
            # Update provider stats
            if provider in self.feedback_system.provider_stats:
                stats = self.feedback_system.provider_stats[provider]
                if success:
                    stats["success_count"] += 1
                else:
                    stats["failure_count"] += 1
                
                # Recalculate success rate
                total_feedback = stats["success_count"] + stats["failure_count"]
                if total_feedback > 0:
                    stats["success_rate"] = stats["success_count"] / total_feedback
    
    # Return the router instance
    return Router(app)

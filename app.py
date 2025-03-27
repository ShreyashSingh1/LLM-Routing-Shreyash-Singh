"""Streamlit web application for the Dynamic LLM Router."""

import time
import os
import streamlit as st
from dotenv import load_dotenv
from router import create_router

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Dynamic LLM Router",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# We're using mock implementations, so no need to check for real API keys
# Set environment variables with mock keys to avoid warnings
os.environ["OPENAI_API_KEY"] = "mock-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "mock-anthropic-key"
os.environ["MISTRAL_API_KEY"] = "mock-mistral-key"
os.environ["GOOGLE_API_KEY"] = "mock-google-key"

# Create the router
@st.cache_resource
def get_router():
    """Create and cache the LLM router."""
    return create_router()

llm_router = get_router()

# App title and description
st.title("Dynamic LLM Router")
st.markdown("""
This application dynamically routes your queries to the most appropriate LLM provider based on query characteristics.
It analyzes your input, selects the best provider, and returns the response.
""")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.info("""
    The Dynamic LLM Router analyzes your query and routes it to the most appropriate LLM provider based on:
    - Query complexity
    - Domain relevance
    - Provider performance history
    - Cost efficiency
    - Response time
    """)
    
    st.header("Available Providers")
    st.markdown("""
    - **OpenAI** (GPT-4o, GPT-3.5-Turbo)
    - **Anthropic** (Claude-3-Opus, Claude-3-Sonnet)
    - **Mistral** (Mistral-Large, Mistral-Medium)
    - **Google** (Gemini-Pro, Gemini-Flash)
    """)
    
    # Example queries
    st.header("Example Queries")
    example_queries = [
        "What is the capital of France?",  # Simple general knowledge
        "Write a poem about artificial intelligence.",  # Creative writing
        "Explain the theory of relativity in simple terms.",  # Science explanation
        "def fibonacci(n):\n    pass\n\nComplete this function to calculate the nth Fibonacci number.",  # Coding
        "What are the ethical implications of using AI in healthcare?",  # Reasoning/safety
    ]
    
    for i, query in enumerate(example_queries):
        if st.button(f"Example {i+1}", key=f"example_{i}"):
            st.session_state.query = query
            st.session_state.run_query = True

# Initialize session state variables
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'run_query' not in st.session_state:
    st.session_state.run_query = False
if 'response' not in st.session_state:
    st.session_state.response = None
if 'query_analysis' not in st.session_state:
    st.session_state.query_analysis = None
if 'provider_scores' not in st.session_state:
    st.session_state.provider_scores = None
if 'selected_provider' not in st.session_state:
    st.session_state.selected_provider = None
if 'execution_time' not in st.session_state:
    st.session_state.execution_time = None
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False

# Query input
query = st.text_area("Enter your query:", value=st.session_state.query, height=150)
submit_button = st.button("Submit")

# Process the query when the submit button is clicked
if submit_button or st.session_state.run_query:
    st.session_state.query = query
    st.session_state.run_query = False
    st.session_state.feedback_given = False
    
    if st.session_state.query:
        with st.spinner("Routing your query to the best LLM provider..."):
            # Capture the start time
            start_time = time.time()
            
            # Step 1: Analyze the query
            query_analysis = llm_router.analyzer.analyze(st.session_state.query)
            st.session_state.query_analysis = query_analysis
            
            # Step 2: Select provider
            performance_history = llm_router.feedback_system.get_recent_performance()
            
            # Calculate scores for each provider
            provider_scores = {}
            for provider_name in llm_router.providers.keys():
                # Calculate individual factor scores
                complexity_score = llm_router.selector._score_complexity(provider_name, query_analysis)
                domain_score = llm_router.selector._score_domain_match(provider_name, query_analysis)
                performance_score = llm_router.selector._score_performance(provider_name, performance_history)
                cost_score = llm_router.selector._score_cost(provider_name, query_analysis)
                response_time_score = llm_router.selector._score_response_time(provider_name, query_analysis)
                
                # Combine scores using weights
                weighted_score = (
                    llm_router.selector.weights["query_complexity"] * complexity_score +
                    llm_router.selector.weights["domain_match"] * domain_score +
                    llm_router.selector.weights["performance_history"] * performance_score +
                    llm_router.selector.weights["cost"] * cost_score +
                    llm_router.selector.weights["response_time"] * response_time_score
                )
                
                provider_scores[provider_name] = {
                    "total": weighted_score,
                    "complexity": complexity_score,
                    "domain": domain_score,
                    "performance": performance_score,
                    "cost": cost_score,
                    "response_time": response_time_score
                }
            
            st.session_state.provider_scores = provider_scores
            
            # Select the provider with the highest score
            selected_provider = max(provider_scores.items(), key=lambda x: x[1]["total"])[0]
            st.session_state.selected_provider = selected_provider
            
            # Step 3: Generate response
            try:
                response = llm_router.route(st.session_state.query)
                st.session_state.response = response
                
                # Calculate execution time
                end_time = time.time()
                st.session_state.execution_time = end_time - start_time
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Display results if a response exists
if st.session_state.response:
    # Create tabs for different sections
    tabs = st.tabs(["Response", "Routing Analysis", "Provider Comparison"])
    
    # Tab 1: Response
    with tabs[0]:
        st.subheader("LLM Response")
        st.markdown(f"**Selected Provider:** {st.session_state.response.get('provider', 'unknown')}")
        st.markdown(f"**Model:** {st.session_state.response.get('model', 'unknown')}")
        st.markdown("**Response:**")
        st.markdown(st.session_state.response.get('content', 'No content'))
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Response Time", f"{st.session_state.execution_time:.2f} sec")
        with col2:
            st.metric("Total Tokens", st.session_state.response.get('tokens', {}).get('total', 0))
        with col3:
            st.metric("Estimated Cost", f"${st.session_state.response.get('cost', 0):.6f}")
        
        # Feedback section
        if not st.session_state.feedback_given:
            st.subheader("Provide Feedback")
            st.markdown("Was this response helpful?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Yes"):
                    llm_router.add_feedback(
                        st.session_state.query, 
                        st.session_state.response.get('provider'), 
                        True, 
                        "User found the response helpful"
                    )
                    st.session_state.feedback_given = True
                    st.success("Thank you for your feedback!")
            with col2:
                if st.button("👎 No"):
                    llm_router.add_feedback(
                        st.session_state.query, 
                        st.session_state.response.get('provider'), 
                        False, 
                        "User found the response unhelpful"
                    )
                    st.session_state.feedback_given = True
                    st.success("Thank you for your feedback!")
    
    # Tab 2: Routing Analysis
    with tabs[1]:
        if st.session_state.query_analysis:
            st.subheader("Query Analysis")
            
            # Query characteristics
            st.markdown("### Query Characteristics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Complexity Score", 
                    f"{st.session_state.query_analysis['complexity_score']:.2f}",
                    f"{st.session_state.query_analysis['complexity_category']} complexity"
                )
            with col2:
                st.metric("Primary Domain", st.session_state.query_analysis['primary_domain'])
            with col3:
                st.metric("Contains Code", "Yes" if st.session_state.query_analysis['has_code'] else "No")
            
            # Domain relevance
            st.markdown("### Domain Relevance")
            domain_scores = st.session_state.query_analysis['domain_scores']
            domain_df = {"Domain": [], "Score": []}
            for domain, score in sorted(domain_scores.items(), key=lambda x: x[1], reverse=True):
                domain_df["Domain"].append(domain)
                domain_df["Score"].append(score)
            
            # Create a bar chart for domain scores
            st.bar_chart(domain_df, x="Domain", y="Score", height=300)
    
    # Tab 3: Provider Comparison
    with tabs[2]:
        if st.session_state.provider_scores:
            st.subheader("Provider Comparison")
            
            # Create a dataframe for the scores
            provider_df = {"Provider": [], "Total Score": [], "Complexity": [], 
                          "Domain": [], "Performance": [], "Cost": [], "Response Time": []}
            
            for provider, scores in sorted(st.session_state.provider_scores.items(), 
                                          key=lambda x: x[1]["total"], reverse=True):
                provider_df["Provider"].append(provider)
                provider_df["Total Score"].append(scores["total"])
                provider_df["Complexity"].append(scores["complexity"])
                provider_df["Domain"].append(scores["domain"])
                provider_df["Performance"].append(scores["performance"])
                provider_df["Cost"].append(scores["cost"])
                provider_df["Response Time"].append(scores["response_time"])
            
            # Display as a table
            st.dataframe(provider_df, hide_index=True)
            
            # Create a bar chart for total scores
            st.markdown("### Total Provider Scores")
            chart_df = {"Provider": provider_df["Provider"], "Score": provider_df["Total Score"]}
            st.bar_chart(chart_df, x="Provider", y="Score", height=300)
            
            # Highlight the selected provider
            st.markdown(f"**Selected Provider:** {st.session_state.selected_provider}")
            st.markdown(f"**Selected Model:** {llm_router.providers[st.session_state.selected_provider].default_model}")

# Footer
st.markdown("---")
st.markdown("Dynamic LLM Router - Intelligently routing queries to the best LLM provider")
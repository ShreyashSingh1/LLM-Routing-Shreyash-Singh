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
    
    # A/B Testing Experiments section in sidebar
    if llm_router.experiments_enabled:
        st.header("A/B Testing")
        
        # Get all experiments and active experiment
        all_experiments = llm_router.experiments.get_all_experiments()
        active_experiment = llm_router.experiments.active_experiment
        
        # Display active experiment if any
        if active_experiment:
            st.success(f"Active experiment: {active_experiment}")
            if st.button("Stop Experiment"):
                llm_router.stop_experiment(active_experiment)
                st.experimental_rerun()
        else:
            st.info("No active experiment")
            
            # Create new experiment section
            with st.expander("Create New Experiment"):
                exp_name = st.text_input("Experiment Name", key="new_exp_name")
                
                # Get available strategies
                strategies = list(llm_router.experiments.strategy_registry.keys())
                if not strategies:
                    strategies = ["standard"]  # Fallback to standard if no strategies registered
                
                # Let user select strategies to compare
                selected_strategies = st.multiselect(
                    "Select Strategies to Compare", 
                    options=strategies,
                    default=["standard"] if "standard" in strategies else []
                )
                
                # Traffic split sliders
                if selected_strategies:
                    st.write("Traffic Split (must sum to 100%)")
                    traffic_values = {}
                    remaining = 100
                    
                    for i, strategy in enumerate(selected_strategies[:-1]):
                        max_val = remaining if i == len(selected_strategies) - 2 else 100
                        val = st.slider(f"{strategy} %", 0, max_val, 50 if i == 0 else 0, 5)
                        traffic_values[strategy] = val / 100.0
                        remaining -= val
                    
                    # Last strategy gets the remainder
                    if selected_strategies:
                        last_strategy = selected_strategies[-1]
                        traffic_values[last_strategy] = remaining / 100.0
                        st.write(f"{last_strategy}: {remaining}%")
                    
                    # Create experiment button
                    if st.button("Create Experiment"):
                        if exp_name and len(selected_strategies) >= 2:
                            try:
                                llm_router.create_experiment(exp_name, selected_strategies, traffic_values)
                                st.success(f"Experiment '{exp_name}' created!")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error creating experiment: {str(e)}")
                        else:
                            st.error("Please provide a name and select at least 2 strategies")
            
            # Start existing experiment section
            if all_experiments:
                with st.expander("Start Existing Experiment"):
                    experiment_names = [name for name, info in all_experiments.items() 
                                      if info.get("status") != "active"]
                    if experiment_names:
                        selected_exp = st.selectbox("Select Experiment", experiment_names)
                        if st.button("Start Experiment"):
                            llm_router.start_experiment(selected_exp)
                            st.success(f"Experiment '{selected_exp}' started!")
                            st.experimental_rerun()
                    else:
                        st.info("No inactive experiments available")
    
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
            performance_history = llm_router.feedback.get_recent_performance()
            
            # Calculate scores for each provider
            provider_scores = {}
            # Use all providers from MODEL_CONFIGS instead of just initialized providers
            for provider_name in llm_router.selector.model_configs.keys():
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
            if provider_scores:
                selected_provider = max(provider_scores.items(), key=lambda x: x[1]["total"])[0]
                st.session_state.selected_provider = selected_provider
            else:
                # Use default provider if no providers are available
                selected_provider = llm_router.selector.default_provider
                st.session_state.selected_provider = selected_provider
                st.warning("No providers available. Using default provider.")
            
            # Step 3: Generate response
            try:
                # Ensure we have at least one provider available
                if not llm_router.providers:
                    # Initialize the default provider if none exists
                    provider_name = st.session_state.selected_provider
                    if provider_name not in llm_router.providers:
                        from models import create_provider
                        llm_router.providers[provider_name] = create_provider(provider_name)
                
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
    tabs = st.tabs(["Response", "Routing Analysis", "Provider Comparison", "A/B Testing", "Cache"])
    
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
                    
                    # Add feedback to experiment metrics if an experiment is active
                    if llm_router.experiments_enabled and llm_router.experiments.active_experiment:
                        # Record positive feedback in experiment metrics
                        strategy = st.session_state.response.get('metadata', {}).get('strategy', 'standard')
                        llm_router.experiments.record_metrics(
                            strategy=strategy,
                            query=st.session_state.query,
                            metrics={
                                "user_rating": 5.0,  # 5-star rating for positive feedback
                                "success": True
                            }
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
                    
                    # Add feedback to experiment metrics if an experiment is active
                    if llm_router.experiments_enabled and llm_router.experiments.active_experiment:
                        # Record negative feedback in experiment metrics
                        strategy = st.session_state.response.get('metadata', {}).get('strategy', 'standard')
                        llm_router.experiments.record_metrics(
                            strategy=strategy,
                            query=st.session_state.query,
                            metrics={
                                "user_rating": 1.0,  # 1-star rating for negative feedback
                                "success": False
                            }
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
    
    # Tab 4: A/B Testing
    with tabs[3]:
        st.subheader("A/B Testing Experiments")
        
        # Check if experiments are enabled
        if not llm_router.experiments_enabled:
            st.info("A/B Testing experiments are not enabled in the configuration.")
        else:
            # Get active experiment information
            active_experiment = llm_router.experiments.active_experiment
            
            if active_experiment:
                # Get experiment results
                results = llm_router.get_experiment_results(active_experiment)
                
                # Display experiment information
                st.markdown(f"**Active Experiment:** {active_experiment}")
                st.markdown(f"**Status:** {results.get('status', 'Unknown')}")
                st.markdown(f"**Total Queries:** {results.get('total_queries', 0)}")
                st.markdown(f"**Duration:** {results.get('duration', 0):.2f} seconds")
                
                # Display metrics comparison
                st.markdown("### Strategy Comparison")
                
                metrics = results.get('metrics', {})
                if metrics:
                    # Create dataframes for visualization
                    strategies = list(metrics.keys())
                    
                    # Success rate comparison
                    success_df = {"Strategy": [], "Success Rate": []}
                    for strategy, data in metrics.items():
                        success_df["Strategy"].append(strategy)
                        success_df["Success Rate"].append(data.get('success_rate', 0))
                    
                    st.markdown("#### Success Rate by Strategy")
                    st.bar_chart(success_df, x="Strategy", y="Success Rate", height=300)
                    
                    # Latency comparison
                    latency_df = {"Strategy": [], "Average Latency (s)": []}
                    for strategy, data in metrics.items():
                        latency_df["Strategy"].append(strategy)
                        latency_df["Average Latency (s)"].append(data.get('avg_latency', 0))
                    
                    st.markdown("#### Average Latency by Strategy")
                    st.bar_chart(latency_df, x="Strategy", y="Average Latency (s)", height=300)
                    
                    # User rating comparison
                    rating_df = {"Strategy": [], "Average User Rating": []}
                    for strategy, data in metrics.items():
                        rating_df["Strategy"].append(strategy)
                        rating_df["Average User Rating"].append(data.get('avg_user_rating', 0))
                    
                    st.markdown("#### Average User Rating by Strategy")
                    st.bar_chart(rating_df, x="Strategy", y="Average User Rating", height=300)
                    
                    # Detailed metrics table
                    st.markdown("### Detailed Metrics")
                    
                    # Create a dataframe for all metrics
                    metrics_df = {"Metric": []}
                    for strategy in strategies:
                        metrics_df[strategy] = []
                    
                    # Add all available metrics
                    all_metric_keys = set()
                    for strategy_metrics in metrics.values():
                        all_metric_keys.update(strategy_metrics.keys())
                    
                    for metric_key in sorted(all_metric_keys):
                        if metric_key != "message":  # Skip non-numeric messages
                            metrics_df["Metric"].append(metric_key)
                            for strategy in strategies:
                                strategy_data = metrics.get(strategy, {})
                                metrics_df[strategy].append(strategy_data.get(metric_key, "N/A"))
                    
                    st.dataframe(metrics_df, hide_index=True)
                else:
                    st.info("No metrics recorded for this experiment yet.")
            else:
                st.info("No active experiment. Start an experiment to see results here.")
                
                # Show available experiments
                all_experiments = llm_router.experiments.get_all_experiments()
                if all_experiments:
                    st.markdown("### Available Experiments")
                    for name, exp_info in all_experiments.items():
                        st.markdown(f"**{name}**")
                        st.markdown(f"Status: {exp_info.get('status', 'Unknown')}")
                        st.markdown(f"Strategies: {', '.join(exp_info.get('strategies', []))}")

    # Tab 5: Cache
    with tabs[4]:
        st.subheader("Cache Statistics")
        
        # Get cache statistics
        try:
            cache_stats = llm_router.get_cache_stats()
            
            # Display cache metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cache Size", f"{cache_stats['size']} / {cache_stats['max_size']}")
            with col2:
                st.metric("Hit Rate", f"{cache_stats['hit_rate']:.2%}")
            with col3:
                st.metric("Hits", cache_stats['hits'])
            with col4:
                st.metric("Misses", cache_stats['misses'])
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Evictions", cache_stats['evictions'])
            with col2:
                st.metric("Expirations", cache_stats['expirations'])
            with col3:
                st.metric("Strategy", cache_stats['strategy'])
            
            # Create a pie chart for cache hit/miss ratio
            if cache_stats['hits'] > 0 or cache_stats['misses'] > 0:
                st.markdown("### Cache Performance")
                hit_miss_data = {
                    "Category": ["Hits", "Misses"],
                    "Count": [cache_stats['hits'], cache_stats['misses']]
                }
                
                # Display the pie chart
                st.bar_chart(hit_miss_data, x="Category", y="Count")
                
            # Cache entries table - get entries directly from cache object
            try:
                cache_entries = llm_router.cache.get_entries(limit=10)
                if cache_entries:
                    st.markdown("### Recent Cache Entries")
                    entries_df = {"Query": [], "Provider": [], "Age": [], "TTL": []}
                    
                    for entry in cache_entries:  # Show only the 10 most recent entries
                        entries_df["Query"].append(entry.get("query", "Unknown")[:50] + "...")
                        entries_df["Provider"].append(entry.get("provider", "Unknown"))
                        entries_df["Age"].append(f"{entry.get('age', 0):.1f} sec")
                        entries_df["TTL"].append(f"{entry.get('ttl', 0):.1f} sec")
                    
                    st.dataframe(entries_df, hide_index=True)
            except Exception as e:
                st.warning(f"Could not retrieve cache entries: {str(e)}")
                
                # Cache control buttons
                if st.button("Clear Cache"):
                    llm_router.clear_cache()
                    st.success("Cache cleared successfully!")
                    st.experimental_rerun()
        except Exception as e:
            st.error(f"Error retrieving cache statistics: {str(e)}")

        # Footer
        st.markdown("---")
        st.markdown("Dynamic LLM Router - Intelligently routing queries to the best LLM provider")
        st.subheader("Cache Statistics")
        
        # Get cache statistics
        try:
            cache_stats = llm_router.get_cache_stats()
            
            # Display cache metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cache Size", f"{cache_stats['size']} / {cache_stats['max_size']}")
            with col2:
                st.metric("Hit Rate", f"{cache_stats['hit_rate']:.2%}")
            with col3:
                st.metric("Hits", cache_stats['hits'])
            with col4:
                st.metric("Misses", cache_stats['misses'])
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Evictions", cache_stats['evictions'])
            with col2:
                st.metric("Expirations", cache_stats['expirations'])
            with col3:
                st.metric("Strategy", cache_stats['strategy'])
            
            # Create a pie chart for cache hit/miss ratio
            if cache_stats['hits'] > 0 or cache_stats['misses'] > 0:
                st.markdown("### Cache Performance")
                hit_miss_data = {
                    "Category": ["Hits", "Misses"],
                    "Count": [cache_stats['hits'], cache_stats['misses']]
                }
                st.bar_chart(hit_miss_data, x="Category", y="Count")
            
            # Cache entries section
            st.markdown("### Cache Entries")
            
            # Get cache entries (limited to 100)
            try:
                cache_entries = llm_router.cache.get_entries(limit=100)
                
                if cache_entries:
                    # Create an expander for cache entries
                    with st.expander("View Cache Entries"):
                        # Create a dataframe for display
                        entries_data = {
                            "Query": [],
                            "Provider": [],
                            "Age (s)": [],
                            "Expires In (s)": [],
                            "Access Count": [],
                            "Status": []
                        }
                        
                        for entry in cache_entries:
                            entries_data["Query"].append(entry["query"][:50] + "..." if len(entry["query"]) > 50 else entry["query"])
                            entries_data["Provider"].append(entry["provider"])
                            entries_data["Age (s)"].append(round(entry["age"], 1))
                            expires_in = max(0, entry["expires_at"] - time.time())
                            entries_data["Expires In (s)"].append(round(expires_in, 1))
                            entries_data["Access Count"].append(entry["access_count"])
                            entries_data["Status"].append("Expired" if entry["is_expired"] else "Active")
                        
                        # Display as a table
                        st.dataframe(entries_data, hide_index=True)
                else:
                    st.info("No cache entries found.")
            except Exception as e:
                st.error(f"Error retrieving cache entries: {str(e)}")
            
            # Cache management section
            st.markdown("### Cache Management")
            if st.button("Clear Cache"):
                llm_router.cache.clear()
                st.success("Cache cleared successfully!")
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Error retrieving cache statistics: {str(e)}")
            st.info("Cache may not be enabled in the configuration.")
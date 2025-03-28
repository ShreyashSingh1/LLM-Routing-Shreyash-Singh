import streamlit as st
import json
import time
from typing import Dict, Any, List

from models.mistral import MistralProvider
from models.knowledge_base import KnowledgeBase
from models.model_discovery import ModelDiscovery
from router.discovery import RouterModelDiscovery
from config import MODEL_CONFIGS

# Set page configuration
st.set_page_config(
    page_title="Dynamic LLM Router - RAG & Discovery",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add sidebar with app information
st.sidebar.title("Dynamic LLM Router")
st.sidebar.markdown("### Test RAG and Model Discovery")
st.sidebar.markdown("""
### How to use this app:
1. Select a tab at the top to test different features
2. Follow the step-by-step instructions in each tab
3. Use the buttons to trigger actions and see results

### Features demonstrated:
- RAG-style fallback responses
- Model discovery capabilities
- Router model discovery integration
- Mistral provider's model discovery
""")

# Main page title
st.title("Dynamic LLM Router - RAG & Discovery Testing")

# Add a welcome message and overview
st.markdown("""
👋 **Welcome to the Dynamic LLM Router testing interface!**

This application helps you understand how the Dynamic LLM Router works by demonstrating its key features.
Use the tabs below to explore different capabilities of the system.
""")

# Create tabs for different test categories with descriptive labels
tabs = st.tabs(["📚 RAG Fallback", "🔍 Model Discovery", "🔄 Router Model Discovery", "🤖 Mistral Model Discovery"])

# Tab 1: RAG Fallback
with tabs[0]:
    st.header("Test RAG-style Fallback")
    st.markdown("""
    ### What is RAG?
    RAG (Retrieval-Augmented Generation) helps LLMs answer questions by retrieving relevant information 
    from a knowledge base when the model doesn't have the answer.
    
    ### How to use this tab:
    1. Select a domain for your query using the dropdown below
    2. The text area will be populated with a sample query (you can modify it)
    3. Click the "Generate Response" button to see the result
    4. Review both the response and the RAG information that shows what knowledge was retrieved
    """)
    
    # Domain selection
    domain = st.selectbox(
        "Select a domain for your query:",
        ["general_knowledge", "coding", "math", "science"],
        index=0
    )
    
    # Predefined queries based on domain
    domain_queries = {
        "general_knowledge": "What is the capital of France?",
        "coding": "Explain what a function is in programming.",
        "math": "What is the Pythagorean theorem?",
        "science": "Explain the theory of relativity."
    }
    
    # Query input with default based on domain
    query = st.text_area("Enter your query:", value=domain_queries[domain], height=100)
    
    # Generate button
    if st.button("Generate Response", key="rag_generate"):
        with st.spinner("Generating response..."):
            # Initialize Mistral provider
            mistral_config = MODEL_CONFIGS["mistral"]
            mistral = MistralProvider(mistral_config)
            
            # Generate response with fallback (RAG)
            response = mistral.generate(query, use_fallback=True)
            
            # Display response
            st.subheader("Response:")
            st.write(response["content"])
            
            # Display RAG info
            st.subheader("RAG Information:")
            rag_info = response.get("rag_info", {})
            if rag_info:
                st.json(rag_info)
            else:
                st.info("No RAG information available for this response.")

# Tab 2: Model Discovery
with tabs[1]:
    st.header("Test Model Discovery")
    st.markdown("""
    ### What is Model Discovery?
    Model Discovery allows the system to find and evaluate new language models that might be suitable for specific tasks.
    
    ### How to use this tab:
    1. Select a model category from the dropdown (e.g., llama, gpt, claude)
    2. Click "Search Models" to find models in that category
    3. Review the model cards that appear
    4. Click "Evaluate" on any model to see its performance metrics
    5. If a model is promising, you can add it to the configuration
    """)
    
    # Category selection
    category = st.selectbox(
        "Select a model category to search:",
        ["llama", "gpt", "claude", "palm", "mistral"],
        index=0
    )
    
    # Search button
    if st.button("Search Models", key="model_search"):
        with st.spinner(f"Searching for {category} models..."):
            # Initialize model discovery
            model_discovery = ModelDiscovery()
            
            # Search for models
            models = model_discovery.search_models(category=category)
            
            # Display results
            st.subheader(f"Found {len(models)} {category} models:")
            
            if models:
                # Create columns for model cards
                cols = st.columns(min(3, len(models)))
                
                for i, model in enumerate(models):
                    with cols[i % min(3, len(models))]:
                        st.markdown(f"**Model {i+1}: {model['name']}**")
                        st.markdown(f"**Provider:** {model['provider']}")
                        st.markdown(f"**Size:** {model['size']}")
                        st.markdown(f"**Strengths:** {', '.join(model['strengths'])}")
                        st.markdown(f"**Cost per 1k tokens:** ${model['cost_per_1k_tokens']}")
                        st.markdown(f"**Response time:** {model['response_time']}")
                        
                        # Add evaluate button for each model
                        if st.button(f"Evaluate {model['name']}", key=f"eval_{i}"):
                            with st.spinner(f"Evaluating {model['name']}..."):
                                # Evaluate model
                                evaluation_scores = model_discovery.evaluate_model(model)
                                
                                # Display evaluation scores
                                st.subheader(f"Evaluation scores for {model['name']}:")
                                for metric, score in evaluation_scores.items():
                                    st.markdown(f"**{metric}:** {score:.2f}")
                                
                                # Check if the model is promising
                                is_promising = model_discovery.is_promising(evaluation_scores)
                                if is_promising:
                                    st.success("This model is promising!")
                                    
                                    # Add to config if promising
                                    if st.button(f"Add {model['name']} to configuration", key=f"add_{i}"):
                                        updated_config = model_discovery.add_to_config(model, model['provider'].lower())
                                        st.json(updated_config.get(model['provider'].lower(), {}))
                                else:
                                    st.warning("This model is not promising enough.")
            else:
                st.info(f"No {category} models found.")

# Tab 3: Router Model Discovery
with tabs[2]:
    st.header("Test Router Model Discovery Integration")
    st.markdown("""
    ### What is Router Model Discovery?
    Router Model Discovery extends basic discovery by finding models that are specifically suited for:
    - Particular model families (like LLaMA variants)
    - Specific domains (like coding, math, science)
    
    ### How to use this tab:
    1. In the left column: Click "Discover LLaMA Variants" to find promising LLaMA models
    2. In the right column: Select a domain and click "Discover Models" to find domain-specific models
    3. At the bottom: Click "View Discovery Log" to see a history of discoveries
    """)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Discover LLaMA Variants")
        if st.button("Discover LLaMA Variants", key="discover_llama"):
            with st.spinner("Discovering LLaMA variants..."):
                # Initialize router model discovery
                router_discovery = RouterModelDiscovery()
                
                # Discover LLaMA variants
                promising_models, updated_config = router_discovery.discover_llama_variants()
                
                # Display results
                st.markdown(f"**Found {len(promising_models)} promising LLaMA variants**")
                
                if promising_models:
                    st.subheader("First promising model:")
                    st.json(promising_models[0])
                
                if updated_config:
                    st.success("Configuration updated successfully")
                else:
                    st.info("No configuration updates were made")
    
    with col2:
        st.subheader("Discover Models for Domain")
        # Domain selection
        domain = st.selectbox(
            "Select a domain:",
            ["coding", "math", "science", "general_knowledge", "creative"],
            index=0
        )
        
        if st.button(f"Discover Models for {domain}", key="discover_domain"):
            with st.spinner(f"Discovering models for {domain}..."):
                # Initialize router model discovery
                router_discovery = RouterModelDiscovery()
                
                # Discover models for domain
                domain_models, domain_config = router_discovery.discover_models_for_domain(domain)
                
                # Display results
                st.markdown(f"**Found {len(domain_models)} promising models for {domain}**")
                
                if domain_models:
                    st.subheader("First domain-specific model:")
                    st.json(domain_models[0])
    
    # Discovery log
    st.subheader("Discovery Log")
    if st.button("View Discovery Log", key="view_log"):
        with st.spinner("Retrieving discovery log..."):
            # Initialize router model discovery
            router_discovery = RouterModelDiscovery()
            
            # Get discovery log
            discovery_log = router_discovery.get_discovery_log()
            
            # Display log
            st.markdown(f"**Discovery log entries: {len(discovery_log)}**")
            st.json(discovery_log)

# Tab 4: Mistral Model Discovery
with tabs[3]:
    st.header("Test Mistral Provider Model Discovery")
    st.markdown("""
    ### What is Mistral Provider Model Discovery?
    The Mistral provider has specialized capabilities to discover and evaluate LLaMA models.
    
    ### How to use this tab:
    1. Click the "Discover LLaMA Models via Mistral" button
    2. Review the discovered models that appear as cards
    3. Click "Evaluate & Add" on any model to assess its performance
    4. If the model is promising, it will be added to the Mistral configuration
    """)
    
    if st.button("Discover LLaMA Models via Mistral", key="mistral_discover"):
        with st.spinner("Discovering LLaMA models through Mistral provider..."):
            # Initialize Mistral provider
            mistral_config = MODEL_CONFIGS["mistral"]
            mistral = MistralProvider(mistral_config)
            
            # Discover LLaMA models
            discovered_models = mistral.discover_llama_models()
            
            # Display results
            st.markdown(f"**Discovered {len(discovered_models)} LLaMA models**")
            
            if discovered_models:
                # Create columns for model cards
                cols = st.columns(min(3, len(discovered_models)))
                
                for i, model in enumerate(discovered_models):
                    with cols[i % min(3, len(discovered_models))]:
                        st.markdown(f"**Model {i+1}: {model['name']}**")
                        st.markdown(f"**Provider:** {model['provider']}")
                        st.markdown(f"**Size:** {model['size']}")
                        st.markdown(f"**Strengths:** {', '.join(model['strengths'])}")
                        
                        # Add evaluate button for each model
                        if st.button(f"Evaluate & Add {model['name']}", key=f"mistral_eval_{i}"):
                            with st.spinner(f"Evaluating {model['name']}..."):
                                # Evaluate and add model
                                updated_config = mistral.evaluate_and_add_model(model)
                                
                                if updated_config:
                                    st.success("Model was promising and added to configuration")
                                    st.subheader("Updated Mistral configuration:")
                                    st.json(updated_config.get("mistral", {}))
                                else:
                                    st.warning("Model was not promising enough to add to configuration")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Dynamic LLM Router - RAG & Discovery Testing Interface</p>
    <p style="font-size: 0.8em; color: #888;">Use the tabs above to navigate between different features. Each tab has step-by-step instructions.</p>
</div>
""", unsafe_allow_html=True)
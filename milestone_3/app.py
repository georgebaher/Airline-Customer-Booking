"""
Streamlit UI for the Airline Graph-RAG Assistant.

Run: streamlit run app.py
"""

import streamlit as st
import json
import time

# Must be first Streamlit command
st.set_page_config(
    page_title="Airline Graph-RAG Assistant",
    page_icon="✈️",
    layout="wide"
)

from intent_classifier import default_classifier
from llm_generation import RAGAgent, AVAILABLE_MODELS

# --- STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .context-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<p class="main-header">Airline Graph-RAG Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about flights, routes, delays, and customer satisfaction</p>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    
    # Model selector
    model_choice = st.selectbox(
        "LLM Model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: f"{x.upper()} ({AVAILABLE_MODELS[x].split('/')[-1]})"
    )
    
    # Embedding settings
    st.subheader("Embedding Settings")
    embedding_model = st.selectbox("Embedding Model", ["minilm", "mpnet"])
    embedding_style = st.selectbox("Description Style", ["rich", "simple"])
    top_k = st.slider("Top-K Results", 1, 10, 5)
    

# --- MAIN QUERY INPUT ---
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input(
        "Your Question",
        placeholder="e.g., Find flights from ORD to LAX with minimal delays",
        label_visibility="collapsed"
    )
with col2:
    submit_btn = st.button("Search", type="primary", width="stretch")

# --- EXAMPLE QUERIES ---
with st.expander("Example Queries"):
    examples = [
        "Find flights from ORD to LAX with minimal delays",
        "Which flights have the best food quality?",
        "Compare Business vs Economy class",
        "What planes fly JFK to SFO?",
        "How do Gold members rate the service?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            user_query = ex
            submit_btn = True

# --- MAIN LOGIC ---
if submit_btn and user_query:
    
    with st.spinner("Processing your query..."):
        
        # Initialize components
        classifier = default_classifier()
        agent = RAGAgent()
        
        # Step 1: Classification & Retrieval
        start_time = time.time()
        intent_result, query_results = classifier.classify_and_run_hybrid(
            user_query,
            embedding_model=embedding_model,
            embedding_style=embedding_style,
            top_k=top_k
        )
        retrieval_time = time.time() - start_time
        
        # Step 2: LLM Generation
        result = agent.generate_answer(
            user_query=user_query,
            intent_result=intent_result,
            query_result=query_results,
            model_name=model_choice,
            return_metrics=True
        )
    
    # --- RESULTS DISPLAY ---
    
    # Answer Section
    st.header("Answer")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;
                color: #333;">
        {result['answer']}
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Response Time", f"{result['response_time_s']}s")
    with col2:
        st.metric("Total Tokens", result['total_tokens'])
    with col3:
        st.metric("Est. Cost", f"${result['cost_usd']:.6f}")
    with col4:
        st.metric("Model", model_choice.upper())
    
    st.divider()
    
    # --- KG CONTEXT SECTION ---
    st.header("Knowledge Graph Context")
    
    # Intent Classification
    with st.expander("Intent Classification", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Intent ID:** `{intent_result.intent_id}`")
        with col2:
            st.markdown(f"**Entities:** `{json.dumps(intent_result.entities)}`")
    
    # Cypher Query
    with st.expander("Cypher Query Executed"):
        if intent_result.cypher_template:
            st.code(intent_result.cypher_template, language="cypher")
        else:
            st.info("No Cypher template for this intent (general_qa)")
    
    # Baseline Results (Structural)
    with st.expander("Structural Search Results (Cypher)"):
        baseline = query_results.get("baseline")
        if baseline and baseline.get("rows"):
            st.dataframe(baseline["rows"], width="stretch")
            st.caption(f"Query time: {baseline.get('elapsed_ms', 0):.1f}ms")
        else:
            st.info("No structural matches found")
    
    # Embedding Results (Semantic)
    with st.expander("Semantic Search Results (Embeddings)", expanded=True):
        embedding_res = query_results.get("embedding", {})
        if embedding_res.get("rows"):
            # Format for display
            rows = embedding_res["rows"]
            display_data = []
            for row in rows:
                display_data.append({
                    "Score": f"{row.get('score', 0):.4f}",
                    "Class": row.get("passenger_class", ""),
                    "Fleet": row.get("fleet", ""),
                    "Route": f"{row.get('origin', '')} → {row.get('destination', '')}",
                    "Delay": f"{row.get('delay', 0)} min",
                    "Food": f"{row.get('food_score', 0)}/5",
                })
            st.dataframe(display_data, width="stretch")
            st.caption(f"Model: {embedding_model} | Style: {embedding_style} | Query time: {embedding_res.get('elapsed_ms', 0):.1f}ms")
        else:
            st.info("No semantic matches found")
    
    # Retrieval Stats
    with st.expander("Retrieval Statistics"):
        st.markdown(f"""
        - **Total Retrieval Time:** {retrieval_time:.2f}s
        - **Embedding Dimensions:** {query_results.get('query_embedding_dims', 'N/A')}
        - **Embedding Model:** {embedding_model}
        - **Description Style:** {embedding_style}
        """)
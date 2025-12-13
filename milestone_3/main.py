"""Main entry point for the Airline Graph-RAG Travel Assistant.

This module provides a simple CLI interface for testing the intent classification
and query execution pipeline.
"""

from __future__ import annotations

import json


from llm_generation import RAGAgent
from intent_classifier import default_classifier

def main():
    """Run a sample query through the full Graph-RAG pipeline."""
    # 1. Configuration
    QUERY = "Find flights from LAX to IAX with minimal delays"
    SELECTED_MODEL = "minilm" # minilm or mpnet
    SELECTED_STYLE = "rich" # rich or simple
    LLM_MODEL_NAME = None # Defaults to 'gemini-2.5-flash' from config
    
    print(f"User Query: '{QUERY}'")
    print("=" * 60)
    
    # 2. Classification & Retrieval
    print("Classifying and Retrieving Data...")
    classifier = default_classifier()
    intent_res, query_res = classifier.classify_and_run_hybrid(
        QUERY,
        embedding_model=SELECTED_MODEL,
        embedding_style=SELECTED_STYLE
    )
    
    print(f"   -> Intent: {intent_res.intent_id}")
    print(f"   -> Baseline Results: {len(query_res['baseline']['rows']) if query_res['baseline'] else 0}")
    print(f"   -> Sim Search Results: {len(query_res['embedding']['rows'])}")

    
    # 3. RAG Generation
    print("\nGenerating Answer with LLM...")
    agent = RAGAgent()
    answer = agent.generate_answer(
        user_query=QUERY,
        intent_result=intent_res,
        query_result=query_res,
        model_name=LLM_MODEL_NAME
    )
    
    print("-" * 60)
    print("FINAL ANSWER:\n")
    print(answer)
    print("=" * 60)

if __name__ == "__main__":
    main()

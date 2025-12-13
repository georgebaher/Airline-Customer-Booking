"""Main entry point for the Airline Graph-RAG Travel Assistant (Non-Hybrid).

This module provides a simple CLI interface for testing the intent classification
and query execution pipeline using ONLY the structural Graph (Cypher).
"""

from __future__ import annotations

import json


from llm_generation import RAGAgent
from intent_classifier import default_classifier

def main():
    """Run a sample query through the specific Graph-RAG pipeline (No Embeddings)."""
    # 1. Configuration
    QUERY = "What is the average delay from LAX to IAX"
    LLM_MODEL_NAME = None # Defaults to 'gemini-2.5-flash' from config
    
    print(f"User Query: '{QUERY}'")
    print("=" * 60)
    
    # 2. Classification & Retrieval
    print("Classifying and Retrieving Data (Structural Only)...")
    classifier = default_classifier()
    
    # Call the non-hybrid version
    intent_res, baseline_res = classifier.classify_and_run(QUERY)
    
    # Wrap results to match RAGAgent expectation
    query_res = {
        "baseline": baseline_res,
        "embedding": None
    }
    
    print(f"   -> Intent: {intent_res.intent_id}")
    baseline_count = len(baseline_res['rows']) if baseline_res and 'rows' in baseline_res else 0
    print(f"   -> Baseline Results: {baseline_count}")
    
    if baseline_res and 'rows' in baseline_res:
        print("\n--- RETRIEVED DATA ---")
        for row in baseline_res['rows']:
            print(json.dumps(row, indent=2, default=str))
        print("----------------------\n")
    # Removed Sim Search Results print as it is not applicable

    
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

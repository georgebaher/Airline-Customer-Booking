"""Main entry point for the Airline Graph-RAG Travel Assistant.

This module provides a simple CLI interface for testing the intent classification
and query execution pipeline.
"""

from __future__ import annotations

import json

from intent_classifier import default_classifier


def main():
    """Run a sample query through the classifier and execute it."""
    query = "Find flights from LAX to IAX with minimal delays"
    
    print(f"Query: {query}\n")
    print("-" * 50)
    
    classifier = default_classifier()
    result, query_result = classifier.classify_and_run(query)
    
    payload = result.__dict__.copy()
    if query_result is not None:
        payload["query_result"] = query_result
    
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

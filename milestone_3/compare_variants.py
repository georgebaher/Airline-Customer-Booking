"""
Compare search results across all 4 embedding variants (Model x Style).

Run this script to see how different models and description styles affect
the search results for a given query.
"""

import json
from intent_classifier import default_classifier

def main():
    # You can change this query to test different scenarios
    query = "Find flights with excellent food and legroom"
    
    print(f"Query: '{query}'\n")
    print("=" * 60)
    
    classifier = default_classifier()
    
    variants = [
        ("minilm", "rich"),
        ("minilm", "simple"),
        ("mpnet", "rich"),
        ("mpnet", "simple"),
    ]
    
    for model, style in variants:
        print(f"\nVariant: {model.upper()} | {style.capitalize()}")
        print("-" * 30)
        
        # We only care about the embedding search results here
        try:
            _, results = classifier.classify_and_run_hybrid(
                query, 
                embedding_model=model,
                embedding_style=style,
                top_k=3
            )
            
            embedding_hits = results.get("embedding", {}).get("rows", [])
            
            if not embedding_hits:
                print("   (No results found)")
                continue
                
            for i, hit in enumerate(embedding_hits, 1):
                # Print key details to help comparison
                score = hit.get("score", 0.0)
                desc = hit.get("fleet", "Unknown")
                pax = hit.get("passenger_class", "Unknown")
                food = hit.get("food_score", "N/A")
                
                print(f"   {i}. [{score:.4f}] {pax} on {desc} (Food: {food})")
                print(f"      Feedback ID: {hit.get('feedback_id')}")

        except Exception as e:
            print(f"   Error: {e}")
            
    print("\n" + "=" * 60)
    print("Compare the results above to decide which variant works best for your use case.")

if __name__ == "__main__":
    main()

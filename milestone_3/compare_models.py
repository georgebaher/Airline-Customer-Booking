"""
Compare LLM responses across different models via OpenRouter.

Tests the same query against all available models and displays:
- Response time
- Token usage (prompt/completion/total)
- Estimated cost
- The actual response

Run: python compare_models.py
"""

from intent_classifier import default_classifier
from llm_generation import RAGAgent, AVAILABLE_MODELS

def main():
    # Test query - change this to test different scenarios
    query = "Which flights had the best food quality and shortest delays?"
    
    print(f"Query: '{query}'")
    print("=" * 70)
    
    # Step 1: Get classification and retrieval results (same for all models)
    classifier = default_classifier()
    intent_result, query_results = classifier.classify_and_run_hybrid(
        query,
        embedding_model="minilm",
        embedding_style="rich",
        top_k=3
    )
    
    print(f"\nIntent: {intent_result.intent_id}")
    print("=" * 70)
    
    # Step 2: Test each model with metrics
    agent = RAGAgent()
    results = []
    
    for alias, full_name in AVAILABLE_MODELS.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {alias.upper()} ({full_name})")
        print("=" * 70)
        
        try:
            result = agent.generate_answer(
                user_query=query,
                intent_result=intent_result,
                query_result=query_results,
                model_name=alias,
                return_metrics=True  # Get metrics!
            )
            
            results.append(result)
            
            # Display metrics
            print(f"\nMETRICS:")
            print(f"   Response Time: {result['response_time_s']}s")
            print(f"   Prompt Tokens: {result['prompt_tokens']}")
            print(f"   Completion Tokens: {result['completion_tokens']}")
            print(f"   Total Tokens: {result['total_tokens']}")
            print(f"   Est. Cost: ${result['cost_usd']:.6f}")
            
            print(f"\nRESPONSE:")
            print("-" * 40)
            print(result['answer'])
            print("-" * 40)
            
        except Exception as e:
            print(f"\nERROR: {e}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Model':<30} {'Time(s)':<10} {'Tokens':<10} {'Cost($)':<12}")
    print("-" * 70)
    for r in results:
        model_short = r['model'].split('/')[-1][:28]
        print(f"{model_short:<30} {r['response_time_s']:<10} {r['total_tokens']:<10} {r['cost_usd']:<12.6f}")
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main()

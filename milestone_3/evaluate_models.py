"""Model Evaluation Script.

Runs a set of test questions against multiple LLM models and saves the results
for qualitative comparison (Part 3.c/d).
"""

import csv
from pathlib import Path
from time import perf_counter
from typing import List

from intent_classifier import default_classifier
from llm_generation import RAGAgent

# Define the models to test
# Define the models to test
MODELS_TO_TEST = [
    "gemini-2.5-flash",
]

TEST_QUESTIONS = [
    "Find flights from LAX to IAX with excellent food",
    "What is the average delay for Business class flights?",
    "Show me direct flights to Chicago with good legroom",
    "Which fleet type has the best food satisfaction?",
]

OUTPUT_FILE = Path(__file__).with_name("model_evaluation_results.csv")

def evaluate_models():
    print("Starting Model Evaluation...")
    print(f"   Models: {MODELS_TO_TEST}")
    print(f"   Questions: {len(TEST_QUESTIONS)}\n")
    
    classifier = default_classifier()
    agent = RAGAgent()
    
    results = []
    
    for q_idx, query in enumerate(TEST_QUESTIONS, 1):
        print(f"Question {q_idx}: {query}")
        
        # 1. Retrieve Context (Shared across all models for fairness)
        intent_res, query_res = classifier.classify_and_run_hybrid(query)
        
        row_base = {
            "Question": query,
            "Intent": intent_res.intent_id,
        }
        
        for model in MODELS_TO_TEST:
            print(f"   Asking {model}...", end="", flush=True)
            start = perf_counter()
            try:
                answer = agent.generate_answer(
                    user_query=query,
                    intent_result=intent_res,
                    query_result=query_res,
                    model_name=model
                )
                elapsed = perf_counter() - start
                
                # Store full answer and timing
                results.append({
                    **row_base,
                    "Model": model,
                    "Answer": answer,
                    "Latency (s)": round(elapsed, 2)
                })
                print(f" Done ({elapsed:.2f}s)")
                
            except Exception as e:
                print(f" Error: {e}")
        
        print("")

    # Save to CSV
    keys = ["Question", "Intent", "Model", "Latency (s)", "Answer"]
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Evaluation complete! Results saved to {OUTPUT_FILE}")
    print("Compare the 'Answer' column to perform your qualitative analysis.")

if __name__ == "__main__":
    evaluate_models()

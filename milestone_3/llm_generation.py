"""RAG Answer Generation Module.

Implements Part 3 of the project:
- Combines KG results (Context)
- Defines Persona (Airline Assistant)
- Generates Answer using LLM via OpenRouter
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import LLMConfig, load_llm_config


# --- AVAILABLE MODELS (via OpenRouter) ---
AVAILABLE_MODELS = {
    "gemini": "google/gemini-2.5-flash",
    "haiku": "anthropic/claude-haiku-4.5", 
    "openai": "openai/gpt-5-mini",
}

DEFAULT_MODEL = "gemini"


# --- PROMPT COMPONENTS ---

PERSONA = """You are an expert Airline Data Analyst and Customer Assistant. 
You work for the airline company and have access to a Knowledge Graph containing detailed journey data.
Your goal is to answer questions accurately based ONLY on the provided context."""

TASK = """
Instructions:
1.  Read the User Query and the Retrieved Context carefully.
2.  Synthesize the information from the context to answer the query.
3.  If the context does not contain the answer, admit that you don't know - do NOT hallucinate.
4.  Cite specific flight numbers, routes, or metrics (like food score, delay) from the context to support your answer.
5.  Keep your tone professional, helpful, and concise.
"""

# --- CONTEXT FORMATTING ---

def format_context(intent_result: Any, query_result: Dict[str, Any]) -> str:
    """
    Format the hybrid retrieval results into a readable text block for the LLM.
    
    Args:
        intent_result: The IntentResult object from the classifier.
        query_result: The dictionary of results from `classify_and_run_hybrid`.
                      Expected keys: 'baseline', 'embedding'
    """
    lines = []
    
    # 1. Intent Info
    lines.append(f"Intent Detected: {intent_result.intent_id} (Confidence: {intent_result.confidence:.2f})")
    lines.append(f"Rationale: {intent_result.rationale}")
    
    # 2. Baseline Results (Cypher)
    lines.append("\n=== STRUCTURAL SEARCH RESULTS (Exact Matches) ===")
    baseline = query_result.get("baseline")
    if baseline and baseline.get("rows"):
        for i, row in enumerate(baseline["rows"], 1):
            lines.append(f"{i}. {json.dumps(row, default=str)}")
    else:
        lines.append("(No exact structural matches found)")

    # 3. Embedding Results (Vector Search)
    lines.append("\n=== SEMANTIC SEARCH RESULTS (Similar Journeys) ===")
    embedding_res = query_result.get("embedding")
    if embedding_res and embedding_res.get("rows"):
        for i, row in enumerate(embedding_res["rows"], 1):
            # Format nicely
            score = row.get("score", 0.0)
            summary = (
                f"Journey: {row.get('passenger_class')} class on {row.get('fleet')} "
                f"from {row.get('origin')} to {row.get('destination')}. "
                f"Delay: {row.get('delay')}min, Food: {row.get('food_score')}/5."
            )
            lines.append(f"{i}. [Similarity: {score:.2f}] {summary}")
            # detailed dump if needed
            # lines.append(f"   Details: {json.dumps(row, default=str)}")
    else:
        lines.append("(No semantic matches found)")
        
    return "\n".join(lines)


# --- GENERATION ---

class RAGAgent:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or load_llm_config()
        
    def generate_answer(
        self, 
        user_query: str, 
        intent_result: Any,
        query_result: Dict[str, Any],
        model_name: Optional[str] = None,
        return_metrics: bool = False
    ) -> str | Dict[str, Any]:
        """
        Generate a final answer using the LLM.
        
        Args:
            user_query: The original question.
            intent_result: Classification result.
            query_result: Retrieval results.
            model_name: Optional model alias ('gemini', 'sonnet', 'openai') or full name.
            return_metrics: If True, returns dict with answer + metrics.
        
        Returns:
            If return_metrics=False: answer string
            If return_metrics=True: dict with keys: answer, model, response_time_s, 
                                    prompt_tokens, completion_tokens, total_tokens, cost_usd
        """
        import time
        
        # 1. Prepare Context
        context_text = format_context(intent_result, query_result)
        
        # 2. Construct Prompt
        system_content = f"{PERSONA}\n\n{TASK}\n\n=== CONTEXT ===\n{context_text}"
        
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"User Query: {user_query}")
        ]
        
        print("\n--- DEBUG: PROMPT SENT TO LLM ---")
        print(f"SYSTEM:\n{system_content}")
        print(f"USER:\nUser Query: {user_query}")
        print("---------------------------------\n")
        
        # 3. Initialize LLM via OpenRouter
        if model_name in AVAILABLE_MODELS:
            target_model = AVAILABLE_MODELS[model_name]
        elif model_name:
            target_model = model_name
        else:
            target_model = AVAILABLE_MODELS.get(DEFAULT_MODEL, self.config.model)
        
        llm = ChatOpenAI(
            model=target_model,
            temperature=0.2,
            api_key=self.config.require_api_key(),
            base_url=self.config.base_url or "https://openrouter.ai/api/v1",
        )
        
        # 4. Invoke with timing
        start_time = time.time()
        response = llm.invoke(messages)
        elapsed = time.time() - start_time
        
        if not return_metrics:
            return response.content
        
        # 5. Extract usage metrics from response_metadata
        # OpenRouter may return usage in different locations
        metadata = response.response_metadata or {}
        usage = metadata.get("usage") or metadata.get("token_usage") or {}
        
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
        
        # 6. Estimate cost (per 1M tokens, OpenRouter pricing)
        cost = self._estimate_cost(target_model, prompt_tokens, completion_tokens)
        
        return {
            "answer": response.content,
            "model": target_model,
            "response_time_s": round(elapsed, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(cost, 6),
        }
    
    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on OpenRouter pricing (per 1M tokens)."""
        # Approximate pricing (check OpenRouter for current rates)
        PRICING = {
            "google/gemini-2.5-flash": {"input": 0.30, "output": 2.50},
            "anthropic/claude-haiku-4.5": {"input": 1.00, "output": 5.00},
            "openai/gpt-5-mini": {"input": 0.25, "output": 2.00},
        }
        rates = PRICING.get(model)
        input_cost = (prompt_tokens / 1_000_000) * rates["input"]
        output_cost = (completion_tokens / 1_000_000) * rates["output"]
        return input_cost + output_cost



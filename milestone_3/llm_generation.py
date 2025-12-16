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

# --- SIMILARITY THRESHOLD CONFIGURATION ---
# Used to filter semantic search results by confidence level
SIMILARITY_THRESHOLD_HIGH = 0.7    # High confidence: include as strong evidence
SIMILARITY_THRESHOLD_MEDIUM = 0.5  # Medium confidence: include with caveat
# Below MEDIUM threshold: excluded from context

TASK_BASELINE = """
Instructions:
1.  The context below contains STRUCTURAL SEARCH RESULTS - exact matches from the Knowledge Graph retrieved via Cypher queries.
2.  Answer the user's query using ONLY the data provided in the structural search results.
3.  If the structural search returned no results or insufficient data, clearly state that no matching data was found for the specified criteria.
4.  When citing data, reference specific values (e.g., flight numbers, fleet types, delay minutes, food scores) from the results.
5.  Keep your response professional, factual, and concise.
"""

TASK_HYBRID = """
Instructions:
1.  The context below contains results from TWO retrieval sources:
    - **STRUCTURAL SEARCH (PRIMARY)**: Exact matches from Cypher queries - these are precise and should be your main source.
    - **SEMANTIC SEARCH (SUPPLEMENTARY)**: Similar journeys found via vector embeddings - these provide additional context.

2.  PRIORITY RULES:
    - Always prioritize STRUCTURAL results when answering specific questions about routes, delays, or metrics.
    - Use SEMANTIC results to enrich your answer when structural data is limited or when the query asks for general patterns/recommendations.
    - Semantic results with similarity scores ≥0.70 are highly relevant; scores 0.50-0.69 are moderately relevant.

3.  HANDLING CONFLICTS:
    - If structural and semantic results conflict, trust the structural results.
    - If structural search is empty but semantic search has results, you may answer based on semantic results without mentioning that you did not find exact matches.

4.  RESPONSE GUIDELINES:
    - If no relevant data is found in either source, clearly state this.
    - Keep your response professional, factual, and concise.
    - Return the results without saying they were from structural or semantic search.
"""

# --- CONTEXT FORMATTING ---

def format_context_baseline(intent_result: Any, query_result: Dict[str, Any]) -> str:
    """
    Format the baseline retrieval results (structural only) into a readable text block for the LLM.
    
    Args:
        intent_result: The IntentResult object from the classifier.
        query_result: The dictionary of results from baseline search.
                      Expected keys: 'baseline' or just the result directly
    """
    lines = []
    
    # 1. Intent Info
    lines.append(f"Intent: {intent_result.intent_id}")
    if intent_result.entities:
        lines.append(f"Extracted Entities: {json.dumps(intent_result.entities)}")
    
    # 2. Baseline Results (Cypher)
    baseline = query_result.get("baseline") or query_result
    rows = baseline.get("rows", []) if baseline else []
    
    lines.append(f"\n=== STRUCTURAL SEARCH RESULTS ({len(rows)} matches) ===")
    if rows:
        for i, row in enumerate(rows, 1):
            lines.append(f"{i}. {json.dumps(row, default=str)}")
    else:
        lines.append("(No exact structural matches found for the specified criteria)")
        
    return "\n".join(lines)


def format_context_hybrid(intent_result: Any, query_result: Dict[str, Any]) -> str:
    """
    Format the hybrid retrieval results (structural + semantic) into a readable text block for the LLM.
    
    Applies similarity threshold filtering to semantic results.
    
    Args:
        intent_result: The IntentResult object from the classifier.
        query_result: The dictionary of results from `classify_and_run_hybrid`.
                      Expected keys: 'baseline', 'embedding'
    """
    lines = []
    
    # 1. Intent Info
    lines.append(f"Intent: {intent_result.intent_id}")
    if intent_result.entities:
        lines.append(f"Extracted Entities: {json.dumps(intent_result.entities)}")
    
    # 2. Baseline Results (Cypher) - PRIMARY
    baseline = query_result.get("baseline")
    baseline_rows = baseline.get("rows", []) if baseline else []
    
    lines.append(f"\n=== STRUCTURAL SEARCH RESULTS [PRIMARY] ({len(baseline_rows)} matches) ===")
    if baseline_rows:
        for i, row in enumerate(baseline_rows, 1):
            lines.append(f"{i}. {json.dumps(row, default=str)}")
    else:
        lines.append("(No exact structural matches found)")

    # 3. Embedding Results (Vector Search) - SUPPLEMENTARY
    # Filter by similarity threshold
    embedding_res = query_result.get("embedding")
    all_rows = embedding_res.get("rows", []) if embedding_res else []
    
    # Filter and categorize by confidence
    high_conf = []
    medium_conf = []
    
    for row in all_rows:
        score = row.get("score", 0.0)
        if score >= SIMILARITY_THRESHOLD_HIGH:
            high_conf.append(row)
        elif score >= SIMILARITY_THRESHOLD_MEDIUM:
            medium_conf.append(row)
        # Below MEDIUM threshold: excluded
    
    filtered_count = len(high_conf) + len(medium_conf)
    lines.append(f"\n=== SEMANTIC SEARCH RESULTS [SUPPLEMENTARY] ({filtered_count} relevant matches) ===")
    
    if high_conf or medium_conf:
        idx = 1
        for row in high_conf:
            score = row.get("score", 0.0)
            summary = _format_journey_summary(row)
            lines.append(f"{idx}. [HIGH RELEVANCE {score:.2f}] {summary}")
            idx += 1
        for row in medium_conf:
            score = row.get("score", 0.0)
            summary = _format_journey_summary(row)
            lines.append(f"{idx}. [MEDIUM RELEVANCE {score:.2f}] {summary}")
            idx += 1
    else:
        lines.append("(No semantically similar journeys found above threshold)")
        
    return "\n".join(lines)


def _format_journey_summary(row: Dict[str, Any]) -> str:
    """Format a journey row into a readable summary string."""
    parts = []
    
    if row.get("passenger_class"):
        parts.append(f"{row['passenger_class']} class")
    if row.get("fleet"):
        parts.append(f"on {row['fleet']}")
    if row.get("origin") and row.get("destination"):
        parts.append(f"from {row['origin']} to {row['destination']}")
    if row.get("delay") is not None:
        parts.append(f"delay: {row['delay']}min")
    if row.get("food_score") is not None:
        parts.append(f"food: {row['food_score']}/5")
    
    return ", ".join(parts) if parts else json.dumps(row, default=str)


# Backward compatibility alias - defaults to hybrid
def format_context(intent_result: Any, query_result: Dict[str, Any]) -> str:
    """
    Format retrieval results into a readable text block for the LLM.
    This function defaults to hybrid formatting for backward compatibility.
    
    Args:
        intent_result: The IntentResult object from the classifier.
        query_result: The dictionary of results.
    """
    return format_context_hybrid(intent_result, query_result)


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
        return_metrics: bool = False,
        approach: str = "hybrid"
    ) -> str | Dict[str, Any]:
        """
        Generate a final answer using the LLM.
        
        Args:
            user_query: The original question.
            intent_result: Classification result.
            query_result: Retrieval results.
            model_name: Optional model alias ('gemini', 'sonnet', 'openai') or full name.
            return_metrics: If True, returns dict with answer + metrics.
            approach: Either 'baseline' or 'hybrid' - determines which prompt and context formatting to use.
        
        Returns:
            If return_metrics=False: answer string
            If return_metrics=True: dict with keys: answer, model, response_time_s, 
                                    prompt_tokens, completion_tokens, total_tokens, cost_usd
        """
        import time
        
        # 1. Prepare Context based on approach
        if approach == "baseline":
            context_text = format_context_baseline(intent_result, query_result)
            task_prompt = TASK_BASELINE
        else:  # hybrid (default)
            context_text = format_context_hybrid(intent_result, query_result)
            task_prompt = TASK_HYBRID
        
        # 2. Construct Prompt
        system_content = f"{PERSONA}\n\n{task_prompt}\n\n=== CONTEXT ===\n{context_text}"
        
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"User Query: {user_query}")
        ]
        
        # print("\n--- DEBUG: PROMPT SENT TO LLM ---")
        # print(f"SYSTEM:\n{system_content}")
        # print(f"USER:\nUser Query: {user_query}")
        # print("---------------------------------\n")
        
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



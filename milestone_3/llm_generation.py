"""RAG Answer Generation Module.

Implements Part 3 of the project:
- Combines KG results (Context)
- Defines Persona (Airline Assistant)
- Generates Answer using LLM
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import LLMConfig, load_llm_config


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
        model_name: Optional[str] = None
    ) -> str:
        """
        Generate a final answer using the LLM.
        
        Args:
            user_query: The original question.
            intent_result: Classification result.
            query_result: Retrieval results.
            model_name: Optional override for the model (e.g., 'gpt-3.5-turbo'). 
                        If None, uses config default.
        """
        
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
        
        # 3. Initialize LLM (allows switching models per request)
        # Use the config's credentials but optionally swap the model name
        target_model = model_name or self.config.model
        
        # Check if we should use Google Gemini
        if "gemini" in target_model.lower():
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError:
                raise ImportError("langchain-google-genai is not installed. Please run `pip install langchain-google-genai`.")
                
            llm = ChatGoogleGenerativeAI(
                model=target_model,
                temperature=0.2,
                google_api_key=self.config.require_google_key()
            )
        else:
            # Default to OpenAI compatible (OpenAI, OpenRouter, etc)
            llm = ChatOpenAI(
                model=target_model,
                temperature=0.2,
                api_key=self.config.require_api_key(),
                base_url=self.config.base_url,
            )
        
        # 4. Invoke
        response = llm.invoke(messages)
        return response.content


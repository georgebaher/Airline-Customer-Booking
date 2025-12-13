"""LLM-based intent classification for the Airline hybrid travel assistant.

Outputs a single intent id, confidence, rationale, and lightweight entity hints
to feed downstream Cypher/embedding retrieval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import LLMConfig, load_llm_config
from intents import AIRLINE_INTENTS, INTENT_IDS, INTENT_ID_SET, INTENT_TEXT
from cypher_templates import get_cypher_template
from run_queries import run_query


# System prompt for LLM intent classification
SYSTEM_PROMPT = f"""You classify user queries for an Airline Graph-RAG assistant (Neo4j-backed).
Pick exactly one intent from the list and return strict JSON only.

Intents:
{INTENT_TEXT}

Output JSON with keys:
- intent_id: one of [{INTENT_IDS}]
- confidence: float 0-1 (≤0.35 if unsure)
- rationale: <=40 words describing the decision
- entities: dict with any of these keys if present:
    origin_station_code, destination_station_code, arrival_delay_minutes,
    passenger_class, number_of_legs, loyalty_program_level,
    generation, fleet_type_description, actual_flown_miles, food_satisfaction_score

Rules:
- If unclear or unsupported, set intent_id="general_qa" and low confidence.
- Do not invent entities; include only those implied by the user query.
- Respond with JSON ONLY, no prose, no markdown.
"""


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from the model, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        # Remove possible json tag like ```json
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def _build_query_params(
    intent_id: str, entities: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Translate classified entities into Cypher parameters expected by templates.

    Translate classified entities into Cypher parameters expected by templates.
    """

    def first_present(keys):
        for key in keys:
            if entities.get(key) is not None:
                return entities[key]
        return None

    param_map = {
        "origin": ("origin_station_code", "origin"),
        "dest": ("destination_station_code", "dest"),
        "class": ("passenger_class", "class"),
        "level": ("loyalty_program_level", "level"),
        "fleet": ("fleet_type_description", "fleet"),
        "min_miles": ("actual_flown_miles", "min_miles"),
        "generation": ("generation",),
    }

    params: Dict[str, Any] = {}
    for cypher_key, candidate_keys in param_map.items():
        value = first_present(candidate_keys)
        params[cypher_key] = value


    return params


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent_id: str
    confidence: float
    rationale: str
    entities: Dict[str, Any]
    cypher_template: Optional[str]
    raw_response: str


class LLMIntentClassifier:
    """LLM-first classifier over predefined airline intents."""

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        config: Optional[LLMConfig] = None,
    ):
        self.config = config or load_llm_config()
        self.llm = llm or ChatOpenAI(
            model=model or self.config.model,
            temperature=temperature,
            api_key=self.config.require_api_key(),
            base_url=self.config.base_url,
        )

    def classify(self, user_query: str) -> IntentResult:
        """Classify user query into an intent with entities."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"User query: {user_query}"),
        ]
        response = self.llm.invoke(messages)
        payload = _parse_json_response(response.content)

        intent_id = payload.get("intent_id", "general_qa")
        confidence = payload.get("confidence", 0.0)
        rationale = payload.get("rationale", "").strip()
        entities = payload.get("entities", {}) or {}
        template = get_cypher_template(intent_id)

        # Clamp values defensively
        try:
            confidence = max(0.0, min(float(confidence), 1.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if not intent_id or intent_id not in INTENT_ID_SET:
            intent_id = "general_qa"
            template = None

        return IntentResult(
            intent_id=intent_id,
            confidence=confidence,
            rationale=rationale,
            entities=entities,
            cypher_template=template,
            raw_response=response.content,
        )

    def run_intent_query(
        self, intent_result: IntentResult
    ) -> Optional[Dict[str, Any]]:
        """
        Execute the Cypher template for a classified intent, if available.

        Execute the Cypher template for a classified intent, if available.
        Returns dict with rows, elapsed_ms, and the final params used; or None
        if no template is associated with the intent.
        """
        if not intent_result.cypher_template:
            return None

        params = _build_query_params(intent_result.intent_id, intent_result.entities)
        rows, elapsed_ms = run_query(intent_result.cypher_template, params)
        return {"rows": rows, "elapsed_ms": elapsed_ms, "params": params}

    def classify_and_run(
        self, user_query: str
    ) -> Tuple[IntentResult, Optional[Dict[str, Any]]]:
        """
        Convenience: classify a query and immediately execute its Cypher template.
        Returns (IntentResult, query_result_dict|None).
        """
        intent_result = self.classify(user_query)
        query_result = self.run_intent_query(intent_result)
        return intent_result, query_result

    def classify_and_run_hybrid(
        self,
        user_query: str,
        embedding_model: str = "minilm",
        embedding_style: str = "rich",
        top_k: int = 5,
    ) -> Tuple[IntentResult, Dict[str, Any]]:
        """
        Hybrid retrieval: combine Cypher baseline with embedding similarity search.
        
        This implements the project requirement to combine both approaches:
        - Baseline: Structured Cypher queries from intent classification
        - Embeddings: Semantic similarity search using feature vector embeddings
        
        Returns (IntentResult, combined_results) where combined_results has:
        - 'baseline': Cypher query results (or None)
        - 'embedding': Similarity search results
        - 'query_embedding': The embedding vector used for search
        """
        # Import here to avoid circular dependency
        from embeddings import embed_query
        from run_queries import similarity_search
        
        # Classify intent and run baseline Cypher query
        intent_result = self.classify(user_query)
        baseline_result = self.run_intent_query(intent_result)
        
        # Run embedding-based similarity search
        query_embedding = embed_query(user_query, model_key=embedding_model)
        
        # Construct property name: embedding_{model}_{style}
        # e.g. embedding_minilm_rich
        # Fallback for backward compat if needed, but we enforce new schema now
        property_name = f"embedding_{embedding_model}_{embedding_style}"
        
        embedding_rows, embedding_ms = similarity_search(
            query_embedding, 
            top_k=top_k, 
            property_name=property_name
        )
        
        combined_results = {
            "baseline": baseline_result,
            "embedding": {
                "rows": embedding_rows,
                "elapsed_ms": embedding_ms,
                "model": embedding_model,
            },
            "query_embedding_dims": len(query_embedding),
        }
        
        return intent_result, combined_results


def default_classifier() -> LLMIntentClassifier:
    """Convenience factory with env-based configuration."""
    return LLMIntentClassifier()




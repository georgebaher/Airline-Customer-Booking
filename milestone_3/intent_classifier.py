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
    intent_id: str, entities: Dict[str, Any], extra_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Translate classified entities into Cypher parameters expected by templates.

    extra_params lets callers override or supply missing values explicitly.
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
        if value is not None:
            params[cypher_key] = value

    if extra_params:
        params.update({k: v for k, v in extra_params.items() if v is not None})

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
        self, intent_result: IntentResult, extra_params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute the Cypher template for a classified intent, if available.

        extra_params can supply or override parameters missing from entities.
        Returns dict with rows, elapsed_ms, and the final params used; or None
        if no template is associated with the intent.
        """
        if not intent_result.cypher_template:
            return None

        params = _build_query_params(intent_result.intent_id, intent_result.entities, extra_params)
        rows, elapsed_ms = run_query(intent_result.cypher_template, params)
        return {"rows": rows, "elapsed_ms": elapsed_ms, "params": params}

    def classify_and_run(
        self, user_query: str, extra_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[IntentResult, Optional[Dict[str, Any]]]:
        """
        Convenience: classify a query and immediately execute its Cypher template.
        Returns (IntentResult, query_result_dict|None).
        """
        intent_result = self.classify(user_query)
        query_result = self.run_intent_query(intent_result, extra_params)
        return intent_result, query_result


def default_classifier() -> LLMIntentClassifier:
    """Convenience factory with env-based configuration."""
    return LLMIntentClassifier()


__all__ = [
    "IntentResult",
    "LLMIntentClassifier",
    "default_classifier",
]

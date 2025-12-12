"""LLM-based intent classification for the Airline hybrid travel assistant.

Outputs a single intent id, confidence, rationale, and lightweight entity hints
to feed downstream Cypher/embedding retrieval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


from config import LLMConfig, load_llm_config
from run_queries import run_query

AIRLINE_INTENTS: List[Dict[str, Any]] = [
    {
        "id": "route_search",
        "label": "Route Search",
        "description": "Find flights between origin and destination.",
        "examples": ["Flights from JFK to LHR", "Show me routes between Tokyo and Paris"],
    },
    {
        "id": "check_avg_delay",
        "label": "Check Average Delay",
        "description": "Check the average arrival delay minutes for a specific route or airline.",
        "examples": ["How late are flights from LAX to NYC usually?", "Average delay for Chicago to Miami"],
    },
    {
        "id": "food_quality_inquiry",
        "label": "Food Quality Inquiry",
        "description": "Find food satisfaction scores, optionally filtered by passenger class.",
        "examples": ["How is the food in Business Class?", "Food rating for Economy flights to Dubai"],
    },
    {
        "id": "fleet_info",
        "label": "Fleet Information",
        "description": "Identify the fleet type or generation used on a specific route.",
        "examples": ["What kind of plane flies to Singapore?", "Is the aircraft generation new on this route?"],
    },
    {
        "id": "loyalty_demographics",
        "label": "Loyalty Program Stats",
        "description": "Analyze metrics based on loyalty program levels.",
        "examples": ["What class do Gold members usually fly?", "Satisfaction score for Platinum members"],
    },
    {
        "id": "direct_flight_search",
        "label": "Direct Flight Search",
        "description": "Search for flights with minimum legs (direct or non-stop).",
        "examples": ["Are there direct flights to Sydney?", "Show me flights with only 1 leg"],
    },
    {
        "id": "distance_inquiry",
        "label": "Distance Inquiry",
        "description": "Query the actual flown miles for a trip.",
        "examples": ["How many miles is the flight to Seattle?", "Distance from LHR to JFK"],
    },
    {
        "id": "reliability_check",
        "label": "Reliability Check",
        "description": "Find routes or fleets with minimal delay minutes.",
        "examples": ["Which flights to London have the least delay?", "Most reliable fleet type"],
    },
    {
        "id": "class_comparison",
        "label": "Class Comparison",
        "description": "Compare metrics (food, delay) between passenger classes.",
        "examples": ["Compare food score between Economy and First Class", "Difference in miles earned for Business vs Economy"],
    },
    {
        "id": "aircraft_performance",
        "label": "Aircraft Performance",
        "description": "Correlate fleet type description with delay or satisfaction.",
        "examples": ["Do Boeing 777s have good food ratings?", "Delays on the Airbus fleet"],
    },
    {
        "id": "long_haul_finder",
        "label": "Long Haul Finder",
        "description": "Find flights based on high actual flown miles.",
        "examples": ["Show me long haul flights over 5000 miles", "Longest routes available"],
    },
    {
        "id": "multi_leg_search",
        "label": "Multi-Leg Search",
        "description": "Find flights that specifically involve layovers (multiple legs).",
        "examples": ["Cheaper flights with stops to Thailand", "Routes with more than 1 leg"],
    },
    {
        "id": "generation_inquiry",
        "label": "Aircraft Generation Inquiry",
        "description": "Filter flights by aircraft generation.",
        "examples": ["I want to fly on the newest generation aircraft", "Old generation planes to avoid"],
    },
    {
        "id": "satisfaction_by_route",
        "label": "Satisfaction by Route",
        "description": "Get the overall satisfaction (food/service) for a specific O&D pair.",
        "examples": ["Are customers happy with the Berlin to Rome flight?", "Satisfaction rating for NYC to LAX"],
    },
    {
        "id": "comprehensive_flight_detail",
        "label": "Comprehensive Flight Detail",
        "description": "Get all available metrics for a specific flight query.",
        "examples": ["Tell me everything about the flight from BOS to DCA", "Stats for Business Class to Paris"],
    },
]

CYPHER_TEMPLATES: Dict[str, str] = {
    "route_search": """
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        RETURN f.flight_number, f.fleet_type_description
        LIMIT 5;
    """,
    "check_avg_delay": """
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        MATCH (j:Journey)-[:ON]->(f)
        RETURN avg(j.arrival_delay_minutes) as average_delay;
    """,
    "food_quality_inquiry": """
        MATCH (j:Journey)
        WHERE ($class IS NULL OR j.passenger_class = $class)
        RETURN j.passenger_class, avg(j.food_satisfaction_score) as avg_food_score
        ORDER BY avg_food_score DESC;
    """,
    "fleet_info": """
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        RETURN DISTINCT f.fleet_type_description
        ORDER BY f.fleet_type_description;
    """,
    "loyalty_demographics": """
        MATCH (p:Passenger {loyalty_program_level: $level})-[:TOOK]->(j:Journey)
        RETURN j.passenger_class, avg(j.food_satisfaction_score) as avg_sat
        ORDER BY avg_sat DESC;
    """,
    "direct_flight_search": """
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        MATCH (j:Journey)-[:ON]->(f)
        WHERE j.number_of_legs = 1
        RETURN o.station_code AS origin_station_code, d.station_code AS destination_station_code, j.actual_flown_miles;
    """,
    "distance_inquiry": """
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        MATCH (j:Journey)-[:ON]->(f)
        RETURN avg(j.actual_flown_miles) as miles LIMIT 1;
    """,
    "reliability_check": """
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        MATCH (j:Journey)-[:ON]->(f)
        RETURN f.fleet_type_description, avg(j.arrival_delay_minutes) as avg_delay
        ORDER BY avg_delay ASC LIMIT 1;
    """,
    "class_comparison": """
        MATCH (j:Journey)
        RETURN j.passenger_class, avg(j.food_satisfaction_score) as food, avg(j.arrival_delay_minutes) as delay
        ORDER BY j.passenger_class;
    """,
    "aircraft_performance": """
        MATCH (f:Flight)
        WHERE f.fleet_type_description CONTAINS $fleet
        MATCH (j:Journey)-[:ON]->(f)
        RETURN f.fleet_type_description, avg(j.arrival_delay_minutes) as delay, avg(j.food_satisfaction_score) as food;
    """,
    "long_haul_finder": """
        MATCH (o:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport)
        MATCH (j:Journey)-[:ON]->(f)
        WHERE j.actual_flown_miles > $min_miles
        RETURN o.station_code AS origin_station_code, d.station_code AS destination_station_code, j.actual_flown_miles
        ORDER BY j.actual_flown_miles DESC LIMIT 10;
    """,
    "multi_leg_search": """
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        MATCH (j:Journey)-[:ON]->(f)
        WHERE j.number_of_legs > 1
        RETURN j.number_of_legs, avg(j.arrival_delay_minutes) as avg_delay;
    """,
    "generation_inquiry": """
        MATCH (p:Passenger {generation: $generation})-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
        RETURN f.fleet_type_description, count(*) as flights_operated
        ORDER BY flights_operated DESC;
    """,
    "satisfaction_by_route": """
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        MATCH (j:Journey)-[:ON]->(f)
        RETURN avg(j.food_satisfaction_score) as global_satisfaction_score;
    """,
    "comprehensive_flight_detail": """
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        MATCH (j:Journey)-[:ON]->(f)
        RETURN j.passenger_class, f.fleet_type_description, j.arrival_delay_minutes, j.food_satisfaction_score
        LIMIT 5;
    """,
}

INTENT_IDS = ", ".join(intent["id"] for intent in AIRLINE_INTENTS)

INTENT_TEXT = "\n".join(
    f"- {intent['id']}: {intent['description']}"
    for intent in AIRLINE_INTENTS
)

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


def get_cypher_template(intent_id: str) -> Optional[str]:
    """Return the Cypher template for a supported intent id."""
    return CYPHER_TEMPLATES.get(intent_id)


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

        if not intent_id or intent_id not in {i["id"] for i in AIRLINE_INTENTS}:
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


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys

    query = "Find flights from LAX to IAX with minimal delays"
    classifier = default_classifier()
    result, query_result = classifier.classify_and_run(query)
    payload = result.__dict__.copy()
    if query_result is not None:
        payload["query_result"] = query_result
    print(json.dumps(payload, indent=2, default=str))

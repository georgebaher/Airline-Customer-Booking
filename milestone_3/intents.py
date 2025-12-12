"""Airline intent definitions for the Graph-RAG Travel Assistant.

Each intent represents a user query category with examples and descriptions.
"""

from typing import Any, Dict, List

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

# Precomputed helpers
INTENT_IDS = ", ".join(intent["id"] for intent in AIRLINE_INTENTS)
INTENT_ID_SET = {intent["id"] for intent in AIRLINE_INTENTS}

INTENT_TEXT = "\n".join(
    f"- {intent['id']}: {intent['description']}"
    for intent in AIRLINE_INTENTS
)


def get_intent_by_id(intent_id: str) -> Dict[str, Any] | None:
    """Return intent definition by id, or None if not found."""
    for intent in AIRLINE_INTENTS:
        if intent["id"] == intent_id:
            return intent
    return None


__all__ = [
    "AIRLINE_INTENTS",
    "INTENT_IDS",
    "INTENT_ID_SET",
    "INTENT_TEXT",
    "get_intent_by_id",
]

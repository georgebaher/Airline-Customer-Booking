"""Airline intent definitions for the Graph-RAG Travel Assistant.

Each intent represents a user query category with examples and descriptions.
"""

from typing import Any, Dict, List

AIRLINE_INTENTS: List[Dict[str, Any]] = [
    { #1
        "id": "route_search",
        "label": "Route Search",
        "description": "Find flights between origin and destination.",
        "examples": ["Flights from JFK to LAX"],
    },
    { #2
        "id": "check_avg_delay",
        "label": "Check Average Delay",
        "description": "Check the average arrival delay minutes for a specific route or airline.",
        "examples": ["How late are flights from LAX to NYC usually?"],
    },
    { #3
        "id": "food_quality_inquiry",
        "label": "Food Quality Inquiry",
        "description": "Find food satisfaction scores, optionally filtered by passenger class.",
        "examples": ["How is the food in Business Class?"],
    },
    { #4
        "id": "fleet_info",
        "label": "Fleet Information",
        "description": "Identify the fleet type used to a specific destination, optionally from a specific origin.",
        "examples": ["What kind of plane flies to LAX?", "What kind of plane flies from JFK to LAX?"],
    },
    { #5
        "id": "loyalty_demographics",
        "label": "Loyalty Program Stats",
        "description": "Analyze metrics based on loyalty program levels.",
        "examples": ["What class do premier gold members usually fly?", "Satisfaction score for premier platinum members"],
    },
    { #6
        "id": "direct_flight_search",
        "label": "Direct Flight Search",
        "description": "Search for flights with minimum legs (direct or non-stop).",
        "examples": ["Are there direct flights to LAX?", "Show me flights with only 1 leg"],
    },
    { #7
        "id": "distance_inquiry",
        "label": "Distance Inquiry",
        "description": "Query the actual flown miles for a trip.",
        "examples": ["How many miles is the flight from LHR to JFK"],
    },
    { #8
        "id": "low_delay_route_search",
        "label": "Low Delay Route Search",
        "description": "Find flights between specific airports with minimal delays.",
        "examples": ["Find flights from ORD to LAX with minimal delays", "Show me low-delay flights from JFK to SFO"],
    },
    { #9
        "id": "class_comparison",
        "label": "Class Comparison",
        "description": "Compare metrics (food, delay) between passenger classes.",
        "examples": ["Compare food score between Economy and First Class", "Difference in miles earned for Business vs Economy"],
    },
    { #10
        "id": "aircraft_performance",
        "label": "Aircraft Performance",
        "description": "Correlate fleet type description with delay or food satisfaction.",
        "examples": ["Do B737-900 fleets have good food ratings?", "Delays on the A320-200 fleet"],
    },
    { #11
        "id": "multi_leg_search",
        "label": "Multi-Leg Search",
        "description": "Find flights that specifically involve layovers (multiple legs).",
        "examples": ["Cheaper flights with stops to LAX", "Routes with more than 1 leg"],
    },
    { #12
        "id": "generation_inquiry",
        "label": "Passenger Generation Inquiry",
        "description": "Analyze flight preferences and metrics based on passenger generation (e.g., Millennial, Gen X).",
        "examples": ["What planes do Millennials usually fly?", "What aircraft do Gen X passengers prefer?", "Fleet usage for Boomers"],
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



"""Feature Vector Embeddings for Airline Journey data.

Implements approach 2.b.2 from the project requirements:
- Builds text descriptions from journey numerical properties
- Generates embeddings using sentence transformers
- Supports multiple embedding models for comparison
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise SystemExit(
        "sentence-transformers is required. Install with: pip install sentence-transformers"
    ) from exc


# Supported embedding models for comparison (as required by project spec)
EMBEDDING_MODELS = {
    "minilm": "all-MiniLM-L6-v2",      # 384 dims, fast, good quality
    "mpnet": "all-mpnet-base-v2",       # 768 dims, higher quality, slower
}

DEFAULT_MODEL = "minilm"


def _get_model(model_key: str) -> SentenceTransformer:
    """Load embedding model."""
    model_name = EMBEDDING_MODELS.get(model_key, model_key)
    return SentenceTransformer(model_name)


def build_journey_description(journey: Dict[str, Any]) -> str:
    """
    Convert journey properties into a searchable text description.
    
    This is the core of approach 2.b.2 - constructing text from numerical
    properties for embedding.
    
    Example output:
        "Economy class B777-200 from LAX to IAX, 1379 miles, food rating 3/5, 
         delay -13 minutes (early), 1 leg direct, non-elite passenger, Millennial"
    """
    parts = []
    
    # Passenger class
    pax_class = journey.get("passenger_class", "Unknown")
    parts.append(f"{pax_class} class")
    
    # Fleet type
    fleet = journey.get("fleet_type_description")
    if fleet:
        parts.append(f"on {fleet}")
    
    # Route (if available)
    origin = journey.get("origin_station_code")
    dest = journey.get("destination_station_code")
    if origin and dest:
        parts.append(f"from {origin} to {dest}")
    
    # medium-haul being between 1,600–4,000 km; 900–2,200 nmi;
    # Distance
    miles = journey.get("actual_flown_miles")
    if miles is not None:
        if miles > 2200:
            parts.append(f"{miles} miles long-haul")
        elif miles > 900:
            parts.append(f"{miles} miles medium-haul")
        else:
            parts.append(f"{miles} miles short-haul")
    
    # Food satisfaction
    food_score = journey.get("food_satisfaction_score")
    if food_score is not None:
        if food_score >= 4:
            parts.append(f"excellent food {food_score}/5")
        elif food_score >= 3:
            parts.append(f"good food {food_score}/5")
        else:
            parts.append(f"poor food {food_score}/5")
    
    # Delay
    delay = journey.get("arrival_delay_minutes")
    if delay is not None:
        if delay <= -15:
            parts.append(f"very early arrival {delay} minutes")
        elif delay < 0:
            parts.append(f"early arrival {delay} minutes")
        elif delay == 0:
            parts.append("on-time arrival")
        elif delay <= 30:
            parts.append(f"slight delay {delay} minutes")
        else:
            parts.append(f"significant delay {delay} minutes")
    
    # Number of legs
    legs = journey.get("number_of_legs")
    if legs is not None:
        if legs == 1:
            parts.append("direct flight")
        else:
            parts.append(f"{legs} legs with connections")
    
    # Loyalty level
    loyalty = journey.get("loyalty_program_level")
    if loyalty:
        parts.append(f"{loyalty} member")
    
    # Generation
    generation = journey.get("generation")
    if generation:
        parts.append(f"{generation} traveler")
    
    return ", ".join(parts)


def build_journey_description_simple(journey: Dict[str, Any]) -> str:
    """
    Convert journey properties into a simple text description.
    
    Just column names and values, no interpretive descriptions.
    
    Example output:
        "passenger_class: Economy, fleet: B777-200, origin: LAX, destination: IAX, 
         miles: 1379, food_score: 3, delay: -13, legs: 1, loyalty: non-elite, generation: Millennial"
    """
    parts = []
    
    if journey.get("passenger_class"):
        parts.append(f"passenger_class: {journey['passenger_class']}")
    
    if journey.get("fleet_type_description"):
        parts.append(f"fleet: {journey['fleet_type_description']}")
    
    if journey.get("origin_station_code"):
        parts.append(f"origin: {journey['origin_station_code']}")
    
    if journey.get("destination_station_code"):
        parts.append(f"destination: {journey['destination_station_code']}")
    
    if journey.get("actual_flown_miles") is not None:
        parts.append(f"miles: {journey['actual_flown_miles']}")
    
    if journey.get("food_satisfaction_score") is not None:
        parts.append(f"food_score: {journey['food_satisfaction_score']}")
    
    if journey.get("arrival_delay_minutes") is not None:
        parts.append(f"delay: {journey['arrival_delay_minutes']}")
    
    if journey.get("number_of_legs") is not None:
        parts.append(f"legs: {journey['number_of_legs']}")
    
    if journey.get("loyalty_program_level"):
        parts.append(f"loyalty: {journey['loyalty_program_level']}")
    
    if journey.get("generation"):
        parts.append(f"generation: {journey['generation']}")
    
    return ", ".join(parts)


def embed_text(text: str, model_key: str = DEFAULT_MODEL) -> List[float]:
    """
    Generate embedding vector for text using specified model.
    
    Args:
        text: Text to embed
        model_key: Key from EMBEDDING_MODELS or full model name
        
    Returns:
        List of floats representing the embedding vector
    """
    model = _get_model(model_key)
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_query(query: str, model_key: str = DEFAULT_MODEL) -> List[float]:
    """
    Embed a user query for similarity search.
    
    This is used at query time to find similar journeys.
    """
    return embed_text(query, model_key)


def get_embedding_dimension(model_key: str = DEFAULT_MODEL) -> int:
    """Get the embedding dimension for a model."""
    model = _get_model(model_key)
    return model.get_sentence_embedding_dimension()


def batch_embed_journeys(
    journeys: List[Dict[str, Any]], 
    model_key: str = DEFAULT_MODEL,
    show_progress: bool = True,
    description_builder=build_journey_description
) -> List[Dict[str, Any]]:
    """
    Batch embed multiple journeys efficiently.
    
    Returns list of dicts with 'journey', 'description', and 'embedding' keys.
    """
    model = _get_model(model_key)
    
    # Build all descriptions first
    descriptions = [description_builder(j) for j in journeys]
    
    # Batch encode for efficiency
    embeddings = model.encode(
        descriptions, 
        convert_to_numpy=True,
        show_progress_bar=show_progress
    )
    
    return [
        {
            "journey": journey,
            "description": desc,
            "embedding": emb.tolist()
        }
        for journey, desc, emb in zip(journeys, descriptions, embeddings)
    ]



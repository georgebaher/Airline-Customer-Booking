from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

try:
    from neo4j import GraphDatabase
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "neo4j driver is required. Install it with `pip install neo4j`."
    ) from exc


CONFIG_PATH = Path(__file__).with_name("neo4jconfig.txt")


@dataclass
class Neo4jConfig:
    uri: str
    username: str
    password: str


def load_config(path: Path = CONFIG_PATH) -> Neo4jConfig:
    """Read connection settings from neo4jconfig.txt."""
    if not path.exists():
        raise FileNotFoundError(f"Neo4j config file not found: {path}")

    settings: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        settings[key.strip().upper()] = value.strip()

    missing = [key for key in ("URI", "USERNAME", "PASSWORD") if key not in settings]
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")

    return Neo4jConfig(settings["URI"], settings["USERNAME"], settings["PASSWORD"])


def run_query(
    query: str, parameters: Optional[Dict[str, Any]] = None, config_path: Path = CONFIG_PATH
) -> Tuple[list, float]:
    """
    Execute a Cypher query and return rows plus elapsed milliseconds.

    Parameters:
        query: Cypher string with $-style placeholders.
        parameters: optional dict of query parameters.
        config_path: override path to the Neo4j config file, if needed.
    """
    config = load_config(config_path)
    driver = GraphDatabase.driver(config.uri, auth=(config.username, config.password))

    verifier = getattr(driver, "verify_connectivity", None)
    if verifier:
        verifier()

    try:
        with driver.session() as session:
            start = perf_counter()
            result = session.run(query, **(parameters or {}))
            rows = result.data()
            elapsed_ms = (perf_counter() - start) * 1000
    finally:
        driver.close()

    return rows, elapsed_ms


def create_vector_index(
    index_name: str = "journey_embeddings",
    node_label: str = "Journey",
    property_name: str = "embedding",
    dimensions: int = 384,
    similarity_function: str = "cosine",
    config_path: Path = CONFIG_PATH
) -> bool:
    """
    Create a vector index in Neo4j for similarity search.
    
    Returns True if index was created, False if it already exists.
    """
    query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (n:{node_label})
    ON n.{property_name}
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: '{similarity_function}'
        }}
    }}
    """
    try:
        run_query(query, config_path=config_path)
        return True
    except Exception:
        return False


def update_journey_properties(
    feedback_id: str,
    properties: Dict[str, Any],
    config_path: Path = CONFIG_PATH
) -> None:
    """Update arbitrary properties on a Journey node."""
    if not properties:
        return

    sets = []
    params = {"feedback_id": feedback_id}
    
    for key, value in properties.items():
        # We assume keys are safe property names
        sets.append(f"j.{key} = ${key}")
        params[key] = value

    query = f"""
    MATCH (j:Journey {{feedback_ID: $feedback_id}})
    SET {", ".join(sets)}
    """
    
    run_query(query, params, config_path)


def similarity_search(
    query_embedding: list,
    top_k: int = 5,
    property_name: str = "embedding",
    config_path: Path = CONFIG_PATH
) -> Tuple[list, float]:
    """
    Find the top-k most similar journeys using vector similarity.
    
    Target specific embedding property with property_name.
    """
    query = f"""
    MATCH (j:Journey)
    WHERE j.{property_name} IS NOT NULL
    WITH j, vector.similarity.cosine(j.{property_name}, $query_embedding) AS score
    ORDER BY score DESC
    LIMIT $top_k
    MATCH (j)-[:ON]->(f:Flight)
    OPTIONAL MATCH (f)-[:DEPARTS_FROM]->(o:Airport)
    OPTIONAL MATCH (f)-[:ARRIVES_AT]->(d:Airport)
    RETURN 
        j.feedback_ID as feedback_id,
        j.passenger_class as passenger_class,
        j.food_satisfaction_score as food_score,
        j.arrival_delay_minutes as delay,
        j.actual_flown_miles as miles,
        j.number_of_legs as legs,
        j.loyalty_program_level as loyalty,
        j.generation as generation,
        f.fleet_type_description as fleet,
        o.station_code as origin,
        d.station_code as destination,
        score
    """
    return run_query(query, {"query_embedding": query_embedding, "top_k": top_k}, config_path)

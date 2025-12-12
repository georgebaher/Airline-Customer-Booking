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


__all__ = ["Neo4jConfig", "load_config", "run_query"]

from __future__ import annotations

import csv
from Config import Config, load_config
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Optional

try:
    from neo4j import GraphDatabase
except ImportError as exc:
    raise SystemExit(
        "neo4j driver is required. Install it with `pip install neo4j`."
    ) from exc

# Define paths to configuration and data files. Path(__file__) gives the current file's path, with_name changes the filename.
CONFIG_PATH = Path(__file__).with_name("config.txt")
DATA_PATH = Path(__file__).with_name("Airline_surveys_sample.csv")


# Function to clean string values by stripping whitespace.
def clean(value: str) -> str:
    return value.strip() # Remove leading/trailing whitespace

# Function to read rows from the CSV file and yield cleaned dictionaries.
def read_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # IMPORTANT: the whitespaces in the keys below are intentional, as they match the CSV header names.
            yield {
                "flight_number": clean(row["flight_number"]),
                "origin": clean(row[" origin_station_code"]),
                "destination": clean(row[" destination_station_code"]),
                "record_locator": clean(row[" record_locator"]),
                "arrival_delay_minutes": int(clean(row[" arrival_delay_minutes"])),
                "passenger_class": clean(row[" passenger_class"]),
                "number_of_legs": int(clean(row[" number_of_legs"])),
                "loyalty_program_level": clean(row[" loyalty_program_level"]),
                "generation": clean(row[" generation"]),
                "fleet_type_description": clean(row[" fleet_type_description"]),
                "actual_flown_miles": float(clean(row[" actual_flown_miles"])),
                "food_satisfaction_score":int(clean(row[" food_satisfaction_score"])),
                "feedback_ID": clean(row[" feedback_ID"]),
            }

# Function to create uniqueness constraints in the Neo4j database.
def create_constraints(session) -> None:
    """Ensure uniqueness constraints match the schema."""
    session.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Passenger) REQUIRE p.record_locator IS UNIQUE"
    )
    session.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Journey) REQUIRE j.feedback_ID IS UNIQUE"
    )
    session.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Airport) REQUIRE a.station_code IS UNIQUE"
    )
    session.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Flight) REQUIRE "
        "(f.flight_number, f.fleet_type_description) IS UNIQUE"
    )

# Function to clear the database before loading fresh data.
def clear_database(session) -> None:
    """Remove all nodes and relationships before loading fresh data."""
    session.run("MATCH (n) DETACH DELETE n")

# Function to ingest a single row into the Neo4j database.
def ingest_row(tx, row: Dict[str, object]) -> None:
    tx.run(
        """
        MERGE (p:Passenger {record_locator: $record_locator})
        SET p.loyalty_program_level = $loyalty_program_level,
            p.generation = $generation

        MERGE (j:Journey {feedback_ID: $feedback_ID})
        SET j.food_satisfaction_score = $food_satisfaction_score,
            j.arrival_delay_minutes = $arrival_delay_minutes,
            j.actual_flown_miles = $actual_flown_miles,
            j.number_of_legs = $number_of_legs,
            j.passenger_class = $passenger_class

        MERGE (f:Flight {flight_number: $flight_number, fleet_type_description: $fleet_type_description})

        MERGE (origin:Airport {station_code: $origin})
        MERGE (dest:Airport {station_code: $destination})

        MERGE (p)-[:TOOK]->(j)
        MERGE (j)-[:ON]->(f)
        MERGE (f)-[:DEPARTS_FROM]->(origin)
        MERGE (f)-[:ARRIVES_AT]->(dest)
        """,
        **row,
    )

# Function to load the entire graph into Neo4j.
def load_graph(config: Config) -> None:
    print(f"[INFO] Connecting to {config.uri} as {config.username}...")
    driver = GraphDatabase.driver(config.uri, auth=(config.username, config.password))
    verifier = getattr(driver, "verify_connectivity", None)
    if verifier:
        verifier()
        print("[SUCCESS] Connectivity verified.")

    start = perf_counter()
    with driver.session() as session:
        print("[INFO] Clearing database...")
        clear_database(session)
        print("[SUCCESS] Database cleared.")
        print("[INFO] Creating constraints...")
        create_constraints(session)
        print("[SUCCESS] Constraints created.")
        print(f"[INFO] Ingesting data from {DATA_PATH}...")
        rows_ingested = 0
        for row in read_rows(DATA_PATH):
            session.execute_write(ingest_row, row)
            rows_ingested += 1
        print(f"[SUCCESS] Ingested {rows_ingested} rows.")
    driver.close()
    elapsed_ms = (perf_counter() - start) * 1000
    print(f"[INFO] load_graph completed in {elapsed_ms:.1f} ms")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")
    config = load_config(CONFIG_PATH)
    load_graph(config)

if __name__ == "__main__":
    main()

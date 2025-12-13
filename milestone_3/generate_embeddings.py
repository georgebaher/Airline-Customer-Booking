"""Generate embeddings for all journeys from the CSV file.

This script:
1. Reads journey data from Airline_surveys_sample.csv
2. Builds text descriptions for each journey
3. Generates embeddings using sentence-transformers
4. Stores embeddings back in Neo4j
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable

from embeddings import (
    EMBEDDING_MODELS,
    batch_embed_journeys,
    get_embedding_dimension,
    build_journey_description_simple,
    build_journey_description,
)
from run_queries import (
    create_vector_index,
    update_journey_properties,
)



# ========== CONFIGURATION ==========
MODELS = ["minilm", "mpnet"]
STYLES = [
    ("rich", build_journey_description),
    ("simple", build_journey_description_simple),
]
CREATE_INDEX = True
LIMIT = None
# ===================================


CSV_PATH = Path(__file__).with_name("Airline_surveys_sample.csv")


def clean(value: str) -> str:
    return value.strip()


def read_journeys_from_csv(path: Path = CSV_PATH) -> Iterable[Dict[str, object]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {
                "feedback_ID": clean(row["feedback_ID"]),
                "passenger_class": clean(row["passenger_class"]),
                "food_satisfaction_score": int(clean(row["food_satisfaction_score"])),
                "arrival_delay_minutes": int(clean(row["arrival_delay_minutes"])),
                "actual_flown_miles": float(clean(row["actual_flown_miles"])),
                "number_of_legs": int(clean(row["number_of_legs"])),
                "fleet_type_description": clean(row["fleet_type_description"]),
                "origin_station_code": clean(row["origin_station_code"]),
                "destination_station_code": clean(row["destination_station_code"]),
                "loyalty_program_level": clean(row["loyalty_program_level"]),
                "generation": clean(row["generation"]),
            }


def generate_embeddings(limit: int = LIMIT):
    """Main function to generate and store 4-way embeddings."""
    
    # 1. Setup Indexes
    if CREATE_INDEX:
        print("Checking/Creating vector indexes...")
        for model in MODELS:
            dims = get_embedding_dimension(model)
            for style, _ in STYLES:
                index_name = f"journey_embedding_{model}_{style}"
                prop_name = f"embedding_{model}_{style}"
                print(f"   - {index_name} ({dims} dims)")
                create_vector_index(
                    index_name=index_name,
                    property_name=prop_name,
                    dimensions=dims
                )

    # 2. Read Data
    print(f"\nReading journeys from {CSV_PATH.name}...")
    journeys = list(read_journeys_from_csv())
    if limit:
        journeys = journeys[:limit]
        print(f"   (Limited to {len(journeys)} for testing)")
    
    if not journeys:
        print("No journeys found!")
        sys.exit(1)

    # Dictionary to aggregate all updates per journey ID
    # structure: {feedback_id: {prop: value, ...}}
    updates_by_id: Dict[str, Dict[str, Any]] = {
        j["feedback_ID"]: {} for j in journeys
    }

    # 3. Generate Embeddings (Loop over Model x Style)
    total_start = perf_counter()
    
    for model in MODELS:
        dims = get_embedding_dimension(model)
        print(f"\nModel: {model} ({dims} dims)")
        
        for style, builder in STYLES:
            print(f"   Style: {style}...")
            start = perf_counter()
            
            results = batch_embed_journeys(
                journeys,
                model_key=model,
                show_progress=True,
                description_builder=builder
            )
            
            # Aggregate results
            for res in results:
                fid = res["journey"]["feedback_ID"]
                if fid not in updates_by_id:
                    continue
                    
                # Store the vector
                vec_prop = f"embedding_{model}_{style}"
                updates_by_id[fid][vec_prop] = res["embedding"]
                
                # Store the text description (overwrite is fine as it depends only on style)
                desc_prop = f"description_{style}"
                updates_by_id[fid][desc_prop] = res["description"]
            
            elapsed = perf_counter() - start
            print(f"     -> Done in {elapsed:.1f}s")

    gen_time = perf_counter() - total_start
    print(f"\nAll embeddings generated in {gen_time:.1f}s")

    # 4. Store in Neo4j
    print("\nStoring aggregated properties to Neo4j...")
    store_start = perf_counter()
    success = 0
    errors = 0
    
    sorted_ids = sorted(updates_by_id.keys()) # sort for consistent progress
    total = len(sorted_ids)
    
    for i, fid in enumerate(sorted_ids):
        props = updates_by_id[fid]
        if not props:
            continue
            
        try:
            update_journey_properties(fid, props)
            success += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"   Error {fid}: {e}")
        
        if (i+1) % 100 == 0:
            print(f"   ... {i+1}/{total}")

    store_time = perf_counter() - store_start
    print(f"\nFinished!")
    print(f"   Stored: {success}/{total} nodes")
    print(f"   Errors: {errors}")
    print(f"   Total Time: {gen_time + store_time:.1f}s")


if __name__ == "__main__":
    generate_embeddings()



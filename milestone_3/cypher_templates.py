"""Cypher query templates for the Airline Graph-RAG system.

Each template corresponds to an intent and uses parameterized queries for Neo4j.
"""

from typing import Dict, Optional

CYPHER_TEMPLATES: Dict[str, str] = {
    #1
    "route_search": """ 
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        RETURN f.flight_number, f.fleet_type_description
        LIMIT 5;
    """,
    #2
    "check_avg_delay": """
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        MATCH (j:Journey)-[:ON]->(f)
        RETURN avg(j.arrival_delay_minutes) as average_delay;
    """,
    #3
    "food_quality_inquiry": """
        MATCH (j:Journey)
        WHERE ($class IS NULL OR j.passenger_class = $class)
        RETURN j.passenger_class, avg(j.food_satisfaction_score) as avg_food_score
        ORDER BY avg_food_score DESC;
    """,
    #4
    "fleet_info": """
        MATCH (f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        WHERE $origin IS NULL OR EXISTS {
            MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f)
        }
        RETURN DISTINCT f.fleet_type_description
        ORDER BY f.fleet_type_description;
    """,
    #5
    "loyalty_demographics": """
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        WHERE toLower(p.loyalty_program_level) = toLower($level)
        RETURN j.passenger_class, avg(j.food_satisfaction_score) as avg_sat
        ORDER BY avg_sat DESC;
    """,
    #6
    "direct_flight_search": """
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        MATCH (j:Journey)-[:ON]->(f)
        WHERE j.number_of_legs = 1
        RETURN o.station_code AS origin_station_code, d.station_code AS destination_station_code, j.actual_flown_miles;
    """,
    #7
    "distance_inquiry": """
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        MATCH (j:Journey)-[:ON]->(f)
        RETURN avg(j.actual_flown_miles) as miles LIMIT 1;
    """,
    #8
    "low_delay_route_search": """
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        MATCH (j:Journey)-[:ON]->(f)
        RETURN f.flight_number, f.fleet_type_description, avg(j.arrival_delay_minutes) as avg_delay
        ORDER BY avg_delay ASC
        LIMIT 5;
    """,
    #9
    "class_comparison": """
        MATCH (j:Journey)
        RETURN j.passenger_class, avg(j.food_satisfaction_score) as food, avg(j.arrival_delay_minutes) as delay
        ORDER BY j.passenger_class;
    """,
    #10
    "aircraft_performance": """
        MATCH (f:Flight)
        WHERE f.fleet_type_description = $fleet
        MATCH (j:Journey)-[:ON]->(f)
        RETURN f.fleet_type_description, avg(j.arrival_delay_minutes) as delay, avg(j.food_satisfaction_score) as food;
    """,
    #11
    "multi_leg_search": """
        MATCH (o:Airport {station_code: $origin})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: $dest})
        MATCH (j:Journey)-[:ON]->(f)
        WHERE j.number_of_legs > 1
        RETURN j.number_of_legs, avg(j.arrival_delay_minutes) as avg_delay;
    """,
    #12
    "generation_inquiry": """
        MATCH (p:Passenger {generation: $generation})-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
        RETURN f.fleet_type_description, count(*) as flights_operated
        ORDER BY flights_operated DESC;
    """,
}


def get_cypher_template(intent_id: str) -> Optional[str]:
    """Return the Cypher template for a supported intent id."""
    return CYPHER_TEMPLATES.get(intent_id)



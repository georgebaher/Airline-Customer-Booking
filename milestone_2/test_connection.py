"""
Lightweight connectivity check against the configured Neo4j instance.
- Loads credentials from config.txt
- Verifies the driver can reach the server
- Executes a trivial test query
"""

import sys
import time
from pathlib import Path

from neo4j import GraphDatabase

from Config import load_config

CONFIG_PATH = Path(__file__).with_name("config.txt")


def run_test(config_path: Path) -> int:
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return 1

    print(f"[INFO] Loading config from {config_path}")
    config = load_config(config_path)

    driver = None
    try:
        print(f"[INFO] Connecting to {config.uri} as {config.username} ...")
        driver = GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
        )
        verifier = getattr(driver, "verify_connectivity", None)
        if verifier:
            verifier()
            print("[SUCCESS] Connectivity verified.")

        with driver.session() as session:
            start = time.perf_counter()
            _ = session.run("RETURN 1") # Trivial test query
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"[SUCCESS] Connection test succeeded in {elapsed_ms:.2f} ms.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Connection test failed: {exc}")
        return 1
    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    sys.exit(run_test(CONFIG_PATH))

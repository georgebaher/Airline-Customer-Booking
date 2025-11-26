from pathlib import Path
from time import perf_counter
import sys
import json

try:
    from neo4j import GraphDatabase
except ImportError as exc:
    raise SystemExit(
        "neo4j driver is required. Install it with `pip install neo4j`."
    ) from exc

from Config import load_config

CONFIG_PATH = Path(__file__).with_name("config.txt")
QUERIES_PATH = Path(__file__).with_name("queries.txt")
OUTPUT_DIR = Path(__file__).with_name("Airline - Our Output Query Answers")


def load_queries(path: Path):
    """Load queries separated by trailing semicolons, skipping // comments."""
    if not path.exists():
        raise FileNotFoundError(f"queries file not found: {path}")

    statements = []
    buffer = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        buffer.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(buffer).rstrip(";\n\r ")
            if stmt:
                statements.append(stmt)
            buffer = []

    return statements


def run_query(session, query: str, index: int, total: int):
    print(f"\n[INFO] Running query {index}/{total}...")
    start = perf_counter()
    result = session.run(query)
    rows = result.data()
    elapsed_ms = (perf_counter() - start) * 1000

    print(f"[SUCCESS] Completed in {elapsed_ms:.1f} ms; returned {len(rows)} row(s).")
    return rows, elapsed_ms


def save_results(index: int, rows, elapsed_ms: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"query_{index}.json"
    payload = {
        "query_index": index,
        "row_count": len(rows),
        "elapsed_ms": round(elapsed_ms, 3),
        "rows": rows,
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    print(f"[SUCCESS] Saved results to {output_file}.")


def main():
    config = load_config(CONFIG_PATH)
    queries = load_queries(QUERIES_PATH)
    if not queries:
        print("[ERROR] No queries found in queries.txt")
        return 1

    selected = list(enumerate(queries, start=1))
    print(f"[INFO] Connecting to {config.uri} as {config.username}...")
    driver = GraphDatabase.driver(config.uri, auth=(config.username, config.password))
    verifier = getattr(driver, "verify_connectivity", None)
    if verifier:
        verifier()
        print("[SUCCESS] Connectivity verified.")

    try:
        with driver.session() as session:
            total = len(selected)
            for i, (query_idx, q) in enumerate(selected, start=1):
                rows, elapsed_ms = run_query(session, q, i, total)
                save_results(query_idx, rows, elapsed_ms)
    finally:
        driver.close()
        print("\n[INFO] All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

# Define configuration dataclass. A dataclass is a simple way to store data attributes and access in a clean way (eg. config.uri).
@dataclass
class Config:
    uri: str
    username: str
    password: str

# Function to load configuration from a file.
def load_config(path: Path) -> Config:
    """Read connection settings from config.txt."""
    settings: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or "=" not in line: # Skip empty lines or lines without '='
            continue
        key, value = line.split("=", 1) 
        settings[key.strip().upper()] = value.strip()
    missing = [key for key in ("URI", "USERNAME", "PASSWORD") if key not in settings]
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    return Config(settings["URI"], settings["USERNAME"], settings["PASSWORD"])
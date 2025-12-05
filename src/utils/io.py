"""I/O utilities for saving/loading results"""

import json
from pathlib import Path
from datetime import datetime


def save_results(results: dict, output_dir: str):
    """Save results with timestamp"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.json"

    with open(Path(output_dir) / filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}/{filename}")


def load_results(filepath: str) -> dict:
    """Load results from file"""
    with open(filepath) as f:
        return json.load(f)

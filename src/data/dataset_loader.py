"""Unified dataset loader for various formats"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class DatasetLoader:
    """
    Unified loader for medical QA datasets.

    Supports:
    - JSONL format
    - JSON format (array of objects)
    - Multiple dataset types (MedQA, BioASQ, PubMedQA, etc.)
    """

    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: Root directory for datasets
        """
        self.data_dir = Path(data_dir)

    def load_dataset(
        self,
        dataset_path: str,
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load dataset from file.

        Args:
            dataset_path: Path to dataset file (relative to data_dir or absolute)
            max_samples: Maximum number of samples to load

        Returns:
            List of dataset items
        """
        # Resolve path
        path = Path(dataset_path)
        if not path.is_absolute():
            path = self.data_dir / dataset_path

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        # Load based on extension
        if path.suffix == ".jsonl":
            data = self._load_jsonl(path)
        elif path.suffix == ".json":
            data = self._load_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Limit samples
        if max_samples is not None:
            data = data[:max_samples]

        return data

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSON file (array or object)"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # If dict, try common keys
            for key in ["data", "questions", "samples"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # Otherwise, wrap in list
            return [data]
        else:
            raise ValueError(f"Unexpected JSON format in {path}")

    def save_dataset(
        self,
        data: List[Dict[str, Any]],
        output_path: str,
        format: str = "jsonl"
    ):
        """
        Save dataset to file.

        Args:
            data: Dataset items
            output_path: Output file path
            format: Output format ("jsonl" or "json")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

"""Configuration loader and component builder"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class RetrieverConfig:
    """Configuration for retriever component"""
    type: str
    top_k: int = 5
    embedder: Optional[str] = None
    faiss_index_paths: Optional[Dict[str, str]] = None
    articles_paths: Optional[Dict[str, str]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankerConfig:
    """Configuration for reranker component"""
    type: str
    top_k: Optional[int] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratorConfig:
    """Configuration for generator component"""
    type: str
    model_path: str
    max_tokens: int = 200
    temperature: float = 0.0
    batch_size: int = 8
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatorConfig:
    """Configuration for evaluator component"""
    metrics: list = field(default_factory=lambda: ["accuracy", "rouge"])
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    # Pipeline
    pipeline: str
    dataset: str
    output_path: str = "experiments/results"

    # Components
    retriever: Optional[RetrieverConfig] = None
    reranker: Optional[RerankerConfig] = None
    generator: Optional[GeneratorConfig] = None
    evaluator: Optional[EvaluatorConfig] = None

    # Global settings
    max_samples: Optional[int] = None
    seed: int = 42
    use_few_shot: bool = True
    dataset_name: str = "med_qa"

    # Additional kwargs
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)


class ConfigLoader:
    """Loads and merges YAML configuration files"""

    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """Load YAML file"""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configs (override takes precedence).

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def load_config(
        config_path: str,
        base_config_path: Optional[str] = None
    ) -> ExperimentConfig:
        """
        Load experiment configuration from YAML.

        Args:
            config_path: Path to experiment config file
            base_config_path: Path to base config file (optional)

        Returns:
            ExperimentConfig object
        """
        # Load base config if provided
        config_dict = {}
        if base_config_path and Path(base_config_path).exists():
            config_dict = ConfigLoader.load_yaml(base_config_path)

        # Load experiment config and merge
        exp_config = ConfigLoader.load_yaml(config_path)
        config_dict = ConfigLoader.merge_configs(config_dict, exp_config)

        # Convert to typed config objects
        config = ConfigLoader._dict_to_config(config_dict)
        return config

    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dict to typed ExperimentConfig"""
        # Extract component configs
        retriever_config = None
        if "retriever" in config_dict:
            ret_dict = config_dict["retriever"]
            retriever_config = RetrieverConfig(
                type=ret_dict.get("type", "FAISSRetriever"),
                top_k=ret_dict.get("top_k", 5),
                embedder=ret_dict.get("embedder"),
                faiss_index_paths=ret_dict.get("faiss_index_paths"),
                articles_paths=ret_dict.get("articles_paths"),
                kwargs=ret_dict.get("kwargs", {})
            )

        reranker_config = None
        if "reranker" in config_dict:
            rer_dict = config_dict["reranker"]
            reranker_config = RerankerConfig(
                type=rer_dict.get("type", "NoReranker"),
                top_k=rer_dict.get("top_k"),
                kwargs=rer_dict.get("kwargs", {})
            )

        generator_config = None
        if "generator" in config_dict:
            gen_dict = config_dict["generator"]
            generator_config = GeneratorConfig(
                type=gen_dict.get("type", "LlamaGenerator"),
                model_path=gen_dict.get("model_path", "meta-llama/Llama-2-7b-chat-hf"),
                max_tokens=gen_dict.get("max_tokens", 200),
                temperature=gen_dict.get("temperature", 0.0),
                batch_size=gen_dict.get("batch_size", 8),
                kwargs=gen_dict.get("kwargs", {})
            )

        evaluator_config = None
        if "evaluator" in config_dict:
            eval_dict = config_dict["evaluator"]
            evaluator_config = EvaluatorConfig(
                metrics=eval_dict.get("metrics", ["accuracy"]),
                kwargs=eval_dict.get("kwargs", {})
            )

        return ExperimentConfig(
            pipeline=config_dict.get("pipeline", "StandardRAG"),
            dataset=config_dict.get("dataset", "med_qa"),
            output_path=config_dict.get("output_path", "experiments/results"),
            retriever=retriever_config,
            reranker=reranker_config,
            generator=generator_config,
            evaluator=evaluator_config,
            max_samples=config_dict.get("max_samples"),
            seed=config_dict.get("seed", 42),
            use_few_shot=config_dict.get("use_few_shot", True),
            dataset_name=config_dict.get("dataset_name", "med_qa"),
            kwargs=config_dict.get("kwargs", {})
        )

    @staticmethod
    def save_config(config: ExperimentConfig, output_path: str):
        """Save config to YAML file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_config(config_path: str) -> dict:
    """Simple config loader - returns dict"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_components(config: dict) -> dict:
    """Build all components from config"""
    from ..components.retrievers import FAISSRetriever
    from ..components.generators import VLLMGenerator, SelfBioRAGGenerator
    from ..components.evaluators import AccuracyEvaluator

    components = {}

    # Build retriever
    if config.get("retriever"):
        ret_cfg = config["retriever"]
        if ret_cfg["type"] == "FAISSRetriever":
            components["retriever"] = FAISSRetriever(
                faiss_index_paths=ret_cfg["faiss_index_paths"],
                articles_paths=ret_cfg["articles_paths"]
            )

    # Build generator
    gen_cfg = config["generator"]
    if gen_cfg["type"] == "VLLMGenerator":
        components["generator"] = VLLMGenerator(model_path=gen_cfg["model_path"])
    elif gen_cfg["type"] == "SelfBioRAGGenerator":
        components["generator"] = SelfBioRAGGenerator(model_path=gen_cfg["model_path"])

    # Build evaluator
    components["evaluator"] = AccuracyEvaluator()

    return components

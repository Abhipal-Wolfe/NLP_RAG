"""Component registry for factory pattern

Allows registering and retrieving components by name.
This enables config-driven component selection.
"""

from typing import Dict, Type, Any, Optional
import inspect


class ComponentRegistry:
    """Registry for managing component implementations"""

    def __init__(self):
        self._retrievers: Dict[str, Type] = {}
        self._rerankers: Dict[str, Type] = {}
        self._generators: Dict[str, Type] = {}
        self._evaluators: Dict[str, Type] = {}
        self._pipelines: Dict[str, Type] = {}

    def register_retriever(self, name: str, cls: Type):
        """Register a retriever implementation"""
        self._retrievers[name] = cls

    def register_reranker(self, name: str, cls: Type):
        """Register a reranker implementation"""
        self._rerankers[name] = cls

    def register_generator(self, name: str, cls: Type):
        """Register a generator implementation"""
        self._generators[name] = cls

    def register_evaluator(self, name: str, cls: Type):
        """Register an evaluator implementation"""
        self._evaluators[name] = cls

    def register_pipeline(self, name: str, cls: Type):
        """Register a pipeline implementation"""
        self._pipelines[name] = cls

    def get_retriever(self, name: str) -> Type:
        """Get retriever class by name"""
        if name not in self._retrievers:
            raise ValueError(f"Retriever '{name}' not registered. Available: {list(self._retrievers.keys())}")
        return self._retrievers[name]

    def get_reranker(self, name: str) -> Type:
        """Get reranker class by name"""
        if name not in self._rerankers:
            raise ValueError(f"Reranker '{name}' not registered. Available: {list(self._rerankers.keys())}")
        return self._rerankers[name]

    def get_generator(self, name: str) -> Type:
        """Get generator class by name"""
        if name not in self._generators:
            raise ValueError(f"Generator '{name}' not registered. Available: {list(self._generators.keys())}")
        return self._generators[name]

    def get_evaluator(self, name: str) -> Type:
        """Get evaluator class by name"""
        if name not in self._evaluators:
            raise ValueError(f"Evaluator '{name}' not registered. Available: {list(self._evaluators.keys())}")
        return self._evaluators[name]

    def get_pipeline(self, name: str) -> Type:
        """Get pipeline class by name"""
        if name not in self._pipelines:
            raise ValueError(f"Pipeline '{name}' not registered. Available: {list(self._pipelines.keys())}")
        return self._pipelines[name]

    def list_components(self) -> Dict[str, list]:
        """List all registered components"""
        return {
            "retrievers": list(self._retrievers.keys()),
            "rerankers": list(self._rerankers.keys()),
            "generators": list(self._generators.keys()),
            "evaluators": list(self._evaluators.keys()),
            "pipelines": list(self._pipelines.keys()),
        }


# Global registry instance
_registry = ComponentRegistry()


def register_component(component_type: str, name: str):
    """
    Decorator to register a component.

    Usage:
        @register_component("retriever", "FAISSRetriever")
        class MyRetriever(Retriever):
            ...
    """
    def decorator(cls):
        if component_type == "retriever":
            _registry.register_retriever(name, cls)
        elif component_type == "reranker":
            _registry.register_reranker(name, cls)
        elif component_type == "generator":
            _registry.register_generator(name, cls)
        elif component_type == "evaluator":
            _registry.register_evaluator(name, cls)
        elif component_type == "pipeline":
            _registry.register_pipeline(name, cls)
        else:
            raise ValueError(f"Unknown component type: {component_type}")
        return cls
    return decorator


def get_component(component_type: str, name: str, **kwargs) -> Any:
    """
    Factory function to instantiate a component.

    Args:
        component_type: Type of component (retriever, reranker, generator, evaluator, pipeline)
        name: Registered name of the component
        **kwargs: Arguments to pass to component constructor

    Returns:
        Instantiated component
    """
    if component_type == "retriever":
        cls = _registry.get_retriever(name)
    elif component_type == "reranker":
        cls = _registry.get_reranker(name)
    elif component_type == "generator":
        cls = _registry.get_generator(name)
    elif component_type == "evaluator":
        cls = _registry.get_evaluator(name)
    elif component_type == "pipeline":
        cls = _registry.get_pipeline(name)
    else:
        raise ValueError(f"Unknown component type: {component_type}")

    # Filter kwargs to only include valid constructor parameters
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return cls(**filtered_kwargs)


def get_registry() -> ComponentRegistry:
    """Get the global component registry"""
    return _registry

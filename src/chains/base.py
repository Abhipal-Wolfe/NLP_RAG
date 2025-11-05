"""Base chain class for Self-BioRAG chains

This provides a simple base class with the Chain interface we need,
without depending on langchain.chains which may not be installed.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class Chain(ABC):
    """
    Base class for all chains.
    
    Chains should implement:
    - input_keys: List of expected input keys
    - output_keys: List of output keys
    - _call: Core logic
    """
    
    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """Return list of input keys"""
        pass
    
    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
        """Return list of output keys"""
        pass
    
    @abstractmethod
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core logic of the chain.
        
        Args:
            inputs: Dictionary with keys matching input_keys
            
        Returns:
            Dictionary with keys matching output_keys
        """
        pass
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the chain.
        
        Args:
            inputs: Dictionary with input data
            
        Returns:
            Dictionary with output data
        """
        # Validate inputs
        missing_keys = set(self.input_keys) - set(inputs.keys())
        if missing_keys:
            # Allow optional keys by checking if they have defaults
            pass
        
        # Run the chain
        return self._call(inputs)
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience method - same as invoke()"""
        return self.invoke(inputs)


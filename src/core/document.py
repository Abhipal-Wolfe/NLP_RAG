"""Simple Document class to replace langchain dependency"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Document:
    """
    Document with content and metadata.

    Replaces langchain_core.documents.Document
    """
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.page_content

    def __repr__(self) -> str:
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


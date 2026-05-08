from __future__ import annotations


class TextChunker:
    """Simple character-based chunker with overlap."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(cleaned):
            end = min(start + self.chunk_size, len(cleaned))
            if end < len(cleaned):
                split_at = cleaned.rfind("\n\n", start, end)
                if split_at <= start:
                    split_at = cleaned.rfind("\n", start, end)
                if split_at <= start:
                    split_at = cleaned.rfind(" ", start, end)
                if split_at > start:
                    end = split_at

            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(cleaned):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks


from __future__ import annotations


class TextChunker:
    """按字符窗口切分文本的分块器。

    设计目标是提供一个稳定、可预测的最小实现：
    1. 以字符数控制单块大小，便于和 embedding 模型的输入上限对齐。
    2. 通过 overlap 保留上下文连续性，降低问答时跨块断裂的问题。
    3. 优先在段落、换行或空格位置截断，尽量减少语义被硬切开的情况。
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """初始化分块参数并做基础校验。"""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """将原始文本切分为多个可入库的文本块。

        返回值中的每个元素都是去掉首尾空白后的有效文本。
        如果输入为空文本，则直接返回空列表。
        """
        cleaned = text.strip()
        if not cleaned:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(cleaned):
            end = min(start + self.chunk_size, len(cleaned))
            if end < len(cleaned):
                # 优先按段落、换行、空格回退，减少语义被截断的概率。
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
            # 下一段从 overlap 位置回退，确保块之间存在可检索的上下文衔接。
            start = max(end - self.chunk_overlap, start + 1)
        return chunks

"""
文本分段模块
============

将长文档按照指定规则切分成小块（chunks）。

教学要点:
- 分段策略：基于分隔符的递归分段
- 分段大小：控制每个分段的长度
- 重叠机制：相邻分段之间保留重叠，保持上下文连贯性

为什么需要文本分段？
- LLM 有上下文长度限制，无法一次性处理过长文本
- 向量检索中，较短的文本片段匹配更精确
- 每个分段将作为一个独立的向量存储
"""

from typing import List, Optional


class TextChunker:
    """
    文本分段器

    功能：
    - 按分隔符切分文本
    - 控制分段大小
    - 保持相邻分段之间的重叠
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        """
        初始化分段器

        Args:
            chunk_size: 每个分段的最大字符数
            chunk_overlap: 相邻分段之间的重叠字符数
            separators: 分隔符列表，按优先级排序
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 默认分隔符（按优先级排序）
        self.separators = separators or ['\n# ', '\n## ', '\n### ', '\n\n', '\n', '。']

    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """使用指定分隔符切分文本"""
        if separator == '':
            return list(text)
        return text.split(separator)

    def _merge_chunks(self, chunks: List[str]) -> List[str]:
        """
        合并小片段，确保每个分段达到最小大小
        """
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for i in range(1, len(chunks)):
            chunk = chunks[i]

            if len(current) + len(chunk) <= self.chunk_size:
                current += chunk
            else:
                if current:
                    merged.append(current)
                current = chunk

        if current:
            merged.append(current)

        return merged

    def chunk_text(self, text: str) -> List[str]:
        """
        将文本切分成多个分段

        递归分段策略：
        1. 首先尝试使用最高优先级的分隔符（如双换行）切分
        2. 如果切分后的片段仍然太大，继续使用次优先级的分隔符切分
        3. 重复此过程，直到所有片段都在目标大小内
        4. 最后合并相邻的小片段
        """
        if not text or not text.strip():
            return []

        def split_recursively(
            text: str,
            separator_idx: int
        ) -> List[str]:
            # 如果已经达到最后一个分隔符，或者文本已经够短，直接返回
            if separator_idx >= len(self.separators):
                # 按字符切分
                return [text[i:i + self.chunk_size]
                        for i in range(0, len(text), self.chunk_size)]

            separator = self.separators[separator_idx]

            # 如果文本足够短，直接返回
            effective_chunk_size = self.chunk_size if self.chunk_size > 0 else len(text) + 1
            if len(text) <= effective_chunk_size:
                return [text]

            # 使用当前分隔符切分
            parts = text.split(separator)

            # 如果切分后片段数量为1，说明没有找到分隔符，尝试下一个
            if len(parts) == 1:
                return split_recursively(text, separator_idx + 1)

            # 递归处理每个部分
            result = []
            for part in parts:
                if part.strip():
                    sub_chunks = split_recursively(part, separator_idx + 1)
                    result.extend(sub_chunks)

            return result

        # 开始递归切分
        raw_chunks = split_recursively(text, 0)

        # 过滤空片段
        raw_chunks = [c.strip() for c in raw_chunks if c.strip()]

        # 合并过小的片段
        merged_chunks = self._merge_chunks(raw_chunks)

        return merged_chunks

    def chunk_text_with_overlap(self, text: str) -> List[dict]:
        """
        将文本切分成重叠的分段

        返回分段字典列表，每个字典包含：
        - index: 分段索引
        - text: 分段文本内容
        - start_char: 在原文中的起始位置
        - end_char: 在原文中的结束位置
        """
        chunks = self.chunk_text(text)

        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                start_char = 0
            else:
                prev_end = result[i - 1]['end_char']
                start_char = max(0, prev_end - self.chunk_overlap)

            end_char = start_char + len(chunk)

            if end_char > len(text):
                start_char = len(text) - len(chunk)
                end_char = len(text)

            result.append({
                'index': i,
                'text': chunk,
                'start_char': start_char,
                'end_char': end_char
            })

        return result

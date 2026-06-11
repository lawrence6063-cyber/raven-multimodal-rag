"""Document Structure Splitter — 基于文档结构元素的智能分割。

================================================================================
📖 Document Structure Split 原理说明
================================================================================

【核心思想】
不同于 Recursive Splitter 按字符分隔符切分，Document Structure Splitter
识别文档的**结构化元素**（标题、段落、表格、代码块、图片引用等），
将它们作为独立的语义单元进行切分。

【为什么多模态 RAG 需要它？】
在图文混排的文档中：
- 表格应该作为完整的一个 chunk，不能从中间截断
- 代码块应该保持完整，不能在函数中间切断
- 图片引用（[IMAGE: id]）应该与其上下文描述绑定
- 每个 chunk 应该携带所属章节的元数据（section hierarchy）

【算法步骤】
1. 解析文档结构：识别标题层级、代码块、表格、图片引用等
2. 按结构元素切分为"结构块"（structural blocks）
3. 对每个结构块标注类型（text/table/code/image）和所属章节
4. 对超长的文本块，使用 Recursive Splitter 进行二次切分
5. 短的相邻文本块合并（不超过 chunk_size）

【适用场景】
- Markdown/HTML 文档（含表格、代码块、图片）
- 技术文档、API 文档
- 图文混排的知识库文档
- 需要保留文档层级结构的场景

【与 Recursive Splitter 的关系】
Document Structure Splitter 是 Recursive 的**上层补充**：
- Layer 1: Document Structure → 按结构元素切分
- Layer 2: Recursive → 对超长文本块进行二次切分
两者配合使用，而非互相替代。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.libs.splitter.base_splitter import BaseSplitter, SplitterError
from src.libs.splitter.splitter_factory import register_splitter

if TYPE_CHECKING:
    from src.core.settings import SplitterSettings


@dataclass
class StructuralBlock:
    """表示一个文档结构块。"""

    content: str
    block_type: str  # "text" | "table" | "code" | "image" | "heading"
    section_hierarchy: list[str] = field(default_factory=list)  # 所属章节层级
    metadata: dict = field(default_factory=dict)


@register_splitter("document_structure")
class DocumentStructureSplitter(BaseSplitter):
    """基于文档结构元素的智能分割器。

    识别 Markdown 文档中的结构化元素（标题、表格、代码块、图片引用），
    将它们作为独立的语义单元进行切分，并为每个 chunk 附加结构元数据。
    """

    # 正则模式
    # 代码块：```...```
    CODE_BLOCK_PATTERN = re.compile(
        r'^```[\w]*\n(.*?)^```',
        re.MULTILINE | re.DOTALL,
    )
    # 表格：至少包含一行 | xxx | xxx | 格式
    TABLE_PATTERN = re.compile(
        r'((?:^\|.+\|$\n?){2,})',
        re.MULTILINE,
    )
    # 图片引用：[IMAGE: id] 或 ![alt](url)
    IMAGE_REF_PATTERN = re.compile(
        r'(\[IMAGE:\s*[^\]]+\]|!\[[^\]]*\]\([^)]+\))',
    )
    # 标题：# ## ### 等
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def __init__(self, settings: "SplitterSettings"):
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap

    def _parse_structural_blocks(self, text: str) -> list[StructuralBlock]:
        """解析文档为结构化块列表。"""
        blocks: list[StructuralBlock] = []
        current_sections: list[str] = []  # 当前章节层级栈

        # 标记特殊块的位置（代码块、表格）
        special_spans: list[tuple[int, int, str, str]] = []  # (start, end, type, content)

        # 找到所有代码块
        for m in self.CODE_BLOCK_PATTERN.finditer(text):
            special_spans.append((m.start(), m.end(), "code", m.group(0)))

        # 找到所有表格
        for m in self.TABLE_PATTERN.finditer(text):
            # 确保不在代码块内
            if not self._in_special_span(m.start(), special_spans):
                special_spans.append((m.start(), m.end(), "table", m.group(0)))

        # 按位置排序
        special_spans.sort(key=lambda x: x[0])

        # 逐段处理
        pos = 0
        for span_start, span_end, block_type, content in special_spans:
            # 处理特殊块之前的普通文本
            if pos < span_start:
                text_before = text[pos:span_start]
                self._parse_text_blocks(text_before, current_sections, blocks)

            # 添加特殊块
            blocks.append(StructuralBlock(
                content=content.strip(),
                block_type=block_type,
                section_hierarchy=list(current_sections),
            ))
            pos = span_end

        # 处理最后一段普通文本
        if pos < len(text):
            remaining = text[pos:]
            self._parse_text_blocks(remaining, current_sections, blocks)

        return blocks

    def _parse_text_blocks(
        self,
        text: str,
        current_sections: list[str],
        blocks: list[StructuralBlock],
    ) -> None:
        """解析普通文本段，识别标题并按段落切分。"""
        # 按行处理，识别标题
        lines = text.split('\n')
        buffer: list[str] = []

        for line in lines:
            heading_match = self.HEADING_PATTERN.match(line)
            if heading_match:
                # 先输出之前积累的文本
                if buffer:
                    content = '\n'.join(buffer).strip()
                    if content:
                        # 检查是否包含图片引用
                        block_type = "image" if self.IMAGE_REF_PATTERN.search(content) else "text"
                        blocks.append(StructuralBlock(
                            content=content,
                            block_type=block_type,
                            section_hierarchy=list(current_sections),
                        ))
                    buffer = []

                # 更新章节层级
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                # 截断到当前层级
                current_sections = current_sections[:level - 1]
                current_sections.append(title)

                # 标题本身作为一个 heading 块
                blocks.append(StructuralBlock(
                    content=line.strip(),
                    block_type="heading",
                    section_hierarchy=list(current_sections),
                ))
            else:
                buffer.append(line)

        # 输出剩余文本
        if buffer:
            content = '\n'.join(buffer).strip()
            if content:
                block_type = "image" if self.IMAGE_REF_PATTERN.search(content) else "text"
                blocks.append(StructuralBlock(
                    content=content,
                    block_type=block_type,
                    section_hierarchy=list(current_sections),
                ))

    @staticmethod
    def _in_special_span(pos: int, spans: list[tuple[int, int, str, str]]) -> bool:
        """检查某个位置是否在已标记的特殊块内。"""
        return any(start <= pos < end for start, end, _, _ in spans)

    def _merge_small_blocks(self, blocks: list[StructuralBlock]) -> list[StructuralBlock]:
        """合并相邻的小文本块（不超过 chunk_size）。

        只合并同类型、同章节的相邻块。
        """
        if not blocks:
            return []

        merged: list[StructuralBlock] = []
        current = blocks[0]

        for block in blocks[1:]:
            # 只合并文本类型、同章节的相邻块
            can_merge = (
                current.block_type == "text"
                and block.block_type == "text"
                and current.section_hierarchy == block.section_hierarchy
                and len(current.content) + len(block.content) + 1 <= self._chunk_size
            )
            if can_merge:
                current = StructuralBlock(
                    content=current.content + "\n" + block.content,
                    block_type="text",
                    section_hierarchy=current.section_hierarchy,
                )
            else:
                merged.append(current)
                current = block

        merged.append(current)
        return merged

    def _split_oversized_block(self, block: StructuralBlock) -> list[str]:
        """对超长块进行二次切分（使用 Recursive 策略）。"""
        if len(block.content) <= self._chunk_size:
            return [block.content]

        # 对于代码块和表格，即使超长也尽量保持完整
        # 但如果实在太长（超过 3 倍 chunk_size），仍需切分
        if block.block_type in ("code", "table") and len(block.content) <= self._chunk_size * 3:
            return [block.content]

        # 使用 Recursive 策略进行二次切分
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                keep_separator=True,
                strip_whitespace=True,
            )
            return splitter.split_text(block.content)
        except ImportError:
            # 降级为简单的滑动窗口切分
            chunks = []
            start = 0
            while start < len(block.content):
                end = start + self._chunk_size
                chunks.append(block.content[start:end].strip())
                start = end - self._chunk_overlap
            return [c for c in chunks if c]

    def split_text(self, text: str) -> list[str]:
        """基于文档结构对文本进行分割。

        Args:
            text: 待分割的文本（Markdown 格式）。

        Returns:
            分割后的文本块列表。

        Raises:
            SplitterError: 分割失败时抛出。
        """
        if not text or not text.strip():
            return []

        try:
            # Step 1: 解析文档结构
            blocks = self._parse_structural_blocks(text)

            # Step 2: 合并相邻的小文本块
            blocks = self._merge_small_blocks(blocks)

            # Step 3: 对超长块进行二次切分，生成最终 chunks
            chunks: list[str] = []
            for block in blocks:
                sub_chunks = self._split_oversized_block(block)
                chunks.extend(sub_chunks)

            return [c for c in chunks if c.strip()]

        except SplitterError:
            raise
        except Exception as e:
            raise SplitterError(
                f"文档结构分割失败: {e}",
                provider="document_structure",
            ) from e

    @property
    def provider_name(self) -> str:
        return "document_structure"

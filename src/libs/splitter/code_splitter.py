"""Code Splitter — 基于代码语法结构的智能分割。

================================================================================
📖 Code Split 原理说明
================================================================================

【核心思想】
Code Splitter 利用编程语言的语法结构（函数、类、方法定义等）作为切分边界，
确保每个 chunk 是一个完整的代码逻辑单元，而非在函数中间被截断。

【为什么需要它？】
- Recursive Splitter 按字符分隔符切分，可能在函数体中间截断
- 代码的语义边界是函数/类/方法定义，而非换行或空行
- 代码检索需要返回完整的函数/类定义，才能被 LLM 正确理解和使用

【算法步骤】
1. 根据文件语言选择对应的语法分隔符（class/def/function 等）
2. 使用 LangChain RecursiveCharacterTextSplitter.from_language() 进行切分
3. 对于无法识别语言的文件，降级为通用 Recursive 策略
4. 保证每个 chunk 是完整的函数/类/代码块

【支持的语言】
Python, JavaScript, TypeScript, Java, Go, Rust, C/C++, C#, Ruby,
PHP, Scala, Swift, Markdown, LaTeX, HTML, Kotlin, Lua, Haskell 等

【适用场景】
- 代码仓库的 RAG 检索
- 代码问答系统
- 代码审查辅助
- 技术文档中嵌入的代码片段

【与 Recursive Splitter 的关系】
Code Splitter 本质上是 Recursive Splitter 的**语言感知特化版本**：
- 使用语言特定的分隔符（如 Python 的 \\nclass, \\ndef）
- 分隔符优先级按代码结构层级排列（类 > 函数 > 控制流 > 行）
- 最终仍然降级到通用分隔符（\\n\\n → \\n → 空格 → 字符）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.splitter.base_splitter import BaseSplitter, SplitterError
from src.libs.splitter.splitter_factory import register_splitter

if TYPE_CHECKING:
    from src.core.settings import SplitterSettings


# 文件扩展名到 LangChain Language 枚举的映射
_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "js",
    ".jsx": "js",
    ".ts": "ts",
    ".tsx": "ts",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".scala": "scala",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".lua": "lua",
    ".hs": "haskell",
    ".ex": "elixir",
    ".exs": "elixir",
    ".ps1": "powershell",
    ".md": "markdown",
    ".markdown": "markdown",
    ".tex": "latex",
    ".html": "html",
    ".htm": "html",
    ".sol": "sol",
    ".proto": "proto",
    ".r": "r",
    ".R": "r",
    ".rst": "rst",
    ".cobol": "cobol",
    ".cob": "cobol",
}


@register_splitter("code")
class CodeSplitter(BaseSplitter):
    """基于代码语法结构的智能分割器。

    根据编程语言的语法结构（函数、类、方法定义等）进行切分，
    确保每个 chunk 是完整的代码逻辑单元。
    """

    def __init__(self, settings: "SplitterSettings"):
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap
        # 可通过 settings 扩展字段指定语言，默认自动检测
        self._language: str | None = getattr(settings, "language", None)

    def _detect_language(self, text: str) -> str | None:
        """根据代码内容启发式检测编程语言。

        Args:
            text: 代码文本。

        Returns:
            检测到的语言名称，或 None（无法识别）。
        """
        # 简单的启发式规则
        indicators = {
            "python": [r'\bdef \w+\(', r'\bclass \w+[:(]', r'\bimport \w+', r'\bfrom \w+ import'],
            "js": [r'\bfunction \w+\(', r'\bconst \w+ =', r'\blet \w+ =', r'\brequire\('],
            "ts": [r'\binterface \w+', r'\btype \w+ =', r'\benum \w+', r': \w+\[\]'],
            "java": [r'\bpublic class \w+', r'\bprivate \w+', r'\bprotected \w+'],
            "go": [r'\bfunc \w+\(', r'\bpackage \w+', r'\btype \w+ struct'],
            "rust": [r'\bfn \w+\(', r'\blet mut \w+', r'\bimpl \w+', r'\bpub fn'],
        }

        import re
        scores: dict[str, int] = {}
        for lang, patterns in indicators.items():
            score = sum(1 for p in patterns if re.search(p, text[:2000]))
            if score > 0:
                scores[lang] = score

        if scores:
            return max(scores, key=scores.get)  # type: ignore[arg-type]
        return None

    def _get_language_enum(self, language: str):
        """将语言字符串转换为 LangChain Language 枚举。"""
        from langchain_text_splitters import Language

        lang_map = {
            "python": Language.PYTHON,
            "js": Language.JS,
            "ts": Language.TS,
            "java": Language.JAVA,
            "go": Language.GO,
            "rust": Language.RUST,
            "c": Language.C,
            "cpp": Language.CPP,
            "csharp": Language.CSHARP,
            "ruby": Language.RUBY,
            "php": Language.PHP,
            "scala": Language.SCALA,
            "swift": Language.SWIFT,
            "kotlin": Language.KOTLIN,
            "lua": Language.LUA,
            "haskell": Language.HASKELL,
            "elixir": Language.ELIXIR,
            "powershell": Language.POWERSHELL,
            "markdown": Language.MARKDOWN,
            "latex": Language.LATEX,
            "html": Language.HTML,
            "sol": Language.SOL,
            "proto": Language.PROTO,
            "r": Language.R,
            "rst": Language.RST,
            "cobol": Language.COBOL,
        }
        return lang_map.get(language)

    def split_text(self, text: str) -> list[str]:
        """基于代码语法结构对文本进行分割。

        Args:
            text: 待分割的代码文本。

        Returns:
            分割后的代码块列表。

        Raises:
            SplitterError: 分割失败时抛出。
        """
        if not text or not text.strip():
            return []

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            # 确定语言
            language = self._language or self._detect_language(text)
            lang_enum = self._get_language_enum(language) if language else None

            if lang_enum:
                # 使用语言感知的分割器
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang_enum,
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap,
                    strip_whitespace=True,
                )
            else:
                # 降级为通用 Recursive 策略（使用代码友好的分隔符）
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap,
                    separators=[
                        "\nclass ",
                        "\ndef ",
                        "\n\ndef ",
                        "\n\n",
                        "\n",
                        " ",
                        "",
                    ],
                    keep_separator=True,
                    strip_whitespace=True,
                )

            chunks = splitter.split_text(text)
            return [c for c in chunks if c.strip()]

        except ImportError:
            raise SplitterError(
                "langchain-text-splitters not installed. Run: pip install langchain-text-splitters",
                provider="code",
            )
        except Exception as e:
            raise SplitterError(f"代码分割失败: {e}", provider="code") from e

    @classmethod
    def for_file(cls, file_path: str, settings: "SplitterSettings") -> "CodeSplitter":
        """根据文件路径创建对应语言的 CodeSplitter。

        Args:
            file_path: 文件路径（用于推断语言）。
            settings: Splitter 配置。

        Returns:
            配置了对应语言的 CodeSplitter 实例。
        """
        from pathlib import Path

        ext = Path(file_path).suffix.lower()
        language = _EXTENSION_TO_LANGUAGE.get(ext)

        instance = cls(settings)
        instance._language = language
        return instance

    @property
    def provider_name(self) -> str:
        return "code"

"""MetadataEnricher — adds title/summary/tags to chunk metadata."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.core.settings import Settings

# 外置 prompt 文件路径
_PROMPT_PATH = "config/prompts/metadata_enrichment.txt"
# system/user 分隔符
_PROMPT_DELIMITER = "---USER---"
# 送入 LLM 的文本最大字符数
_MAX_TEXT_LEN = 1000


class MetadataEnricher(BaseTransform):
    """Enriches chunk metadata with title, summary, and tags."""

    def __init__(self, settings: "Settings"):
        self._use_llm = settings.ingestion.metadata_enricher.use_llm
        self._settings = settings
        self._llm = None  # lazily created once, then reused across all chunks
        self._system_prompt, self._user_template = self._load_prompt()

    def _get_llm(self):
        """Lazily create and cache the LLM client (avoid rebuilding per chunk)."""
        if self._llm is None:
            from src.libs.llm.llm_factory import LLMFactory

            self._llm = LLMFactory.create(self._settings.llm)
        return self._llm

    def _load_prompt(self) -> tuple[str, str]:
        """Load prompt template from file, split into system and user parts."""
        path = Path(_PROMPT_PATH)
        if path.exists():
            content = path.read_text(encoding="utf-8")
            if _PROMPT_DELIMITER in content:
                parts = content.split(_PROMPT_DELIMITER, 1)
                return parts[0].strip(), parts[1].strip()
            # 没有分隔符时，整体作为 user prompt
            return "", content.strip()
        # 内置 fallback
        return (
            "你是一个文档元数据提取助手。",
            "请为以下文档片段提取 title、summary、tags，返回 JSON。\n\n{text}",
        )

    def transform(self, chunks: list[Chunk]) -> list[Chunk]:
        """Enrich chunks with semantic metadata."""
        import time as _time

        enriched = []
        total = len(chunks)
        llm_total_time = 0.0
        llm_success = 0
        llm_fail = 0

        logger.info("MetadataEnricher: starting enrichment for %d chunks (use_llm=%s)", total, self._use_llm)

        for idx, chunk in enumerate(chunks):
            try:
                meta = dict(chunk.metadata)

                if self._use_llm:
                    t0 = _time.perf_counter()
                    llm_meta = self._llm_enrich(chunk.text)
                    elapsed = _time.perf_counter() - t0
                    llm_total_time += elapsed

                    if llm_meta:
                        meta.update(llm_meta)
                        meta["enriched_by"] = "llm"
                        llm_success += 1
                        logger.info(
                            "  [%d/%d] LLM enrich OK (%.2fs) title=%r",
                            idx + 1, total, elapsed,
                            (llm_meta.get("title", "")[:60]),
                        )
                    else:
                        meta.update(self._rule_enrich(chunk.text))
                        meta["enriched_by"] = "rule"
                        llm_fail += 1
                        logger.warning(
                            "  [%d/%d] LLM enrich FAILED (%.2fs), fallback to rule",
                            idx + 1, total, elapsed,
                        )
                else:
                    meta.update(self._rule_enrich(chunk.text))
                    meta["enriched_by"] = "rule"

                enriched.append(Chunk(id=chunk.id, text=chunk.text, metadata=meta, source_ref=chunk.source_ref))
            except Exception as e:
                logger.error("  [%d/%d] chunk enrich exception: %s", idx + 1, total, e)
                chunk.metadata["enriched_by"] = "none"
                enriched.append(chunk)

        if self._use_llm:
            logger.info(
                "MetadataEnricher: done. LLM calls=%d success=%d fail=%d total_time=%.1fs avg=%.2fs/chunk",
                total, llm_success, llm_fail, llm_total_time,
                llm_total_time / max(total, 1),
            )

        return enriched

    def _rule_enrich(self, text: str) -> dict:
        """Rule-based metadata extraction."""
        lines = text.strip().split('\n')
        # Title: first non-empty line (strip markdown headers)
        title = ""
        for line in lines:
            stripped = line.strip().lstrip('#').strip()
            if stripped:
                title = stripped[:100]
                break

        # Summary: first 200 chars
        summary = text[:200].replace('\n', ' ').strip()

        # Tags: extract words that look like keywords (capitalized, technical terms)
        words = text.split()
        tags = list(set(w.strip('.,;:()[]') for w in words if len(w) > 3 and w[0].isupper()))[:5]

        return {"title": title, "summary": summary, "tags": tags}

    def _llm_enrich(self, text: str) -> dict | None:
        """LLM-based metadata extraction with structured prompt and robust JSON parsing."""
        try:
            from src.libs.llm.base_llm import ChatMessage

            llm = self._get_llm()

            # 构建 messages：system + user
            user_content = self._user_template.replace("{text}", text[:_MAX_TEXT_LEN])
            messages = []
            if self._system_prompt:
                messages.append(ChatMessage(role="system", content=self._system_prompt))
            messages.append(ChatMessage(role="user", content=user_content))

            logger.debug("  -> LLM request: model=%s, user_content_len=%d", getattr(llm, '_model', '?'), len(user_content))
            response = llm.chat(messages)
            logger.debug(
                "  <- LLM response: content_len=%d, usage=%s, raw_content=%r",
                len(response.content or ""),
                response.usage,
                (response.content or "")[:200],
            )

            data = self._parse_json(response.content)
            if data is None:
                logger.warning("  JSON parse failed for response: %r", (response.content or "")[:300])
                return None

            return {
                "title": str(data.get("title", "")).strip(),
                "summary": str(data.get("summary", "")).strip(),
                "tags": data.get("tags", []) if isinstance(data.get("tags"), list) else [],
            }
        except Exception as e:
            logger.warning("LLM metadata enrichment failed (fallback to rule): %s", e)
            return None

    @staticmethod
    def _parse_json(text: str | None) -> dict | None:
        """Robustly extract JSON from LLM response (handles markdown fences, extra text)."""
        if not text:
            return None
        text = text.strip()
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 尝试提取 ```json ... ``` 代码块
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        # 尝试提取第一个 { ... } 块
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

    @property
    def name(self) -> str:
        return "metadata_enricher"

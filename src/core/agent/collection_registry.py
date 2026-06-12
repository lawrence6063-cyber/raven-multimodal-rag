"""CollectionRegistry — enumerate available collections for the router.

The router (§3.1) needs the set of collections it may route a query to. The
vector store has no native collection enumeration (a single Chroma collection
plus a ``collection`` metadata filter is used), so we reuse the directory-scan
logic of ``ListCollectionsTool`` to derive the list of collection names.

Kept deliberately lightweight; see P1_AGENTIC_RAG_SPEC §5.6 for the optional
future evolution (native ``list_collections`` on the vector store).
"""

from __future__ import annotations

from src.mcp_server.tools.list_collections import ListCollectionsTool
from src.observability.logger import get_logger

logger = get_logger("core.agent.collection_registry")

# _DEFAULT_DOCUMENTS_DIR default documents root scanned for collections
_DEFAULT_DOCUMENTS_DIR = "data/documents"


class CollectionRegistry:
    """Provides the list of available collection names to the router."""

    def __init__(self, documents_dir: str = _DEFAULT_DOCUMENTS_DIR):
        self._lister = ListCollectionsTool(documents_dir)

    def list_collections(self) -> list[str]:
        """Return available collection names (empty list on any failure).

        Returns:
            Sorted collection names. Never raises — failures degrade to an empty
            list so the router falls back to searching all collections.
        """
        try:
            result = self._lister.run()
            structured = result.structured_content or {}
            collections = structured.get("collections", [])
            return [entry["name"] for entry in collections if entry.get("name")]
        except Exception as e:  # never block routing on enumeration failure
            logger.warning(f"Collection enumeration failed, treating as empty: {e}")
            return []

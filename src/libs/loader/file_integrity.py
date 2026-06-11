"""File integrity checker — SHA256-based deduplication for ingestion.

Uses SQLite to persist file processing history, supporting incremental ingestion.
"""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime


class FileIntegrityChecker(ABC):
    """Abstract interface for file integrity checking."""

    @abstractmethod
    def compute_sha256(self, path: str | Path) -> str:
        """Compute SHA256 hash of a file."""

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """Check if a file with this hash was already successfully processed."""

    @abstractmethod
    def mark_success(self, file_hash: str, file_path: str, chunk_count: int = 0) -> None:
        """Mark a file as successfully processed."""

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """Mark a file as failed during processing."""


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """SQLite-backed file integrity checker.

    Stores processing history in data/db/ingestion_history.db (configurable).
    Uses WAL mode for concurrent safety.
    """

    def __init__(self, db_path: str | Path = "data/db/ingestion_history.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'processing')),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_msg TEXT,
                    chunk_count INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_processed_at ON ingestion_history(processed_at)")

    def _connect(self) -> sqlite3.Connection:
        """Create a connection with WAL mode."""
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def compute_sha256(self, path: str | Path) -> str:
        """Compute SHA256 hash of a file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        """Check if file was already successfully processed."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ? AND status = 'success'",
                (file_hash,),
            ).fetchone()
            return row is not None

    def mark_success(self, file_hash: str, file_path: str, chunk_count: int = 0) -> None:
        """Mark file as successfully processed."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ingestion_history
                   (file_hash, file_path, file_size, status, processed_at, chunk_count)
                   VALUES (?, ?, ?, 'success', ?, ?)""",
                (file_hash, file_path, Path(file_path).stat().st_size if Path(file_path).exists() else 0,
                 datetime.now().isoformat(), chunk_count),
            )

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """Mark file as failed."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ingestion_history
                   (file_hash, file_path, status, processed_at, error_msg)
                   VALUES (?, '', 'failed', ?, ?)""",
                (file_hash, datetime.now().isoformat(), error_msg),
            )

    def list_processed(self) -> list[dict]:
        """List all successfully processed files."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT file_hash, file_path, processed_at, chunk_count FROM ingestion_history WHERE status = 'success'"
            ).fetchall()
            return [{"file_hash": r[0], "file_path": r[1], "processed_at": r[2], "chunk_count": r[3]} for r in rows]

    def remove_record(self, file_hash: str) -> None:
        """Remove a record from history (for re-processing)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM ingestion_history WHERE file_hash = ?", (file_hash,))

"""ImageStorage — stores images and maintains SQLite index mapping."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from datetime import datetime


class ImageStorage:
    """Manages image file storage and SQLite-based index lookup."""

    def __init__(self, base_dir: str = "data/images", db_path: str = "data/db/image_index.db"):
        self._base_dir = Path(base_dir)
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the image index database."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def save(self, image_id: str, source_path: str, collection: str = "default",
             doc_hash: str = "", page_num: int = 0) -> str:
        """Save an image file and record in index.

        Args:
            image_id: Unique image identifier.
            source_path: Path to the source image file.
            collection: Collection name for organization.
            doc_hash: Document hash for association.
            page_num: Page number in source document.

        Returns:
            Destination file path.
        """
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {source_path}")

        dest_dir = self._base_dir / collection
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{image_id}{src.suffix}"

        shutil.copy2(str(src), str(dest))

        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO image_index
                   (image_id, file_path, collection, doc_hash, page_num, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (image_id, str(dest), collection, doc_hash, page_num, datetime.now().isoformat()),
            )

        return str(dest)

    def get_path(self, image_id: str) -> str | None:
        """Look up image file path by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?", (image_id,)
            ).fetchone()
            return row[0] if row else None

    def list_images(self, collection: str | None = None) -> list[dict]:
        """List all images, optionally filtered by collection."""
        with self._connect() as conn:
            if collection:
                rows = conn.execute(
                    "SELECT image_id, file_path, collection, doc_hash, page_num FROM image_index WHERE collection = ?",
                    (collection,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT image_id, file_path, collection, doc_hash, page_num FROM image_index"
                ).fetchall()

        return [{"image_id": r[0], "file_path": r[1], "collection": r[2], "doc_hash": r[3], "page_num": r[4]} for r in rows]

    def delete(self, image_id: str) -> None:
        """Delete an image file and its index entry."""
        path = self.get_path(image_id)
        if path and Path(path).exists():
            Path(path).unlink()
        with self._connect() as conn:
            conn.execute("DELETE FROM image_index WHERE image_id = ?", (image_id,))

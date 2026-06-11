"""Tests for file integrity checker (SHA256 + SQLite)."""

import pytest
from pathlib import Path

from src.libs.loader.file_integrity import SQLiteIntegrityChecker


class TestSQLiteIntegrityChecker:
    """Test SQLiteIntegrityChecker."""

    def test_compute_sha256_consistent(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        checker = SQLiteIntegrityChecker(db_path=tmp_path / "test.db")
        h1 = checker.compute_sha256(f)
        h2 = checker.compute_sha256(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex length

    def test_compute_sha256_different_content(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        checker = SQLiteIntegrityChecker(db_path=tmp_path / "test.db")
        assert checker.compute_sha256(f1) != checker.compute_sha256(f2)

    def test_compute_sha256_file_not_found(self, tmp_path):
        checker = SQLiteIntegrityChecker(db_path=tmp_path / "test.db")
        with pytest.raises(FileNotFoundError):
            checker.compute_sha256("/nonexistent/file.txt")

    def test_should_skip_not_processed(self, tmp_path):
        checker = SQLiteIntegrityChecker(db_path=tmp_path / "test.db")
        assert checker.should_skip("abc123") is False

    def test_mark_success_then_skip(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake pdf content")
        checker = SQLiteIntegrityChecker(db_path=tmp_path / "test.db")
        file_hash = checker.compute_sha256(f)

        assert checker.should_skip(file_hash) is False
        checker.mark_success(file_hash, str(f), chunk_count=5)
        assert checker.should_skip(file_hash) is True

    def test_mark_failed_does_not_skip(self, tmp_path):
        checker = SQLiteIntegrityChecker(db_path=tmp_path / "test.db")
        checker.mark_failed("hash_fail", "some error")
        assert checker.should_skip("hash_fail") is False

    def test_list_processed(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content")
        checker = SQLiteIntegrityChecker(db_path=tmp_path / "test.db")
        h = checker.compute_sha256(f)
        checker.mark_success(h, str(f), chunk_count=3)
        records = checker.list_processed()
        assert len(records) == 1
        assert records[0]["chunk_count"] == 3

    def test_remove_record(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content")
        checker = SQLiteIntegrityChecker(db_path=tmp_path / "test.db")
        h = checker.compute_sha256(f)
        checker.mark_success(h, str(f))
        assert checker.should_skip(h) is True
        checker.remove_record(h)
        assert checker.should_skip(h) is False

    def test_db_file_created(self, tmp_path):
        db_path = tmp_path / "sub" / "dir" / "history.db"
        SQLiteIntegrityChecker(db_path=db_path)
        assert db_path.exists()

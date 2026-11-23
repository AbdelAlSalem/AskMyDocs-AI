import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

DocumentStatus = Literal["processing", "ready", "deleting"]


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    filename: str
    page_count: int
    chunk_count: int
    status: DocumentStatus
    uploaded_at: datetime
    size_bytes: int


class DocumentCatalog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('processing', 'ready', 'deleting')),
                    uploaded_at TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE
                )
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            yield connection

    def has_digest(self, digest: str) -> bool:
        with self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM documents WHERE sha256 = ?", (digest,)
            ).fetchone() is not None

    def insert(self, record: DocumentRecord, digest: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents
                    (id, filename, page_count, chunk_count, status, uploaded_at, size_bytes, sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id, record.filename, record.page_count, record.chunk_count,
                    record.status, record.uploaded_at.isoformat(), record.size_bytes, digest,
                ),
            )

    def list(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY uploaded_at DESC, id")
            return [
                DocumentRecord(
                    id=row["id"],
                    filename=row["filename"],
                    page_count=row["page_count"],
                    chunk_count=row["chunk_count"],
                    status=row["status"],
                    uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
                    size_bytes=row["size_bytes"],
                )
                for row in rows
            ]

    def set_status(self, document_id: str, status: DocumentStatus) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE documents SET status = ? WHERE id = ?", (status, document_id)
            )
            return cursor.rowcount > 0

    def remove(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

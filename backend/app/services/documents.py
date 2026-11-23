import hashlib
import logging
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import BinaryIO
from uuid import UUID, uuid4

from app.rag.chunking import split_pages
from app.rag.document_wide import DocumentCoverage, DocumentIntent, SummaryPageScope, document_coverage
from app.rag.embeddings import EmbeddingError
from app.rag.vector_store import RetrievedChunk, VectorModelMismatch, VectorStore, VectorStoreError
from app.services.catalog import DocumentCatalog, DocumentRecord
from app.services.errors import DocumentError
from app.services.pdf import extract_pages, read_pdf_upload

logger = logging.getLogger(__name__)
STORAGE_ERRORS = (OSError, sqlite3.Error, VectorStoreError, EmbeddingError)


def _unavailable(error: Exception) -> DocumentError:
    if isinstance(error, EmbeddingError):
        return DocumentError(
            503,
            "embedding_unavailable",
            "The local embedding model is unavailable. Check the model setup and first-download connection.",
        )
    if isinstance(error, VectorModelMismatch):
        return DocumentError(
            503,
            "embedding_model_mismatch",
            "The saved index uses a different embedding model. Restore that model or use a fresh data directory.",
        )
    return DocumentError(503, "storage_unavailable", "Local document storage is temporarily unavailable.")


class DocumentService:
    def __init__(
        self,
        *,
        upload_dir: Path,
        sqlite_path: Path,
        vectors: VectorStore,
        max_upload_bytes: int = 10 * 1024 * 1024,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        self._upload_dir = upload_dir
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        self._catalog = DocumentCatalog(sqlite_path)
        self._vectors = vectors
        self._max_upload_bytes = max_upload_bytes
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        # The local server has one worker. Serialize vector mutations and reads within it.
        self._lock = Lock()
        for record in self._catalog.list():
            if record.status != "ready":
                self._try_purge(record.id)

    def _file_path(self, document_id: str) -> Path:
        return self._upload_dir / f"{UUID(document_id)}.pdf"

    def _purge(self, document_id: str) -> None:
        self._vectors.delete(document_id)
        self._file_path(document_id).unlink(missing_ok=True)
        self._catalog.remove(document_id)

    def _try_purge(self, document_id: str) -> None:
        try:
            self._purge(document_id)
        except STORAGE_ERRORS as exc:
            logger.warning("Document cleanup can be retried (%s).", type(exc).__name__)

    def ingest(self, filename: str, file: BinaryIO) -> DocumentRecord:
        try:
            filename, contents = read_pdf_upload(filename, file, self._max_upload_bytes)
            digest = hashlib.sha256(contents).hexdigest()
            with self._lock:
                return self._ingest(filename, contents, digest)
        except STORAGE_ERRORS as exc:
            raise _unavailable(exc) from exc

    def _ingest(self, filename: str, contents: bytes, digest: str) -> DocumentRecord:
        if self._catalog.has_digest(digest):
            raise DocumentError(409, "duplicate_document", "A PDF with the same contents already exists.")

        document_id = str(uuid4())
        pages = extract_pages(contents)
        chunks = split_pages(
            pages, document_id=document_id, filename=filename,
            chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap,
            token_budget=self._vectors.chunk_token_budget(),
        )
        record = DocumentRecord(
            id=document_id, filename=filename, page_count=len(pages), chunk_count=len(chunks),
            status="processing", uploaded_at=datetime.now(UTC), size_bytes=len(contents),
        )

        # Record intent before changing the file and index so interrupted writes can be cleaned up.
        self._catalog.insert(record, digest)
        try:
            with self._file_path(document_id).open("xb") as destination:
                destination.write(contents)
            self._vectors.add(chunks)
            self._catalog.set_status(document_id, "ready")
        except STORAGE_ERRORS:
            self._try_purge(document_id)
            raise
        return replace(record, status="ready")

    def list_documents(self) -> list[DocumentRecord]:
        try:
            return self._catalog.list()
        except STORAGE_ERRORS as exc:
            raise _unavailable(exc) from exc

    def delete(self, document_id: str) -> None:
        try:
            with self._lock:
                if not self._catalog.set_status(document_id, "deleting"):
                    raise DocumentError(404, "document_not_found", "This document could not be found.")
                self._purge(document_id)
        except STORAGE_ERRORS as exc:
            raise _unavailable(exc) from exc

    def retrieve(
        self, question: str, document_ids: list[str] | None, *, top_k: int, min_similarity: float
    ) -> list[RetrievedChunk]:
        try:
            with self._lock:
                ready_ids = [record.id for record in self._catalog.list() if record.status == "ready"]
                if document_ids is not None and set(document_ids).difference(ready_ids):
                    raise DocumentError(404, "invalid_document_scope", "A selected document is no longer available.")
                return self._vectors.retrieve(
                    question,
                    document_ids=ready_ids if document_ids is None else document_ids,
                    top_k=top_k,
                    min_similarity=min_similarity,
                )
        except STORAGE_ERRORS as exc:
            raise _unavailable(exc) from exc

    def retrieve_document_wide(
        self, document_ids: list[str] | None, *, intent: DocumentIntent, max_context_chars: int,
        page_scope: SummaryPageScope | None = None,
    ) -> DocumentCoverage:
        try:
            with self._lock:
                ready = {record.id: record.page_count for record in self._catalog.list() if record.status == "ready"}
                if document_ids is not None and set(document_ids).difference(ready):
                    raise DocumentError(404, "invalid_document_scope", "A selected document is no longer available.")
                scope = ready if document_ids is None else {doc_id: ready[doc_id] for doc_id in document_ids}
                return document_coverage(
                    self._vectors.passages(list(scope)), scope, intent, max_context_chars, page_scope=page_scope,
                )
        except STORAGE_ERRORS as exc:
            raise _unavailable(exc) from exc

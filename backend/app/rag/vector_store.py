from collections.abc import Iterator, Sequence
from heapq import nlargest
from itertools import batched
from math import isfinite
from pathlib import Path
from threading import Lock

import chromadb
from chromadb import Where
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from app.rag.chunking import Chunk, ChunkTokenBudget
from app.rag.debug import RetrievalCandidate, log_retrieval
from app.rag.embeddings import EmbeddingError, LocalEmbeddings
from app.rag.lexical import LexicalQuery, comparison_evidence, comparison_queries, retrieval_question
from app.rag.types import RetrievedChunk

COLLECTION_NAME = "askmydocs_chunks"
LEXICAL_BATCH_SIZE = 256
LEXICAL_RANK_WEIGHT = 0.8


class VectorStoreError(RuntimeError):
    """The local vector index could not complete an operation."""


class VectorModelMismatch(VectorStoreError):
    """The existing collection was created with a different embedding model."""


def _lexical_candidates(
    store: Chroma, query: LexicalQuery, scope: Where, top_k: int
) -> dict[str, float]:
    if len(query.terms) < 2:
        return {}

    def scored_chunks() -> Iterator[tuple[str, float]]:
        offset = 0
        text_filter = query.document_filter()
        while True:
            result = store.get(
                where=scope,
                where_document=text_filter,
                include=["documents"],
                limit=LEXICAL_BATCH_SIZE,
                offset=offset,
            )
            for chunk_id, text in zip(result["ids"], result["documents"], strict=True):
                score = query.score(text)
                if score > 0:
                    yield chunk_id, score
            if len(result["ids"]) < LEXICAL_BATCH_SIZE:
                return
            offset += LEXICAL_BATCH_SIZE

    return dict(nlargest(top_k, scored_chunks(), key=lambda candidate: candidate[1]))


def _cosine_similarity(distance: float) -> float:
    return max(-1.0, min(1.0, 1.0 - distance))


def _retrieval_candidates(
    matches: list[tuple[Document, float]],
    lexical: dict[str, float],
    ranked: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievalCandidate]:
    ranks = {match.chunk_id: rank for rank, match in enumerate(ranked, start=1)}
    candidates = []
    for document, distance in matches:
        metadata = document.metadata
        chunk_id = metadata["chunk_id"]
        rank = ranks.get(chunk_id)
        valid_distance = isfinite(distance)
        if not valid_distance:
            disposition = "invalid_distance"
        elif rank is None:
            disposition = "below_threshold"
        elif rank > top_k:
            disposition = "dropped_by_top_k"
        else:
            disposition = "retained"
        candidates.append(
            RetrievalCandidate(
                document_id=metadata["document_id"],
                filename=metadata["filename"],
                page_number=metadata["page_number"],
                chunk_id=chunk_id,
                cosine_distance=float(distance) if valid_distance else None,
                cosine_similarity=_cosine_similarity(distance) if valid_distance else None,
                lexical_score=lexical.get(chunk_id, 0.0),
                rank=rank,
                disposition=disposition,
            )
        )
    return candidates


class VectorStore:
    def __init__(
        self,
        *,
        directory: Path,
        model_name: str,
        embeddings: Embeddings | None = None,
        debug_api_key: SecretStr | None = None,
    ) -> None:
        self._directory = directory
        self._model_name = model_name
        self._embeddings = embeddings if embeddings is not None else LocalEmbeddings(model_name)
        # A SecretStr, even with an empty value, enables opt-in tracing and redaction.
        self._debug_api_key = debug_api_key
        self._store: Chroma | None = None
        self._indexed_model: str | None = None
        self._lock = Lock()

    def chunk_token_budget(self) -> ChunkTokenBudget | None:
        if isinstance(self._embeddings, LocalEmbeddings):
            return self._embeddings.chunk_token_budget()
        return None

    def _get_store(self, *, verify_model: bool = True) -> Chroma:
        with self._lock:
            if self._store is None:
                client = chromadb.PersistentClient(
                    path=str(self._directory), settings=Settings(anonymized_telemetry=False)
                )
                store = Chroma(
                    client=client,
                    collection_name=COLLECTION_NAME,
                    embedding_function=self._embeddings,
                    collection_metadata={"embedding_model": self._model_name},
                    collection_configuration={"hnsw": {"space": "cosine"}},
                )
                collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
                self._indexed_model = (collection.metadata or {}).get("embedding_model")
                self._store = store
            if verify_model and self._indexed_model != self._model_name:
                raise VectorModelMismatch
            return self._store

    def add(self, chunks: Sequence[Chunk]) -> None:
        try:
            store = self._get_store()
            # Bound the batch size for local inference and Chroma's per-write limits.
            for batch in batched(chunks, 128):
                documents = [
                    Document(
                        id=chunk.chunk_id,
                        page_content=chunk.text,
                        metadata={
                            "document_id": chunk.document_id,
                            "filename": chunk.filename,
                            "page_number": chunk.page_number,
                            "chunk_id": chunk.chunk_id,
                        },
                    )
                    for chunk in batch
                ]
                store.add_documents(documents, ids=[chunk.chunk_id for chunk in batch])
        except (EmbeddingError, VectorModelMismatch):
            raise
        except Exception as exc:
            raise VectorStoreError from exc

    def retrieve(
        self, question: str, *, document_ids: list[str], top_k: int, min_similarity: float
    ) -> list[RetrievedChunk]:
        search_text = retrieval_question(question)
        if not document_ids:
            if self._debug_api_key is not None:
                log_retrieval(
                    search_text, document_ids, [], top_k=top_k, min_similarity=min_similarity,
                    api_key=self._debug_api_key.get_secret_value(),
                )
            return []
        try:
            scope: Where = {"document_id": {"$in": document_ids}}
            store = self._get_store()
            original = self._search(store, search_text, document_ids, scope, top_k, min_similarity)
            parts = comparison_queries(search_text) if top_k >= 2 else ()
            if not parts:
                return original
            focused = [self._search(store, part, document_ids, scope, top_k, min_similarity) for part in parts]
            return comparison_evidence(original, focused, top_k)
        except (EmbeddingError, VectorModelMismatch):
            raise
        except Exception as exc:
            raise VectorStoreError from exc

    def _search(
        self, store: Chroma, search_text: str, document_ids: list[str], scope: Where,
        top_k: int, min_similarity: float,
    ) -> list[RetrievedChunk]:
        query = LexicalQuery.from_text(search_text)
        lexical = _lexical_candidates(store, query, scope, top_k)
        embedding = self._embeddings.embed_query(search_text)
        matches = store.similarity_search_by_vector_with_relevance_scores(
            embedding, k=top_k, filter=scope,
        )
        semantic_ids = {document.metadata["chunk_id"] for document, _ in matches}
        lexical_only = sorted(lexical.keys() - semantic_ids)
        if lexical_only:
            matches.extend(store.similarity_search_by_vector_with_relevance_scores(
                embedding, k=len(lexical_only), filter=scope, ids=lexical_only,
            ))
        results = []
        for document, distance in matches:
            if not isfinite(distance):
                continue
            similarity = _cosine_similarity(distance)
            metadata = document.metadata
            lexical_score = lexical.get(metadata["chunk_id"], 0.0)
            if similarity >= min_similarity or lexical_score > 0:
                results.append(RetrievedChunk(
                    document_id=metadata["document_id"], filename=metadata["filename"],
                    page_number=metadata["page_number"], chunk_id=metadata["chunk_id"],
                    text=document.page_content, similarity=similarity, lexical_score=lexical_score,
                ))
        results.sort(
            key=lambda match: max(0.0, match.similarity) + LEXICAL_RANK_WEIGHT * match.lexical_score,
            reverse=True,
        )
        if self._debug_api_key is not None:
            log_retrieval(
                search_text, document_ids, _retrieval_candidates(matches, lexical, results, top_k),
                top_k=top_k, min_similarity=min_similarity,
                api_key=self._debug_api_key.get_secret_value(),
            )
        return results[:top_k]

    def delete(self, document_id: str) -> None:
        try:
            self._get_store(verify_model=False).delete(where={"document_id": document_id})
        except Exception as exc:
            raise VectorStoreError from exc

    def passages(self, document_ids: list[str]) -> Iterator[RetrievedChunk]:
        if not document_ids:
            return
        try:
            store = self._get_store(verify_model=False)
            scope: Where = {"document_id": {"$in": document_ids}}
            offset = 0
            while True:
                batch = store.get(
                    where=scope, include=["documents", "metadatas"], limit=LEXICAL_BATCH_SIZE, offset=offset,
                )
                for text, metadata in zip(batch["documents"], batch["metadatas"], strict=True):
                    yield RetrievedChunk(
                        document_id=metadata["document_id"],
                        filename=metadata["filename"],
                        page_number=metadata["page_number"],
                        chunk_id=metadata["chunk_id"],
                        text=text,
                        similarity=0.0,
                    )
                if len(batch["ids"]) < LEXICAL_BATCH_SIZE:
                    return
                offset += len(batch["ids"])
        except Exception as exc:
            raise VectorStoreError from exc

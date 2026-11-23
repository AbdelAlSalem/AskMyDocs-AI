from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

from app.rag.chunking import ChunkTokenBudget

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingError(RuntimeError):
    """The local embedding model could not be loaded or used."""


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._lock = Lock()

    def _load_model(self) -> SentenceTransformer:
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.model_name, device="cpu", trust_remote_code=False,
                )
            return self._model

    def chunk_token_budget(self) -> ChunkTokenBudget:
        try:
            model = self._load_model()
            tokenizer = model.tokenizer
            maximum = model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
            if maximum <= 0:
                raise ValueError("The embedding encoder has no available tokens for document text.")
        except Exception as exc:
            raise EmbeddingError from exc

        def count(text: str) -> int:
            try:
                return len(tokenizer.encode(
                    _embedding_input(text), add_special_tokens=False, truncation=False, verbose=False,
                ))
            except Exception as exc:
                raise EmbeddingError from exc

        return ChunkTokenBudget(count, maximum)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._load_model().encode(
                [_embedding_input(text) for text in texts],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
        except Exception as exc:
            raise EmbeddingError from exc

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _embedding_input(text: str) -> str:
    return text.replace("\n", " ")

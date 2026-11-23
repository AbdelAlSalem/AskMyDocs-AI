import json
import logging
import re
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from app.rag.types import RetrievedChunk


logger = logging.getLogger("uvicorn.error.askmydocs.rag")
_trace: ContextVar[dict | None] = ContextVar("rag_request_trace", default=None)
LABELED_SECRET = re.compile(
    r"""(?im)(\b(?:\w+[-_])?(?:api[\s_-]*key|(?:access|auth)[\s_-]*token|token|secret|password|authorization)\b["']?\s*[:=]\s*)[^\r\n]+"""
)
CREDENTIAL = re.compile(
    r"(?i)\b(?:bearer\s+[\w./+=-]+|(?:gsk_|hf_|sk-)[\w-]{12,})|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    cosine_distance: float | None
    cosine_similarity: float | None
    lexical_score: float
    rank: int | None
    disposition: str


def _redact(text: str, api_key: str) -> str:
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    text = LABELED_SECRET.sub(r"\1[REDACTED]", text)
    return CREDENTIAL.sub("[REDACTED]", text)


def _safe_fields(value, key: str):
    if isinstance(value, str):
        return _redact(value, key)
    if isinstance(value, (list, tuple)):
        return [_safe_fields(item, key) for item in value]
    if isinstance(value, dict):
        return {name: _safe_fields(item, key) for name, item in value.items()}
    return value


@contextmanager
def request_trace(enabled: bool, document_ids: Sequence[str] | None, key: str):
    if not enabled:
        yield
        return
    fields = {
        "request_id": str(uuid4()), "mode": "local", "selected_document_ids": document_ids,
        "provider_requested": False, "upstream_status": None, "finish_reason": None,
        "generation_completed": False, "schema_validation": "not_run", "verification": "not_run",
    }
    token = _trace.set(fields)
    try:
        yield
    except Exception as error:
        fields.setdefault("outcome", "request_failed")
        fields["error_type"] = type(error).__name__
        raise
    finally:
        _trace.reset(token)
        logger.info("RAG request trace: %s", json.dumps(_safe_fields(fields, key), ensure_ascii=False))


def trace_fields(**fields) -> None:
    current = _trace.get()
    if current is not None:
        current.update(fields)


def trace_wire_prompt(user_prompt: str, key: str) -> None:
    current = _trace.get()
    if current is not None:
        # The wire format for summaries contains numbered spans. Keep conversation
        # history and HTTP headers out of the local developer's passage inspection.
        evidence = {
            name: value for name, value in json.loads(user_prompt).items()
            if name not in {"conversation_history", "question"}
        }
        logger.info("RAG provider input [%s]: %s", current["request_id"],
                    json.dumps(_safe_fields(evidence, key), ensure_ascii=False))


def log_retrieval(
    effective_query: str,
    document_ids: Sequence[str],
    candidates: Sequence[RetrievalCandidate],
    *,
    top_k: int,
    min_similarity: float,
    api_key: str,
) -> None:
    entries = []
    for candidate in candidates:
        entry = asdict(candidate)
        for key in ("document_id", "filename", "chunk_id"):
            entry[key] = _redact(entry[key], api_key)
        entries.append(entry)
    payload = {
        "effective_query": _redact(effective_query, api_key),
        "document_ids": [_redact(document_id, api_key) for document_id in document_ids],
        "top_k": top_k,
        "min_cosine_similarity": min_similarity,
        "candidate_pool": "merged semantic and lexical shortlists",
        "candidates": entries,
    }
    current = _trace.get()
    if current is not None:
        payload["request_id"] = current["request_id"]
        current.setdefault("retrieval_queries", []).append({
            "query": payload["effective_query"], "threshold": min_similarity, "top_k": top_k,
            "candidates": entries,
        })
        if len(current["retrieval_queries"]) > 1:
            current["strategy"] = "hybrid_multi_query"
    logger.info("RAG retrieval (credentials redacted): %s", json.dumps(payload, ensure_ascii=False))


def log_context(prompt: str, chunks: Sequence["RetrievedChunk"], api_key: str) -> None:
    passages = json.loads(prompt)["retrieved_passages"]
    current = _trace.get()
    ranked = current is None or current.get("strategy") != "document_wide"
    for passage, chunk in zip(passages, chunks, strict=True):
        passage["document_id"] = chunk.document_id
        passage["chunk_id"] = chunk.chunk_id
        passage["cosine_similarity"] = round(chunk.similarity, 4) if ranked else None
        passage["lexical_score"] = round(chunk.lexical_score, 4) if ranked else None
        passage["filename"] = _redact(passage["filename"], api_key)
        passage["text"] = _redact(passage["text"], api_key)
    if current is not None:
        current["context"] = [
            {k: v for k, v in passage.items() if k not in ("filename", "text")}
            | {"characters": len(chunk.text)}
            for passage, chunk in zip(passages, chunks, strict=True)
        ]
    logger.info("RAG context prepared for Groq [%s] (credentials redacted): %s",
                current["request_id"] if current else "local", json.dumps(_safe_fields(passages, api_key), ensure_ascii=False))

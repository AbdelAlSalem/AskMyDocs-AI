import json
import logging
from uuid import UUID
from unittest.mock import Mock

import groq
import httpx
import pytest

from app.core.config import DemoSettings, Settings
from app.models.schemas import ChatRequest
from app.rag.answer import AnswerService, Retriever
from app.rag.types import RetrievedChunk
from app.services.groq_client import GroqGenerator, ProviderError


@pytest.mark.parametrize(("case", "outcome", "schema", "verification"), [
    ("empty", "retrieval_no_evidence", "not_run", "not_run"),
    ("abstain", "evidence_incomplete", "passed", "not_run"),
    ("429", "provider_request_failed", "not_run", "not_run"),
    ("length", "provider_truncated", "not_run", "not_run"),
    ("bad_json", "structured_output_invalid", "failed", "not_run"),
    ("bad_quote", "verification_rejected", "passed", "failed"),
    ("ok", "generated", "passed", "passed"),
])
def test_development_trace_tracks_actual_wire_and_decision(
    case, outcome, schema, verification, monkeypatch, caplog,
):
    key = "test-credential-never-print"
    config = Settings(rag_debug_context=True, groq_api_key=key)
    docs = Mock(spec=Retriever)
    fact = "The launch venue is the library."
    docs.retrieve.return_value = [] if case == "empty" else [
        RetrievedChunk("selected", "sample", 3, "selected:3:1", fact + "\nsecret=" + key, 0.8),
    ]
    sdk = groq.Groq

    def upstream(request):
        assert request.headers["authorization"] == f"Bearer {key}"
        if case == "429":
            return httpx.Response(429, json={"error": {"message": key}})
        claim = {"text": fact, "evidence": [{"source_id": 1, "quote": "Invented" if case == "bad_quote" else fact}]}
        body = ({"status": "insufficient_context", "claims": []} if case == "abstain"
                else {"status": "answered", "claims": [claim]})
        return httpx.Response(200, json={
            "id": "test", "model": config.groq_model, "object": "chat.completion", "created": 0,
            "choices": [{"index": 0, "finish_reason": "length" if case == "length" else "stop",
                         "message": {"role": "assistant", "content": key if case == "bad_json" else json.dumps(body)}}],
        })

    monkeypatch.setattr(groq, "Groq", lambda **kwargs: sdk(**kwargs, http_client=httpx.Client(
        transport=httpx.MockTransport(upstream),
    )))
    caplog.set_level(logging.INFO, logger="uvicorn.error.askmydocs.rag")
    answers = AnswerService(docs, GroqGenerator(config), config)
    request = ChatRequest(question="Where is the launch?", document_ids=["selected"])
    if case in ("429", "bad_json", "length"):
        with pytest.raises(ProviderError):
            answers.answer(request)
    else:
        result = answers.answer(request)
        assert result.status == ("answered" if case == "ok" else "insufficient_context")
    trace, = [json.loads(rec.args[0]) for rec in caplog.records if rec.msg.startswith("RAG request trace")]
    assert UUID(trace["request_id"]).version == 4
    assert (trace["mode"], trace["selected_document_ids"], trace["history_messages"]) == ("local", ["selected"], 0)
    assert (trace["outcome"], trace["schema_validation"], trace["verification"]) == (outcome, schema, verification)
    assert trace["provider_requested"] is (case != "empty")
    if case != "empty":
        assert trace["model"] == config.groq_model and trace["temperature"] == 0
        assert trace["context"][0]["chunk_id"] == "selected:3:1"
    if case == "length":
        assert trace["finish_reason"] == "length" and not trace["generation_completed"]
    if case == "bad_quote":
        assert trace["rejection"] == "quote_mismatch" and trace["rejected_source_id"] == 1
        assert "verify" in result.answer
    if case == "429":
        assert trace["upstream_status"] == 429 and trace["finish_reason"] is None
    assert key not in caplog.text


def test_trace_ids_are_per_request_and_never_exposed_in_public_mode(caplog):
    docs = Mock(spec=Retriever)
    docs.retrieve.return_value = []
    caplog.set_level(logging.INFO, logger="uvicorn.error.askmydocs.rag")
    request = ChatRequest(question="Find the venue.")
    for config in (Settings(rag_debug_context=True), Settings(rag_debug_context=True),
                   DemoSettings(demo_mode=True, rag_debug_context=True), Settings()):
        AnswerService(docs, Mock(), config).answer(request)
    traces = [json.loads(rec.args[0]) for rec in caplog.records if rec.msg.startswith("RAG request trace")]
    assert len(traces) == len({row["request_id"] for row in traces}) == 2

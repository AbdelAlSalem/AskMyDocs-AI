import json
from unittest.mock import Mock

import groq
import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import DemoSettings
from app.demo import create_demo_app
from app.rag import demo_retrieval
from app.rag.demo_embeddings import DemoQueryEncoder
from app.rag.demo_retrieval import DemoRetriever
from app.services.groq_client import AnswerGenerator, GeneratedAnswer
from test_demo_retrieval import orbit_query_vector


DEMO_ORIGIN = "https://demo.example"
DEMO_KEY = "synthetic-demo-provider-key"
SAMPLE_ID = "project-orbit"
SAMPLE_FILENAME = "project-orbit.pdf"
LAUNCH_QUESTION = "What is the launch date for Project Orbit?"
LAUNCH_QUOTE = "The launch date for Project Orbit is 18 June 2027."
WORKSHOP_QUOTE = "ستقام الورشة التجريبية في مكتبة النور في عمّان."


def _app(settings: DemoSettings, *, generator: AnswerGenerator | None = None, query_weights=(1.0, 0.0)):
    vector = orbit_query_vector(*query_weights)
    retriever = DemoRetriever(max_context_chars=settings.demo_max_context_chars, embed_query=lambda _: vector)
    return create_demo_app(settings, generator=generator, retriever=retriever)


def _completion(model: str, result: dict) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "mock-completion",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps(result, ensure_ascii=False),
        }}],
    })


@pytest.fixture
def make_demo_settings(tmp_path):
    def configure(**overrides):
        return DemoSettings(
            demo_mode=True,
            demo_allowed_origins=(DEMO_ORIGIN,),
            groq_api_key=DEMO_KEY,
            upload_dir=tmp_path / "uploads",
            chroma_dir=tmp_path / "chroma",
            sqlite_path=tmp_path / "documents.sqlite3",
            conversations_path=tmp_path / "conversations.sqlite3",
            **overrides,
        )

    return configure


@pytest.fixture
def mock_provider(monkeypatch):
    sdk = groq.Groq

    def install(handler):
        def build_client(**kwargs):
            return sdk(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

        monkeypatch.setattr(groq, "Groq", build_client)

    return install


def test_demo_startup_requires_the_semantic_encoder_and_does_not_fall_back(make_demo_settings, monkeypatch, tmp_path):
    monkeypatch.setattr(demo_retrieval, "DemoQueryEncoder", lambda: DemoQueryEncoder(tmp_path / "missing-model"))
    with pytest.raises(RuntimeError, match="Demo encoder assets are missing"):
        create_demo_app(make_demo_settings())


def test_demo_only_lists_bundled_documents_and_does_not_create_local_storage(make_demo_settings, tmp_path):
    generator = Mock(spec=AnswerGenerator)
    with TestClient(_app(make_demo_settings(), generator=generator)) as client:
        health = client.get("/api/health")
        documents = client.get("/api/documents")
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404

        upload = client.post(
            "/api/documents/upload", files={"files": ("untrusted.pdf", b"%PDF-1.7", "application/pdf")},
        )
        deletion = client.delete(f"/api/documents/{SAMPLE_ID}")
        for blocked in (upload, deletion):
            assert blocked.status_code == 403
            assert blocked.json()["detail"]["code"] == "demo_read_only"

        conversation_path = "/api/conversations/example-id"
        for response in (
            client.get("/api/conversations"),
            client.post("/api/conversations", json={}),
            client.get(conversation_path),
            client.patch(conversation_path, json={"title": "test"}),
            client.delete(conversation_path),
        ):
            assert response.status_code == 403
            assert response.json()["detail"]["code"] == "demo_browser_history"
        assert client.get("/api/documents").json() == documents.json()

    assert health.status_code == 200
    assert health.json()["demo_mode"] is True
    assert health.json()["groq_configured"] is True
    assert DEMO_KEY not in health.text
    assert documents.status_code == 200
    (sample,) = documents.json()["documents"]
    assert (sample["id"], sample["filename"], sample["page_count"], sample["status"]) == (
        SAMPLE_ID, SAMPLE_FILENAME, 2, "ready",
    )
    assert not list(tmp_path.iterdir())
    generator.generate.assert_not_called()


def test_demo_cors_only_exposes_the_exact_frontend_origin_and_public_methods(make_demo_settings):
    with TestClient(_app(make_demo_settings())) as client:
        preflight = client.options("/api/chat", headers={
            "Origin": DEMO_ORIGIN, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        })
        for origin in ("https://demo.example.evil.invalid", "http://demo.example", "https://sub.demo.example"):
            cross_origin = client.get("/api/health", headers={"Origin": origin})
            assert "access-control-allow-origin" not in cross_origin.headers
            rejected = client.options("/api/chat", headers={
                "Origin": origin, "Access-Control-Request-Method": "POST",
            })
            assert rejected.status_code == 400
            assert "access-control-allow-origin" not in rejected.headers
        destructive = client.options(f"/api/documents/{SAMPLE_ID}", headers={
            "Origin": DEMO_ORIGIN, "Access-Control-Request-Method": "DELETE",
        })

    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == DEMO_ORIGIN
    assert set(preflight.headers["access-control-allow-methods"].replace(" ", "").split(",")) == {"GET", "POST"}
    assert "access-control-allow-credentials" not in preflight.headers
    assert destructive.status_code == 400


def test_demo_bounds_request_and_prompt_before_generation(make_demo_settings):
    settings = make_demo_settings(
        demo_max_body_bytes=1024,
        demo_max_question_chars=48,
        demo_max_history_messages=2,
        demo_max_history_chars=8,
        demo_max_context_chars=250,
    )
    generator = Mock(spec=AnswerGenerator)
    generator.generate.return_value = GeneratedAnswer(status="insufficient_context", claims=[])
    history = [
        {"role": "user", "content": "Ignored question"},
        {"role": "assistant", "content": "Ignored answer"},
        {"role": "user", "content": "abcdefghij"},
        {"role": "assistant", "content": "reply"},
    ]
    with TestClient(_app(settings, generator=generator)) as client:
        too_long = client.post("/api/chat", json={"question": "x" * 49})
        assert too_long.status_code == 422
        assert too_long.json()["detail"]["code"] == "demo_question_too_long"
        oversized_body = client.post("/api/chat", json={
            "question": LAUNCH_QUESTION, "history": [{"role": "user", "content": "x" * 1050}],
        })
        assert oversized_body.status_code == 413
        invalid_history = client.post("/api/chat", json={
            "question": LAUNCH_QUESTION, "history": [{"role": "user", "content": "x"}] * 9,
        })
        assert invalid_history.status_code == 422
        assert invalid_history.json()["detail"]["code"] == "invalid_request"
        local_payload = client.post("/api/chat", json={
            "question": LAUNCH_QUESTION, "conversation_id": "could-be-private",
        })
        assert local_payload.status_code == 422
        assert "could-be-private" not in local_payload.text
        generator.generate.assert_not_called()

        bounded = client.post("/api/chat", json={"question": LAUNCH_QUESTION, "history": history})

    assert bounded.status_code == 200
    assert bounded.headers["cache-control"] == "no-store"
    generator.generate.assert_called_once()
    prompt = json.loads(generator.generate.call_args.kwargs["user_prompt"])
    assert prompt["conversation_history"] == [
        {"role": "user", "content": "hij"},
        {"role": "assistant", "content": "reply"},
    ]
    (passage,) = prompt["retrieved_passages"]
    assert passage["page_number"] == 1
    assert LAUNCH_QUOTE in passage["text"]
    assert len(passage["text"]) <= settings.demo_max_context_chars


@pytest.mark.parametrize(("question", "grounding"), [
    ("When will Orbit kick off, and which venue will host the training?", [
        (1, LAUNCH_QUOTE, "The project launches on 18 June 2027."),
        (2, WORKSHOP_QUOTE, "The workshop will be at Al Noor Library in Amman."),
    ]),
    ("أين ستُعقد الجلسة التدريبية؟", [(2, WORKSHOP_QUOTE, WORKSHOP_QUOTE)]),
], ids=["cross-page", "arabic"])
def test_demo_semantic_retrieval_through_mocked_groq_sdk(make_demo_settings, mock_provider, question, grounding):
    settings = make_demo_settings(demo_max_completion_tokens=256)
    provider = Mock()

    def complete(request):
        provider(request)
        assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {DEMO_KEY}"
        body = json.loads(request.content)
        assert body["max_completion_tokens"] == 256
        assert body["reasoning_effort"] == "low"
        assert body["response_format"]["json_schema"]["strict"] is True
        prompt = json.loads(body["messages"][1]["content"])
        assert prompt["question"] == question
        passages = prompt["retrieved_passages"]
        claims = []
        for page, quote, answer in grounding:
            passage = next(item for item in passages if item["page_number"] == page and quote in item["text"])
            claims.append({"text": answer, "evidence": [{"source_id": passage["id"], "quote": quote}]})
        return _completion(body["model"], {"status": "answered", "claims": claims})

    mock_provider(complete)
    query_weights = (1.0, 1.0) if len(grounding) > 1 else (0.0, 1.0)
    with TestClient(_app(settings, query_weights=query_weights)) as client:
        response = client.post("/api/chat", json={"question": question, "document_ids": [SAMPLE_ID]})

    assert response.status_code == 200, response.text
    result = response.json()
    assert set(result) == {"status", "answer", "sources"}
    assert result["status"] == "answered"
    assert DEMO_KEY not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert not any(path.exists() for path in (
        settings.upload_dir, settings.chroma_dir, settings.sqlite_path, settings.conversations_path,
    ))
    for source, (page, quote, answer) in zip(result["sources"], grounding, strict=True):
        assert source["document_id"] == SAMPLE_ID
        assert source["filename"] == SAMPLE_FILENAME
        assert source["page_number"] == page
        assert source["snippet"] == quote
        assert source["chunk_ids"] == [f"{SAMPLE_ID}:{page}:1"]
        assert f'{answer} [{source["id"]}]' in result["answer"]
    provider.assert_called_once()


@pytest.mark.parametrize(("question", "page", "quote", "unsupported_claim", "refusal"), [
    (
        "What time does Orbit launch on 18 June 2027?",
        1,
        LAUNCH_QUOTE,
        "Project Orbit launches at 09:00.",
        "enough information",
    ),
    (
        "ما تكلفة حضور الورشة في مكتبة النور؟",
        2,
        WORKSHOP_QUOTE,
        "تكلفة حضور الورشة ١٠ دنانير.",
        "معلومات كافية",
    ),
], ids=["missing-launch-time", "missing-workshop-fee"])
@pytest.mark.parametrize("hallucinated", [False, True], ids=["provider-abstains", "provider-invents-figure"])
def test_demo_abstains_for_unsupported_facts_despite_related_passages(
    make_demo_settings, mock_provider, question, page, quote, unsupported_claim, refusal, hallucinated,
):
    provider = Mock()

    def complete(request):
        provider(request)
        body = json.loads(request.content)
        prompt = json.loads(body["messages"][1]["content"])
        assert prompt["question"] == question
        related = next(
            passage
            for passage in prompt["retrieved_passages"]
            if passage["page_number"] == page and quote in passage["text"]
        )
        if hallucinated:
            return _completion(body["model"], {
                "status": "answered",
                "claims": [{
                    "text": unsupported_claim,
                    "evidence": [{"source_id": related["id"], "quote": quote}],
                }],
            })
        return _completion(body["model"], {"status": "insufficient_context", "claims": []})

    mock_provider(complete)
    weights = (1.0, 0.0) if page == 1 else (0.0, 1.0)
    with TestClient(_app(make_demo_settings(), query_weights=weights)) as client:
        response = client.post("/api/chat", json={"question": question, "document_ids": [SAMPLE_ID]})

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "insufficient_context"
    if hallucinated:
        refusal = "تعذّر التحقق" if question.endswith("؟") else "couldn’t verify"
    assert refusal in result["answer"]
    assert result["sources"] == []
    provider.assert_called_once()


def test_demo_does_not_call_groq_for_absent_evidence_or_out_of_scope_ids(make_demo_settings):
    generator = Mock(spec=AnswerGenerator)
    with TestClient(_app(make_demo_settings(), generator=generator, query_weights=(-1.0, -1.0))) as client:
        absent = client.post("/api/chat", json={"question": "What is the pilot team's catering budget?"})
        deselected = client.post("/api/chat", json={"question": LAUNCH_QUESTION, "document_ids": []})
        local_id = client.post("/api/chat", json={"question": LAUNCH_QUESTION, "document_ids": ["a-local-file"]})

    for response in (absent, deselected):
        assert response.status_code == 200
        assert response.json()["status"] == "insufficient_context"
        assert response.json()["sources"] == []
    assert local_id.status_code == 404
    assert local_id.json()["detail"]["code"] == "invalid_document_scope"
    generator.generate.assert_not_called()


def test_demo_returns_friendly_app_rate_limit_and_ignores_spoofed_ip_headers(make_demo_settings):
    settings = make_demo_settings(demo_ip_rate_limit=1, demo_rate_window_seconds=60)
    generator = Mock(spec=AnswerGenerator)
    generator.generate.return_value = GeneratedAnswer(status="insufficient_context", claims=[])
    with TestClient(_app(settings, generator=generator), client=("198.51.100.5", 51234)) as client:
        first = client.post("/api/chat", json={"question": LAUNCH_QUESTION})
        blocked = client.post("/api/chat", json={"question": LAUNCH_QUESTION}, headers={
            "Origin": DEMO_ORIGIN, "CF-Connecting-IP": "198.51.100.10", "X-Forwarded-For": "198.51.100.10",
        })

    assert first.status_code == 200
    assert blocked.status_code == 429
    assert blocked.json()["detail"]["code"] == "demo_rate_limit"
    assert "try again" in blocked.json()["detail"]["message"].lower()
    assert 1 <= int(blocked.headers["retry-after"]) <= 60
    assert blocked.headers["access-control-expose-headers"] == "Retry-After"
    generator.generate.assert_called_once()


def test_demo_sanitizes_groq_rate_limit_and_preserves_retry_after(make_demo_settings, mock_provider):
    provider = Mock(return_value=httpx.Response(
        429, headers={"Retry-After": "13"}, json={"error": {"message": f"Private provider text: {DEMO_KEY}"}},
    ))
    mock_provider(provider)
    with TestClient(_app(make_demo_settings())) as client:
        response = client.post("/api/chat", json={"question": LAUNCH_QUESTION}, headers={"Origin": DEMO_ORIGIN})

    assert response.status_code == 429
    assert response.json()["detail"]["code"] == "groq_rate_limit"
    assert "retry" in response.json()["detail"]["message"].lower()
    assert response.headers["retry-after"] == "13"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["access-control-expose-headers"] == "Retry-After"
    assert DEMO_KEY not in response.text
    assert "Private provider text" not in response.text
    provider.assert_called_once()


def test_demo_challenge_missing_token_is_rejected_without_provider_or_verification_calls(make_demo_settings, monkeypatch):
    site_key = "synthetic-public-site-key"
    verification_secret = "synthetic-turnstile-secret"
    settings = make_demo_settings(
        demo_turnstile_site_key=site_key,
        demo_turnstile_secret_key=verification_secret,
        demo_turnstile_hostname="demo.example",
    )
    generator = Mock(spec=AnswerGenerator)
    verify = Mock(side_effect=AssertionError("Tokenless requests must not call Turnstile."))
    monkeypatch.setattr("app.services.demo_security.httpx.post", verify)
    with TestClient(_app(settings, generator=generator)) as client:
        health = client.get("/api/health")
        rejected = client.post("/api/chat", json={"question": LAUNCH_QUESTION})

    assert health.json()["turnstile_site_key"] == site_key
    assert health.json()["turnstile_action"] == settings.demo_turnstile_action
    assert verification_secret not in health.text
    assert rejected.status_code == 403
    assert rejected.json()["detail"]["code"] == "demo_verification_required"
    verify.assert_not_called()
    generator.generate.assert_not_called()

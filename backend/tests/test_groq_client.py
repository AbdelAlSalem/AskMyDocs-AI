import json
from unittest.mock import Mock

import groq
import httpx
import pytest

from app.core.config import Settings
from app.models.schemas import ChatRequest
from app.rag.answer import AnswerService
from app.rag.document_wide import DocumentCoverage
from app.rag.types import RetrievedChunk
from app.services.groq_client import GroqGenerator, ProviderError


@pytest.fixture
def mock_groq_transport(monkeypatch):
    sdk_client = groq.Groq

    def install(handler):
        def build_client(**kwargs):
            return sdk_client(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

        monkeypatch.setattr(groq, "Groq", build_client)

    return install


@pytest.mark.parametrize("summary", [False, True], ids=["factual", "summary"])
def test_official_sdk_sends_structured_grounded_request(mock_groq_transport, summary):
    prompt = json.dumps({"retrieved_passages": [{"id": 1, "text": "Ended in June."}]})

    def provider(request):
        assert request.url == "https://api.groq.com/openai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-oss-120b"
        assert body["reasoning_effort"] == ("low" if summary else "medium")
        assert body["max_completion_tokens"] == 2500
        if summary:
            schema = body["response_format"]["json_schema"]
            assert schema["strict"] and schema["name"] == "document_summary"
            point_schema = schema["schema"]["properties"]["points"]["items"]
            assert point_schema["required"] == ["text", "evidence_ids"]
        else:
            assert body["response_format"]["json_schema"]["strict"] is True
            schema = body["response_format"]["json_schema"]["schema"]
            evidence_schema = schema["properties"]["claims"]["items"]["properties"]["evidence"]["items"]
            assert schema["required"] == ["status", "claims"]
            assert evidence_schema["required"] == ["source_id", "quote"]
        system, user = body["messages"]
        assert system == {"role": "system", "content": "Instructions"}
        assert user["role"] == "user"
        passage = json.loads(user["content"])["retrieved_passages"][0]
        if summary:
            assert passage == {"evidence": [[1, "Ended in June."]]}
        else:
            assert passage == {"id": 1, "text": "Ended in June."}
        point = {"text": "June.", "source_id": 1, "quote": "Ended in June."}
        response = {"points": [{"text": "June.", "evidence_ids": [1]}]} if summary else {
            "status": "answered",
            "claims": [{"text": point["text"], "evidence": [{"source_id": 1, "quote": point["quote"]}]}],
        }
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(response),
            }}],
        })

    mock_groq_transport(provider)
    generator = GroqGenerator(Settings(groq_api_key="test-key", groq_model="openai/gpt-oss-120b"))
    answer = generator.generate(system_prompt="Instructions", user_prompt=prompt, summary=summary)

    assert answer.status == "answered"
    assert answer.claims[0].text == "June."
    assert answer.claims[0].evidence[0].quote == "Ended in June."


@pytest.mark.parametrize(("finish_reason", "output", "expected_code", "diagnostic_reason"), [
    ("length", '{"status": "synthetic-partial-output-do-not-log', "groq_output_limit", "length"),
    ("stop", "synthetic-invalid-json-do-not-log", "groq_invalid_response", "invalid_schema"),
    (
        "stop",
        '{"points": [{"text": "synthetic-missing-evidence-do-not-log"}]}',
        "groq_invalid_response",
        "invalid_schema",
    ),
])
def test_summary_reports_truncation_separately_from_invalid_output(
    mock_groq_transport, caplog, finish_reason, output, expected_code, diagnostic_reason,
):
    key = "synthetic-key-do-not-log"
    source = "synthetic-source-do-not-log"
    calls = 0

    def provider(request):
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert body["reasoning_effort"] == "low"
        assert body["max_completion_tokens"] == 2500
        assert body["include_reasoning"] is False
        assert body["response_format"]["json_schema"]["name"] == "document_summary"
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": {
                "role": "assistant", "content": output,
            }}],
            "usage": {
                "prompt_tokens": 2432,
                "completion_tokens": 2500,
                "total_tokens": 4932,
                "completion_tokens_details": {"reasoning_tokens": 2334},
            },
        })

    mock_groq_transport(provider)
    generator = GroqGenerator(Settings(groq_api_key=key))
    user_prompt = json.dumps({"retrieved_passages": [{"id": 1, "text": source}]})
    with pytest.raises(ProviderError) as exc:
        generator.generate(system_prompt="Instructions", user_prompt=user_prompt, summary=True)

    assert calls == 1
    assert (exc.value.status_code, exc.value.code) == (502, expected_code)
    assert exc.value.output_failure
    assert f"reason={diagnostic_reason} upstream_status=200" in caplog.text
    assert f"context_chars={len(source)}" in caplog.text
    assert "reasoning_effort=low" in caplog.text
    assert "completion_tokens=2500 reasoning_tokens=2334" in caplog.text
    assert "do-not-log" not in caplog.text
    assert "do-not-log" not in str(exc.value)


@pytest.mark.parametrize(("provider_code", "logged_code"), [
    ("json_validate_failed", "json_validate_failed"),
    ("synthetic-secret-error-code", "None"),
])
def test_provider_bad_request_logs_only_allowlisted_metadata(
    mock_groq_transport, caplog, provider_code, logged_code,
):
    key = "synthetic-key-do-not-log"
    details = "synthetic-error-details-do-not-log"
    partial = "synthetic-failed-generation-do-not-log"
    mock_groq_transport(lambda _: httpx.Response(400, json={"error": {
        "code": provider_code, "message": details, "failed_generation": partial,
    }}))
    generator = GroqGenerator(Settings(groq_api_key=key))

    with pytest.raises(ProviderError) as exc:
        generator.generate(system_prompt="Instructions", user_prompt='{"retrieved_passages":[]}', summary=True)

    assert (exc.value.status_code, exc.value.code) == (502, "groq_request_failed")
    assert exc.value.output_failure is (provider_code == "json_validate_failed")
    assert f"reason=provider_error upstream_status=400 upstream_code={logged_code}" in caplog.text
    for value in (key, details, partial, "synthetic-secret-error-code"):
        assert value not in caplog.text
        assert value not in str(exc.value)


@pytest.mark.parametrize(("provider_status", "http_status", "code"), [
    (401, 503, "groq_auth_error"),
    (404, 503, "groq_model_invalid"),
    (429, 429, "groq_rate_limit"),
    (500, 502, "groq_request_failed"),
])
def test_provider_http_errors_are_sanitized(mock_groq_transport, provider_status, http_status, code):
    calls = 0

    def provider(_):
        nonlocal calls
        calls += 1
        return httpx.Response(provider_status, json={"error": {"message": "private details"}})

    mock_groq_transport(provider)
    generator = GroqGenerator(Settings(groq_api_key="test-key"))
    with pytest.raises(ProviderError) as exc:
        generator.generate(system_prompt="Instructions", user_prompt="Evidence")

    assert calls == 1
    assert (exc.value.status_code, exc.value.code) == (http_status, code)
    assert "private details" not in str(exc.value)


@pytest.mark.parametrize(("provider_error", "http_status", "code"), [
    (httpx.ConnectTimeout, 504, "groq_timeout"),
    (httpx.ConnectError, 502, "groq_network_error"),
])
def test_provider_network_errors_are_sanitized(mock_groq_transport, provider_error, http_status, code):
    def provider(request):
        raise provider_error("private details", request=request)

    mock_groq_transport(provider)
    generator = GroqGenerator(Settings(groq_api_key="test-key"))
    with pytest.raises(ProviderError) as exc:
        generator.generate(system_prompt="Instructions", user_prompt="Evidence")

    assert (exc.value.status_code, exc.value.code) == (http_status, code)
    assert "private details" not in str(exc.value)


def test_invalid_model_is_rejected_without_contacting_groq(mock_groq_transport):
    def provider(_):
        pytest.fail("An invalid model must not trigger a provider request.")

    mock_groq_transport(provider)
    generator = GroqGenerator(Settings(groq_api_key="test-key", groq_model="invalid-model"))
    with pytest.raises(ProviderError, match="GROQ_MODEL") as exc:
        generator.generate(system_prompt="Instructions", user_prompt="Evidence")

    assert exc.value.code == "groq_model_invalid"


@pytest.mark.parametrize(("question", "point"), [
    ("Summarize the entire document in one bullet point with page citations.",
     "- The pilot trained 12 operators, and 8 stations adopted the design."),
    ("لخص أهم المعلومات الموجودة في المستند كاملًا في نقطة واحدة باللغة العربية.",
     "- درّب المشروع التجريبي 12 مشغّلًا واعتمدت 8 محطات التصميم."),
])
def test_summary_resolves_multi_page_generated_point_to_exact_source_spans(mock_groq_transport, question, point):
    first_page = (
        "The transport pilot focused on planning work at stations serving the harbor district. "
        "Operators reviewed the training plan, compared it with the route timetable, and documented the results. "
        "The team collected feedback from local supervisors during the training period. "
        "Its final report includes the changes made to station procedures after each exercise. "
        "The transport pilot trained 12 operators."
    )
    passages = [
        RetrievedChunk("pilot", "pilot.pdf", 1, "pilot:1:1", first_page, 0.0),
        RetrievedChunk("pilot", "pilot.pdf", 4, "pilot:4:1",
                       "Eight stations joined the pilot. All 8 stations adopted the design.", 0.0),
    ]
    documents = Mock()
    documents.retrieve_document_wide.return_value = DocumentCoverage(passages, True)

    def provider(request):
        wire = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert wire["question"] == question and wire["conversation_history"] == []
        units = {span: number for row in wire["retrieved_passages"] for number, span in row["evidence"]}
        assert wire["summary_language"] == ("Arabic" if question.startswith("لخص") else "English")
        ids = [units[passage.text] for passage in passages]
        return httpx.Response(200, json={
            "id": "mock", "object": "chat.completion", "created": 0, "model": "openai/gpt-oss-20b",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps({"points": [{"text": point, "evidence_ids": ids}]}),
            }}],
        })

    mock_groq_transport(provider)
    settings = Settings(groq_api_key="test-key")
    answer = AnswerService(documents, GroqGenerator(settings), settings).answer(
        ChatRequest(question=question, document_ids=["pilot"]),
    )

    assert answer.status == "answered" and answer.answer == f"{point} [1] [2]"
    assert first_page.index("12 operators") > 360
    assert [(s.page_number, s.snippet) for s in answer.sources] == [
        (1, passages[0].text), (4, passages[1].text),
    ]
    documents.retrieve.assert_not_called()


def test_summary_unknown_span_id_fails_verification_after_parsing(mock_groq_transport):
    mock_groq_transport(lambda _: httpx.Response(200, json={
        "id": "mock", "object": "chat.completion", "created": 0, "model": "openai/gpt-oss-20b",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant",
            "content": json.dumps({"points": [{"text": "The route changed.", "evidence_ids": [999]}]}),
        }}],
    }))
    prompt = json.dumps({"retrieved_passages": [{"id": 1, "text": "The route changed."}]})
    with pytest.raises(ProviderError) as failure:
        GroqGenerator(Settings(groq_api_key="test-key")).generate(system_prompt="Summarize", user_prompt=prompt, summary=True)
    assert failure.value.code == "groq_unverified_evidence" and failure.value.output_failure


@pytest.mark.parametrize("valid_alternative", [True, False])
def test_grouped_summary_checks_every_candidate_before_count_and_coverage(mock_groq_transport, caplog, valid_alternative):
    passages = [
        RetrievedChunk("pilot", "pilot.pdf", 1, "pilot:1:1", "The harbor trial trained 12 station operators.", 0.0),
        RetrievedChunk("pilot", "pilot.pdf", 2, "pilot:2:1", "The new route connects 8 stations.", 0.0),
    ]
    documents = Mock()
    documents.retrieve_document_wide.return_value = DocumentCoverage(passages, True)
    calls = []

    def provider(request):
        wire = json.loads(request.content)
        calls.append(request)
        schema = wire["response_format"]["json_schema"]["schema"]
        assert schema["required"] == ["page_1", "page_2"]
        first = schema["properties"]["page_1"]
        assert first["minItems"] == 1 and first["items"]["properties"]["evidence_ids"]["items"]["enum"] == [1]
        points = {
            "page_1": [{"text": "The harbor trial trained 900 operators.", "evidence_ids": [1]},
                       {"text": "The harbor trial trained 12 operators." if valid_alternative else "It trained 17 operators.",
                        "evidence_ids": [1]}],
            "page_2": [{"text": "The new route connects 8 stations.", "evidence_ids": [2]}],
        }
        return httpx.Response(200, json={
            "id": "mock", "object": "chat.completion", "created": 0, "model": "openai/gpt-oss-20b",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(points),
            }}],
        })
    mock_groq_transport(provider)
    caplog.set_level("INFO", logger="uvicorn.error.askmydocs.rag")
    settings = Settings(groq_api_key="test-key", rag_debug_context=True)
    answer = AnswerService(documents, GroqGenerator(settings), settings).answer(
        ChatRequest(question="Summarize the entire document in two bullet points."),
    )
    assert len(calls) == 1
    assert "900" not in answer.answer and "17 operators" not in answer.answer
    trace = next(json.loads(record.getMessage().split("RAG request trace: ", 1)[1])
                 for record in caplog.records if record.getMessage().startswith("RAG request trace: "))
    assert trace["rejected_summary_candidates"][0] == {"candidate": 1, "reason": "unsupported_figures"}
    if valid_alternative:
        assert trace["outcome"] == "generated" and "rejection" not in trace
        assert answer.answer == "- The harbor trial trained 12 operators. [1]\n\n- The new route connects 8 stations. [2]"
        assert [s.page_number for s in answer.sources] == [1, 2]
    else:
        assert trace["outcome"] == "verification_rejected" and trace["rejection"] == "summary_coverage_incomplete"
        assert "cited excerpts" in answer.answer

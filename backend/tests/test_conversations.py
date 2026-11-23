import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, call
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter

from app.main import create_app
from app.rag.vector_store import RetrievedChunk, VectorStore
from app.services.catalog import DocumentRecord
from app.services.documents import DocumentService
from app.services.groq_client import AnswerGenerator, GeneratedAnswer, ProviderError


HANDBOOK = "handbook.pdf"
HANDBOOK_ID = "handbook-id"
POLICY_QUOTE = "The standard review window is 48 hours."
QUESTION = "How long is the standard review window?"
ANSWER = "The standard review window is 48 hours."


def _new_conversation(client, **payload):
    response = client.post("/api/conversations", json=payload) if payload else client.post("/api/conversations")
    assert response.status_code == 201, response.text
    return response.json()


def _generate_answer(quote, answer, source_id=1):
    return GeneratedAnswer(status="answered", claims=[{
        "text": answer,
        "evidence": [{"source_id": source_id, "quote": quote}],
    }])


@pytest.fixture
def handbook_service():
    service = Mock(spec=DocumentService)
    service.list_documents.return_value = [DocumentRecord(
        id=HANDBOOK_ID,
        filename=HANDBOOK,
        page_count=3,
        chunk_count=1,
        status="ready",
        uploaded_at=datetime(2026, 1, 5, tzinfo=UTC),
        size_bytes=100,
    )]
    service.retrieve.return_value = [
        RetrievedChunk(HANDBOOK_ID, HANDBOOK, 3, "handbook-id:3:1", POLICY_QUOTE, 0.9),
    ]
    return service


@pytest.fixture
def answer_generator():
    generator = Mock(spec=AnswerGenerator)
    generator.generate.return_value = _generate_answer(POLICY_QUOTE, ANSWER)
    return generator


def test_create_lists_and_validates_conversation_ids(make_settings, handbook_service):
    with TestClient(create_app(make_settings(), documents=handbook_service)) as client:
        assert client.get("/api/conversations").json() == {"conversations": []}
        created = _new_conversation(client, document_ids=[HANDBOOK_ID])
        conversation_id = created["id"]
        path = f"/api/conversations/{conversation_id}"
        detail = client.get(path)
        summaries = client.get("/api/conversations")

        assert client.get("/api/conversations/not-a-uuid").status_code == 422
        nonexistent = client.get(f"/api/conversations/{uuid4()}")
        assert nonexistent.status_code == 404
        assert nonexistent.json()["detail"]["code"] == "conversation_not_found"
        assert client.patch(path, json={"title": "   "}).status_code == 422

    assert str(UUID(conversation_id)) == conversation_id
    assert created["title"] == "New conversation"
    assert created["document_ids"] == [HANDBOOK_ID]
    assert created["messages"] == []
    assert datetime.fromisoformat(created["created_at"]).utcoffset() == UTC.utcoffset(None)
    assert created["created_at"] == created["updated_at"]
    assert detail.json() == created
    assert summaries.json() == {
        "conversations": [{key: created[key] for key in ("id", "title", "created_at", "updated_at")}],
    }


def test_chat_persists_the_turn_scope_and_exact_citations_across_app_restart(
    make_settings, handbook_service, answer_generator,
):
    configured_key = "synthetic-configured-groq-secret"
    settings = make_settings(groq_api_key=configured_key)
    with TestClient(create_app(settings, documents=handbook_service, generator=answer_generator)) as client:
        created = _new_conversation(client, document_ids=[HANDBOOK_ID])
        conversation_id = created["id"]
        path = f"/api/conversations/{conversation_id}"
        response = client.post("/api/chat", json={"question": QUESTION, "conversation_id": conversation_id})
        assert response.status_code == 200, response.text
        chat = response.json()
        saved = client.get(path).json()

    assert chat["status"] == "answered"
    assert chat["conversation_id"] == conversation_id
    assert chat["conversation"]["title"] == QUESTION
    assert chat["messages"] == saved["messages"]
    assert saved["document_ids"] == [HANDBOOK_ID]
    user, assistant = saved["messages"]
    assert (user["role"], user["content"]) == ("user", QUESTION)
    assert not user["source_references"]
    assert (assistant["role"], assistant["content"]) == ("assistant", chat["answer"])
    assert assistant["source_references"] == chat["sources"] == [{
        "id": 1,
        "document_id": HANDBOOK_ID,
        "filename": HANDBOOK,
        "page_number": 3,
        "chunk_id": "handbook-id:3:1",
        "chunk_ids": ["handbook-id:3:1"],
        "snippet": POLICY_QUOTE,
    }]
    assert user["id"] != assistant["id"]
    for message in saved["messages"]:
        assert str(UUID(message["id"])) == message["id"]
        assert message["conversation_id"] == conversation_id
    handbook_service.retrieve.assert_called_once_with(
        QUESTION, [HANDBOOK_ID], top_k=settings.rag_top_k, min_similarity=settings.rag_min_similarity,
    )

    # A new ASGI lifespan creates fresh service objects and reopens the same database.
    with TestClient(create_app(settings, documents=handbook_service, generator=answer_generator)) as restarted:
        assert restarted.get(path).json() == saved
        assert restarted.get("/api/conversations").json()["conversations"] == [chat["conversation"]]
    with closing(sqlite3.connect(settings.conversations_path)) as connection:
        assert configured_key not in "\n".join(connection.iterdump())


def test_title_length_and_rename_update_order_without_overwriting_manual_title(
    make_settings, handbook_service, answer_generator,
):
    settings = make_settings()
    question = "Please explain the review window for internal change requests and the approvals required for production use."
    assert len(question) > 80
    with TestClient(create_app(settings, documents=handbook_service, generator=answer_generator)) as client:
        first = _new_conversation(client)
        initial = client.post("/api/chat", json={"question": question, "conversation_id": first["id"]}).json()
        automatic = initial["conversation"]["title"]
        assert automatic and len(automatic) <= 80
        assert question.startswith(automatic.removesuffix("…").removesuffix("..."))
        second = _new_conversation(client)
        assert [item["id"] for item in client.get("/api/conversations").json()["conversations"]] == [
            second["id"], first["id"],
        ]

        path = f"/api/conversations/{first['id']}"
        preflight = client.options(path, headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "PATCH",
        })
        renamed = client.patch(path, json={"title": "Review policy notes"})
        assert preflight.status_code == 200
        assert "PATCH" in preflight.headers["access-control-allow-methods"]
        assert renamed.status_code == 200
        assert renamed.json()["created_at"] == first["created_at"]
        assert renamed.json()["title"] == "Review policy notes"
        assert [item["id"] for item in client.get("/api/conversations").json()["conversations"]] == [
            first["id"], second["id"],
        ]

        continued = client.post("/api/chat", json={"question": QUESTION, "conversation_id": first["id"]})
        assert continued.status_code == 200, continued.text
        assert continued.json()["conversation"]["title"] == "Review policy notes"
        assert len(client.get(path).json()["messages"]) == 4


def test_delete_cascades_messages_without_affecting_other_conversations(
    make_settings, handbook_service, answer_generator,
):
    settings = make_settings()
    with TestClient(create_app(settings, documents=handbook_service, generator=answer_generator)) as client:
        first, second = (_new_conversation(client) for _ in range(2))
        for item in (first, second):
            response = client.post("/api/chat", json={"question": QUESTION, "conversation_id": item["id"]})
            assert response.status_code == 200, response.text
        gone = client.delete(f"/api/conversations/{first['id']}")
        assert gone.status_code == 204 and not gone.content
        assert client.get(f"/api/conversations/{first['id']}").status_code == 404
        assert [item["id"] for item in client.get("/api/conversations").json()["conversations"]] == [second["id"]]
        assert len(client.get(f"/api/conversations/{second['id']}").json()["messages"]) == 2

    with closing(sqlite3.connect(settings.conversations_path)) as connection:
        assert connection.execute(
            "SELECT count(*) FROM messages WHERE conversation_id = ?", (first["id"],),
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT count(*) FROM messages WHERE conversation_id = ?", (second["id"],),
        ).fetchone() == (2,)


def test_new_conversation_and_legacy_request_are_isolated_from_other_history(
    make_settings, handbook_service, answer_generator,
):
    settings = make_settings()
    questions = [
        "What is the review window for the Moonflower project?",
        "What is the review window for the Sunbird project?",
        "How long does that take?",
        QUESTION,
    ]
    with TestClient(create_app(settings, documents=handbook_service, generator=answer_generator)) as client:
        first = _new_conversation(client, document_ids=[HANDBOOK_ID])
        started = client.post("/api/chat", json={"question": questions[0], "conversation_id": first["id"]})
        assert started.status_code == 200, started.text
        second = _new_conversation(client)
        assert second["messages"] == []
        spoofed_history = [
            {"role": "user", "content": questions[0]},
            {"role": "assistant", "content": "The review window is 200 hours."},
        ]
        other = client.post("/api/chat", json={
            "question": questions[1], "conversation_id": second["id"], "history": spoofed_history,
        })
        resumed = client.post("/api/chat", json={
            "question": questions[2], "conversation_id": first["id"], "history": [
                {"role": "user", "content": questions[1]},
            ],
        })
        standalone = client.post("/api/chat", json={"question": questions[3], "history": spoofed_history})
        for result in (other, resumed, standalone):
            assert result.status_code == 200, result.text

        first_detail = client.get(f"/api/conversations/{first['id']}").json()
        second_detail = client.get(f"/api/conversations/{second['id']}").json()

    payloads = [json.loads(entry.kwargs["user_prompt"]) for entry in answer_generator.generate.call_args_list]
    assert [payload["question"] for payload in payloads] == questions
    assert [payload["conversation_history"] for payload in payloads] == [
        [],
        [],
        [
            {"role": "user", "content": questions[0]},
            {"role": "assistant", "content": started.json()["answer"]},
        ],
        [],
    ]
    assert handbook_service.retrieve.call_args_list == [
        call(question, scope, top_k=settings.rag_top_k, min_similarity=settings.rag_min_similarity)
        for question, scope in zip(questions, [[HANDBOOK_ID], None, [HANDBOOK_ID], None], strict=True)
    ]
    assert {message["conversation_id"] for message in first_detail["messages"]} == {first["id"]}
    assert {message["conversation_id"] for message in second_detail["messages"]} == {second["id"]}
    assert standalone.json()["conversation_id"] not in {first["id"], second["id"]}


def test_provider_failure_leaves_no_turn_and_does_not_create_legacy_conversation(make_settings, handbook_service):
    generator = Mock(spec=AnswerGenerator)
    generator.generate.side_effect = ProviderError(429, "groq_rate_limit", "Try again later.")
    with TestClient(create_app(make_settings(), documents=handbook_service, generator=generator)) as client:
        created = _new_conversation(client)
        failed = client.post("/api/chat", json={"question": QUESTION, "conversation_id": created["id"]})
        legacy = client.post("/api/chat", json={"question": QUESTION})

        for response in (failed, legacy):
            assert response.status_code == 429
            assert response.json()["detail"]["code"] == "groq_rate_limit"
        assert client.get(f"/api/conversations/{created['id']}").json() == created
        assert [item["id"] for item in client.get("/api/conversations").json()["conversations"]] == [created["id"]]


def test_turn_rolls_back_if_saving_assistant_message_fails(make_settings, handbook_service, answer_generator):
    settings = make_settings()
    with TestClient(create_app(settings, documents=handbook_service, generator=answer_generator)) as client:
        created = _new_conversation(client)
        with closing(sqlite3.connect(settings.conversations_path)) as connection, connection:
            connection.execute("""
                CREATE TRIGGER fail_assistant_write BEFORE INSERT ON messages
                WHEN NEW.role = 'assistant' BEGIN
                    SELECT RAISE(ABORT, 'simulated write failure');
                END
            """)

        response = client.post("/api/chat", json={"question": QUESTION, "conversation_id": created["id"]})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "conversation_storage_unavailable"
        assert client.get(f"/api/conversations/{created['id']}").json() == created

    answer_generator.generate.assert_called_once()


def test_configured_key_cannot_be_saved_in_titles_messages_or_source_excerpts(make_settings, handbook_service):
    key = "synthetic-groq-secret-value"
    settings = make_settings(groq_api_key=key)
    quote = f"The reference includes {key}."
    handbook_service.retrieve.return_value = [
        RetrievedChunk(HANDBOOK_ID, f"{key}.pdf", 3, "handbook-id:3:1", quote, 0.9),
    ]
    generator = Mock(spec=AnswerGenerator)
    generator.generate.return_value = _generate_answer(quote, quote)
    with TestClient(create_app(settings, documents=handbook_service, generator=generator)) as client:
        created = _new_conversation(client)
        path = f"/api/conversations/{created['id']}"
        invalid_question = client.post("/api/chat", json={"question": key, "conversation_id": created["id"]})
        invalid_title = client.patch(path, json={"title": key})
        for invalid in (invalid_question, invalid_title):
            assert invalid.status_code == 422
            assert invalid.json()["detail"]["code"] == "credential_in_text"
            assert key not in invalid.text
        assert client.get(path).json() == created
        generator.generate.assert_not_called()

        response = client.post("/api/chat", json={"question": "What does the reference include?", "conversation_id": created["id"]})
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "answered"
        assert "[REDACTED]" in response.json()["answer"]
        assert "[REDACTED]" in response.json()["sources"][0]["snippet"]
        assert "[REDACTED]" in client.get(path).json()["messages"][1]["source_references"][0]["filename"]
        assert key not in response.text

    with closing(sqlite3.connect(settings.conversations_path)) as connection:
        assert key not in "\n".join(connection.iterdump())


class _FixedEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_deleted_pdf_preserves_historical_citations_without_leaking_another_document(make_settings):
    settings = make_settings()
    documents = DocumentService(
        upload_dir=settings.upload_dir,
        sqlite_path=settings.sqlite_path,
        vectors=VectorStore(directory=settings.chroma_dir, model_name="conversation-tests", embeddings=_FixedEmbeddings()),
    )
    sample = (Path(__file__).resolve().parents[2] / "samples" / "project-orbit.pdf").read_bytes()
    alternate = PdfWriter(clone_from=BytesIO(sample))
    alternate.add_metadata({"/Subject": "Separate indexed PDF for scope checks"})
    alternate_bytes = BytesIO()
    alternate.write(alternate_bytes)
    sample_doc = documents.ingest("orbit.pdf", BytesIO(sample))
    other_doc = documents.ingest("orbit-secondary.pdf", BytesIO(alternate_bytes.getvalue()))
    question = "What is the launch date for Project Orbit?"
    quote = "The launch date for Project Orbit is 18 June 2027."
    generator = Mock(spec=AnswerGenerator)

    def generate(*, system_prompt, user_prompt):
        passages = json.loads(user_prompt)["retrieved_passages"]
        assert all(passage["filename"] == sample_doc.filename for passage in passages)
        source_id = next(passage["id"] for passage in passages if quote in passage["text"])
        return _generate_answer(quote, quote, source_id)

    generator.generate.side_effect = generate
    with TestClient(create_app(settings, documents=documents, generator=generator)) as client:
        created = _new_conversation(client, document_ids=[sample_doc.id])
        path = f"/api/conversations/{created['id']}"
        initial = client.post("/api/chat", json={"question": question, "conversation_id": created["id"]})
        assert initial.status_code == 200, initial.text
        before = initial.json()
        assert before["status"] == "answered"
        assert before["sources"][0]["document_id"] == sample_doc.id
        assert client.delete(f"/api/documents/{sample_doc.id}").status_code == 204
        restored = client.get(path).json()
        followup = client.post("/api/chat", json={"question": question, "conversation_id": created["id"]})
        assert followup.status_code == 200, followup.text
        updated = client.get(path).json()

    assert restored["document_ids"] == updated["document_ids"] == []
    assert restored["messages"] == updated["messages"][:2] == before["messages"]
    assert restored["messages"][1]["source_references"] == before["sources"]
    assert updated["messages"][3]["source_references"] == []
    assert followup.json()["status"] == "insufficient_context"
    assert followup.json()["sources"] == []
    generator.generate.assert_called_once()
    unscoped = documents.retrieve(question, None, top_k=4, min_similarity=0.30)
    assert unscoped and {chunk.document_id for chunk in unscoped} == {other_doc.id}

import json
import logging

import pytest
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from app.rag.chunking import Chunk
from app.rag.lexical import retrieval_question
from app.rag.vector_store import VectorStore


INCIDENT_QUESTION = (
    "Did incident LAB-731 meet the P1 restoration target? "
    "Explain the calculation and cite the relevant pages."
)


class ControlledEmbeddings(Embeddings):
    """Rank nearby VLAN 142 highest by cosine, with all rows below the cutoff."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 4.0] if "VLAN 142" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class DiagnosticEmbeddings(ControlledEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if text.startswith("Dense candidate") else vector
                for text, vector in zip(texts, super().embed_documents(texts), strict=True)]


def test_lexical_retrieval_rescues_exact_network_row_without_crossing_scope(tmp_path):
    store = VectorStore(
        directory=tmp_path / "chroma",
        model_name="controlled",
        embeddings=ControlledEmbeddings(),
    )
    store.add([
        Chunk("selected", "lab.pdf", 1, "selected:1:1", "VLAN 142 default gateway is 203.0.113.1."),
        Chunk("selected", "lab.pdf", 2, "selected:2:1", "VLAN 42 default gateway is 198.51.100.1."),
        Chunk("different", "other.pdf", 1, "different:1:1", "VLAN 42 default gateway is 192.0.2.8."),
    ])

    matches = store.retrieve(
        "What is the default gateway for VLAN 42?",
        document_ids=["selected"],
        top_k=1,
        min_similarity=0.3,
    )

    assert [(hit.chunk_id, hit.page_number) for hit in matches] == [("selected:2:1", 2)]
    assert matches[0].similarity == 0.0
    assert matches[0].lexical_score > 0


def test_lexical_retrieval_matches_arabic_and_rejects_unrelated_questions(tmp_path):
    store = VectorStore(
        directory=tmp_path / "chroma",
        model_name="controlled",
        embeddings=ControlledEmbeddings(),
    )
    store.add([
        Chunk("report", "report.pdf", 1, "report:1:1", "The headquarters are in Amman."),
        Chunk("report", "report.pdf", 2, "report:2:1", "يقع المقر الرئيسي في عمّان."),
    ])

    matches = store.retrieve(
        "أين يقع المقر الرئيسي؟", document_ids=["report"], top_k=2, min_similarity=0.3,
    )

    assert [hit.page_number for hit in matches] == [2]
    assert store.retrieve(
        "ما مساحة المستودع؟", document_ids=["report"], top_k=2, min_similarity=0.3,
    ) == []


def test_presentation_instructions_do_not_hide_cross_page_evidence(tmp_path):
    store = VectorStore(
        directory=tmp_path / "chroma",
        model_name="controlled",
        embeddings=ControlledEmbeddings(),
    )
    policy = "The P1 drill sets a service restoration target of 42 minutes."
    timeline = "Incident LAB-731, Priority P1. Incident began: 08:12 UTC. Service restored: 08:45 UTC."
    store.add([
        Chunk("handbook", "handbook.pdf", 1, "handbook:1:1", policy),
        Chunk("handbook", "handbook.pdf", 3, "handbook:3:1", timeline),
        Chunk("other", "another.pdf", 1, "other:1:1", policy),
    ])

    matches = store.retrieve(
        INCIDENT_QUESTION, document_ids=["handbook"], top_k=4, min_similarity=0.3,
    )

    assert {(hit.document_id, hit.page_number, hit.text) for hit in matches} == {
        ("handbook", 1, policy),
        ("handbook", 3, timeline),
    }


def test_retrieval_question_preserves_substantive_follow_up():
    factual = "Did the restoration meet the target? What caused the disruption?"
    detailed = "Did the restoration meet the target? Explain the calculation of the revised target."

    assert retrieval_question(factual) == factual
    assert retrieval_question(detailed) == detailed
    assert retrieval_question(INCIDENT_QUESTION) == "Did incident LAB-731 meet the P1 restoration target?"


@pytest.mark.parametrize("debug_enabled", [False, True])
def test_retrieval_trace_is_opt_in_scoped_and_redacted(tmp_path, caplog, debug_enabled):
    key = "configured-trace-key"
    token = "gsk_synthetic_test_token_123"
    query = f"What is the default gateway for VLAN 42 ({key})?"
    filename = f"lab-{key}-{token}.pdf"
    store = VectorStore(
        directory=tmp_path / "chroma",
        model_name="diagnostic",
        embeddings=DiagnosticEmbeddings(),
        debug_api_key=SecretStr(key) if debug_enabled else None,
    )
    store.add([
        Chunk("selected", filename, 1, "selected:1:1", "Dense candidate. Never log this document body."),
        Chunk("selected", filename, 2, "selected:2:1", "VLAN 142 default gateway is 203.0.113.1."),
        Chunk("selected", filename, 3, "selected:3:1", "VLAN 42 default gateway is 198.51.100.1."),
        Chunk("selected", filename, 4, "selected:4:1", "VLAN 42 default gateway is 198.51.100.2."),
        Chunk("excluded", "excluded.pdf", 3, "excluded:3:1", "VLAN 42 default gateway is 192.0.2.8."),
    ])

    with caplog.at_level(logging.INFO, logger="uvicorn.error.askmydocs.rag"):
        matches = store.retrieve(
            query + " Explain the calculation and cite the relevant pages.",
            document_ids=["selected"],
            top_k=2,
            min_similarity=0.3,
        )

    assert len(matches) == 2
    records = [record for record in caplog.records if record.msg.startswith("RAG retrieval")]
    if not debug_enabled:
        assert records == []
        return

    (record,) = records
    payload = json.loads(record.args[0])
    assert payload["effective_query"] == query.replace(key, "[REDACTED]")
    assert payload["document_ids"] == ["selected"]
    assert payload["top_k"] == 2
    assert payload["min_cosine_similarity"] == 0.3
    assert all(value not in caplog.text for value in (key, token, "excluded.pdf", "Never log this document body"))

    candidates = {entry["chunk_id"]: entry for entry in payload["candidates"]}
    assert len(candidates) == 4
    assert candidates["selected:1:1"]["cosine_distance"] == pytest.approx(0.0)
    assert candidates["selected:1:1"]["rank"] == 1
    assert candidates["selected:2:1"]["cosine_distance"] == pytest.approx(1 - 1 / 17**0.5)
    assert candidates["selected:2:1"]["rank"] is None
    assert candidates["selected:2:1"]["disposition"] == "below_threshold"
    assert {entry["disposition"] for entry in candidates.values()} == {
        "retained", "below_threshold", "dropped_by_top_k",
    }
    assert sorted(entry["rank"] for entry in candidates.values() if entry["disposition"] == "retained") == [1, 2]
    assert all(candidates[f"selected:{page}:1"]["lexical_score"] > 0 for page in (3, 4))

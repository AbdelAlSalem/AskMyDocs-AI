import hashlib
import json
from math import sqrt
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from app.rag import demo_retrieval
from app.rag.demo_embeddings import DemoQueryEncoder
from app.rag.demo_retrieval import CORPUS_PATH, EMBEDDINGS_PATH, DemoRetriever
from app.services.errors import DocumentError


def orbit_query_vector(page_one_weight: float, page_two_weight: float) -> list[float]:
    vectors = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))["vectors"]
    merged = [
        page_one_weight * first + page_two_weight * second
        for first, second in zip(vectors["project-orbit:1:1"], vectors["project-orbit:2:1"], strict=True)
    ]
    norm = sqrt(sum(value ** 2 for value in merged))
    return [value / norm for value in merged]


@pytest.mark.parametrize(
    ("question", "page", "evidence"),
    [
        ("What is the launch date for Project Orbit?", 1, "18 June 2027"),
        ("متى موعد إطلاق مشروع أوربت؟", 1, "18 June 2027"),
        ("How many volunteers are in the pilot team?", 1, "24 volunteers"),
        ("أين ستقام الورشة التجريبية؟", 2, "مكتبة النور في عمّان"),
        ("Where will the pilot workshop be held?", 2, "مكتبة النور في عمّان"),
        ("How long does the workshop last?", 2, "ساعتين"),
    ],
)
def test_historical_lexical_demo_retrieves_grounding_in_either_language(question, page, evidence):
    matches = DemoRetriever(use_semantic=False).retrieve(question, ["project-orbit"])

    assert matches[0].page_number == page
    assert matches[0].filename == "project-orbit.pdf"
    assert matches[0].chunk_id == f"project-orbit:{page}:1"
    assert evidence in matches[0].text


def test_historical_lexical_demo_preserves_cross_page_evidence_within_context_budget():
    question = "When does the project launch, and where is the workshop?"
    full = DemoRetriever(use_semantic=False).retrieve(question, top_k=2)
    assert [passage.page_number for passage in full] == [1, 2]

    budget = min(len(passage.text) for passage in full)
    limited = DemoRetriever(max_context_chars=budget, use_semantic=False).retrieve(question, top_k=2)
    assert len(limited) == 1
    assert limited[0] in full
    assert len(limited[0].text) <= budget


def test_historical_lexical_demo_summary_scope_and_absent_evidence():
    retriever = DemoRetriever(use_semantic=False)

    summary = retriever.retrieve("ما أهم النقاط في المستند؟")
    assert {passage.page_number for passage in summary} == {1, 2}
    assert retriever.retrieve("What is the pilot team's catering budget?") == []
    assert retriever.retrieve("What is the launch date?", []) == []

    with pytest.raises(DocumentError, match="no longer available"):
        retriever.retrieve("What is the launch date?", ["a-local-upload"])


def test_bundled_corpus_matches_the_approved_synthetic_file():
    retriever = DemoRetriever(use_semantic=False)
    (document,) = retriever.list_documents()
    (metadata,) = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))["documents"]
    source = Path(__file__).resolve().parents[2] / "samples" / "project-orbit.pdf"

    assert document.id == "project-orbit"
    assert document.filename == source.name
    assert document.page_count == 2
    assert document.chunk_count == 2
    assert document.status == "ready"
    assert metadata["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.mark.parametrize(("question", "page", "evidence", "weights"), [
    ("Which day is Orbit due to kick off?", 1, "18 June 2027", (1.0, 0.0)),
    ("أين ستُعقد الجلسة التدريبية؟", 2, "مكتبة النور في عمّان", (0.0, 1.0)),
], ids=["english-paraphrase", "arabic-paraphrase"])
def test_semantic_demo_ranks_the_matching_corpus_vector_without_lexical_overlap(question, page, evidence, weights):
    assert DemoRetriever(use_semantic=False).retrieve(question, ["project-orbit"]) == []
    vector = orbit_query_vector(*weights)
    retriever = DemoRetriever(embed_query=lambda _: vector)

    first, *_ = retriever.retrieve(question, ["project-orbit"])

    assert first.page_number == page
    assert first.similarity == pytest.approx(1.0)
    assert first.lexical_score == 0.0
    assert first.filename == "project-orbit.pdf"
    assert first.chunk_id == f"project-orbit:{page}:1"
    assert evidence in first.text


def test_semantic_demo_keeps_both_pages_under_context_budget():
    question = "When will Orbit kick off, and which venue will host the training?"
    vector = orbit_query_vector(1.0, 1.0)
    full = DemoRetriever(embed_query=lambda _: vector).retrieve(question, ["project-orbit"], top_k=2)
    assert {passage.page_number for passage in full} == {1, 2}
    assert all(passage.similarity > 0.30 for passage in full)

    budget = min(len(passage.text) for passage in full)
    limited = DemoRetriever(max_context_chars=budget, embed_query=lambda _: vector).retrieve(question, top_k=2)
    (passage,) = limited
    assert passage in full
    assert len(passage.text) <= budget


def test_semantic_demo_respects_scope_and_rejects_scores_below_cutoff():
    vector = orbit_query_vector(-1.0, -1.0)
    retriever = DemoRetriever(embed_query=lambda _: vector)
    question = "What is the pilot team's catering budget?"
    assert retriever.retrieve(question, ["project-orbit"]) == []
    assert retriever.retrieve(question, []) == []
    with pytest.raises(DocumentError, match="no longer available"):
        retriever.retrieve(question, ["a-local-upload"])


@pytest.mark.parametrize(("question", "alternate"), [
    ("How big are the replication logs on the staging server?", "How large are the replication logs on the staging server?"),
    ("Where can we go for the Monday safety briefing?", "Where is the Monday safety briefing?"),
])
def test_demo_retries_an_empty_paraphrase_search_without_dropping_the_subject(question, alternate):
    calls = []
    unrelated = orbit_query_vector(-1.0, -1.0)
    relevant = orbit_query_vector(1.0, 0.0)

    def embed(text):
        calls.append(text)
        return relevant if text == alternate else unrelated

    retriever = DemoRetriever(embed_query=embed)
    result = retriever.retrieve(question, ["project-orbit"], top_k=1)

    assert calls == [question, alternate]
    assert [passage.page_number for passage in result] == [1]
    assert result[0].lexical_score == 0.0


@pytest.mark.parametrize(("question", "weights", "expected_pages"), [
    ("How big is the accounting department?", (1.0, 0.0), [1]),
    ("What is the accounting budget?", (-1.0, -1.0), []),
])
def test_demo_keeps_the_first_search_when_it_finds_evidence_or_has_no_alternate(question, weights, expected_pages):
    calls = []
    vector = orbit_query_vector(*weights)

    def embed(text):
        calls.append(text)
        return vector

    result = DemoRetriever(embed_query=embed).retrieve(question, ["project-orbit"], top_k=1)

    assert [passage.page_number for passage in result] == expected_pages
    assert calls == [question]


def test_semantic_demo_requires_its_precomputed_vectors(monkeypatch, tmp_path):
    monkeypatch.setattr(demo_retrieval, "EMBEDDINGS_PATH", tmp_path / "unavailable.json")
    with pytest.raises(RuntimeError, match="Demo vectors are missing"):
        DemoRetriever(embed_query=lambda _: [])


def test_demo_sentencepiece_maps_xlmr_ids_and_preserves_eos_when_truncated():
    pieces = list(range(4, 140))
    processor = Mock()
    processor.unk_id.return_value = 0
    processor.encode_as_ids.side_effect = [[9, 0], [17], pieces]
    encoder = object.__new__(DemoQueryEncoder)
    encoder._tokenizer = processor
    encoder._special_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 250001}

    assert encoder._token_ids(" \tمرحبا\n<pad><unk><mask><s></s> mundo  ") == [
        0, 10, 3, 1, 3, 250001, 0, 2, 18, 2,
    ]
    assert encoder._token_ids("a long sentence") == [0, *range(5, 131), 2]
    assert processor.encode_as_ids.call_args_list == [
        call("مرحبا "), call(" mundo"), call("a long sentence"),
    ]

import json
from io import BytesIO
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.models.schemas import ChatRequest, HistoryMessage
from app.rag import demo_retrieval, vector_store
from app.rag.answer import AnswerService
from app.rag.demo_embeddings import EMBEDDING_DIMENSIONS, MODEL_ID, MODEL_REVISION, passage_fingerprint
from app.rag.demo_retrieval import DemoRetriever
from app.rag.document_wide import document_intent
from app.rag.vector_store import VectorStore
from app.services.documents import DocumentService
from app.services.groq_client import AnswerGenerator, GeneratedAnswer, ProviderError
from test_rag_regressions import _pdf


def _mock_generator(claim_factory):
    generator = Mock(spec=AnswerGenerator)

    def generate(*, system_prompt, user_prompt, summary=False):
        passages = json.loads(user_prompt)["retrieved_passages"]
        return GeneratedAnswer(status="answered", claims=claim_factory(passages))

    generator.generate.side_effect = generate
    return generator


def _claim(passage, text=None):
    return {
        "text": passage["text"] if text is None else text,
        "evidence": [{"source_id": passage["id"], "quote": passage["text"]}],
    }


def _supplied(generator):
    return json.loads(generator.generate.call_args.kwargs["user_prompt"])["retrieved_passages"]


@pytest.fixture
def demo_corpus(monkeypatch, tmp_path):
    def create(documents, *, max_context_chars=4500):
        metadata = []
        passages = []
        for document_id, pages in documents.items():
            metadata.append({
                "id": document_id,
                "filename": f"{document_id}.pdf",
                "page_count": len(pages),
                "chunk_count": len(pages),
                "size_bytes": 1200,
                "uploaded_at": "2026-01-01T00:00:00Z",
            })
            passages.extend(
                {
                    "document_id": document_id,
                    "chunk_id": f"{document_id}:{page_number}:1",
                    "page_number": page_number,
                    "text": text,
                }
                for page_number, text in enumerate(pages, start=1)
            )
        corpus_path = tmp_path / "corpus.json"
        index_path = tmp_path / "embeddings.json"
        corpus_path.write_text(json.dumps({"documents": metadata, "passages": passages}), encoding="utf-8")
        index_path.write_text(json.dumps({
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "passage_sha256": passage_fingerprint(passages),
            "vectors": {
                passage["chunk_id"]: [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)
                for passage in passages
            },
        }), encoding="utf-8")
        monkeypatch.setattr(demo_retrieval, "CORPUS_PATH", corpus_path)
        monkeypatch.setattr(demo_retrieval, "EMBEDDINGS_PATH", index_path)
        embed = Mock(side_effect=AssertionError("document-wide questions must not embed the query"))
        return DemoRetriever(max_context_chars=max_context_chars, embed_query=embed)

    return create


def test_local_summary_reads_all_small_selected_pages_from_chroma_without_query_embedding(make_settings, monkeypatch):
    settings = make_settings(rag_min_similarity=0.99, rag_top_k=1)
    embeddings = Mock(spec=Embeddings)
    embeddings.embed_documents.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    embeddings.embed_query.side_effect = AssertionError("the summary must read the stored passages")
    monkeypatch.setattr(vector_store, "LEXICAL_BATCH_SIZE", 2)
    store = VectorStore(directory=settings.chroma_dir, model_name=settings.embedding_model, embeddings=embeddings)
    documents = DocumentService(upload_dir=settings.upload_dir, sqlite_path=settings.sqlite_path, vectors=store)
    research = documents.ingest("harbor.pdf", BytesIO(_pdf(
        [(72, 730, "The survey mapped bicycle traffic around the harbor.")],
        [(72, 730, "Better lighting improved perceived safety.")],
        [(72, 730, "The report recommends upgrades to waterfront crossings.")],
    )))
    rejected = documents.ingest("other.pdf", BytesIO(_pdf(
        [(72, 730, "The railway audit concerns freight schedules.")],
    )))
    appendix = documents.ingest("appendix.pdf", BytesIO(_pdf(
        [(72, 730, "The appendix lists accessibility review procedures.")],
    )))
    generator = _mock_generator(lambda passages: [_claim(passage) for passage in passages])
    request = ChatRequest(
        question="Summarize the main points of these PDFs.", document_ids=[research.id, appendix.id],
    )

    answer = AnswerService(documents, generator, settings).answer(request)

    assert answer.status == "answered"
    assert [(p["filename"], p["page_number"]) for p in _supplied(generator)] == [
        ("harbor.pdf", 1), ("harbor.pdf", 2), ("harbor.pdf", 3), ("appendix.pdf", 1),
    ]
    assert [(source.document_id, source.page_number, source.chunk_id) for source in answer.sources] == [
        (document_id, page, f"{document_id}:{page}:1")
        for document_id, page in [(research.id, 1), (research.id, 2), (research.id, 3), (appendix.id, 1)]
    ]
    assert rejected.id not in {source.document_id for source in answer.sources}
    generator.generate.assert_called_once()
    embeddings.embed_query.assert_not_called()


def test_demo_short_arabic_summary_keeps_source_language_and_metadata(demo_corpus):
    original = "The shelter relocated its supply desk to the east entrance."
    translated = "نقل الملجأ مكتب الإمدادات إلى المدخل الشرقي."
    documents = demo_corpus({"shelter": (original,)})
    generator = _mock_generator(lambda passages: [_claim(passages[0], translated)])

    answer = AnswerService(documents, generator, Settings()).answer(ChatRequest(
        question="ما أهم النقاط في المستند؟", document_ids=["shelter"],
    ))

    assert answer.answer == f"{translated} [1]"
    assert [(source.filename, source.page_number, source.snippet) for source in answer.sources] == [
        ("shelter.pdf", 1, original),
    ]
    assert "including Arabic" in generator.generate.call_args.kwargs["system_prompt"]


def test_page_summary_sends_only_requested_evidence_and_no_prior_answers(demo_corpus):
    overview = "Findings: The trial improved delivery reliability. The authors recommend the new route."
    documents = demo_corpus({
        "trial": (overview, "The appendix describes a separate procurement exercise."),
        "another": ("An unselected report contains unrelated findings.",),
    })
    generator = _mock_generator(lambda passages: [
        _claim(passages[0], "- The trial improved delivery reliability."),
        _claim(passages[0], "- The authors recommend the new route."),
    ])
    question = "Summarize only page 1 of the selected document in three bullet points"
    answers = AnswerService(documents, generator, Settings())

    answer = answers.answer(ChatRequest(
        question=question,
        document_ids=["trial"],
        history=[HistoryMessage(role="assistant", content="Incidental information in an earlier answer.")],
    ))

    generator.generate.assert_called_once()
    call = generator.generate.call_args.kwargs
    assert call["summary"] is True
    assert "central topics" in call["system_prompt"] and "publication details" in call["system_prompt"]
    prompt = json.loads(call["user_prompt"])
    assert prompt["question"] == question
    assert prompt["conversation_history"] == []
    assert [(p["filename"], p["page_number"], p["text"]) for p in prompt["retrieved_passages"]] == [
        ("trial.pdf", 1, overview),
    ]
    assert answer.status == "answered"
    assert answer.answer.count("[1]") == 2
    assert [(source.document_id, source.page_number) for source in answer.sources] == [("trial", 1)]

    generator.generate.reset_mock()
    absent = answers.answer(ChatRequest(question="Summarize page 9.", document_ids=["trial"]))
    assert absent.status == "insufficient_context" and "requested pages" in absent.answer
    generator.generate.assert_not_called()


@pytest.mark.parametrize("invalid_citation", [False, True])
def test_unusable_summary_output_returns_only_cited_excerpts_without_retry(demo_corpus, invalid_citation):
    finding = "Findings: The trial improved delivery reliability across the monitored routes."
    documents = demo_corpus({
        "trial": (f"Version: published by the document control office.\n{finding}", "The appendix is unrelated."),
    })
    generator = Mock(spec=AnswerGenerator)
    if invalid_citation:
        generator.generate.return_value = GeneratedAnswer(status="answered", claims=[{
            "text": "An unverified model response.",
            "evidence": [{"source_id": 1, "quote": "A quote that is absent from the source."}],
        }])
    else:
        generator.generate.side_effect = ProviderError(
            502, "groq_output_limit", "Output limit reached.", output_failure=True,
        )
    answer = AnswerService(documents, generator, Settings()).answer(ChatRequest(question="Summarize page 1."))

    generator.generate.assert_called_once()
    assert answer.status == "answered"
    assert "cited excerpts" in answer.answer and finding in answer.answer
    assert all(text not in answer.answer for text in ("Version:", "appendix", "unverified model response"))
    assert [(source.page_number, source.snippet) for source in answer.sources] == [(1, finding)]


def test_summary_rate_limit_keeps_provider_error_instead_of_fallback(demo_corpus):
    documents = demo_corpus({"trial": ("The trial improved delivery reliability across the monitored routes.",)})
    generator = Mock(spec=AnswerGenerator)
    generator.generate.side_effect = ProviderError(429, "groq_rate_limit", "Slow down.", retry_after=10)

    with pytest.raises(ProviderError) as failure:
        AnswerService(documents, generator, Settings()).answer(ChatRequest(question="Summarize the main points."))

    assert (failure.value.status_code, failure.value.retry_after) == (429, 10)
    generator.generate.assert_called_once()


def test_demo_long_overview_uses_bounded_passages_from_across_the_document(demo_corpus):
    page_count = 36
    budget = 2300
    pages = tuple(
        f"Field report number {page}. " + "Monitoring teams review weather and wildlife along the route. " * 4
        for page in range(1, page_count + 1)
    )
    documents = demo_corpus({"monitoring": pages}, max_context_chars=budget)
    generator = _mock_generator(lambda passages: [
        _claim(passage, "The report covers wildlife monitoring.") for passage in passages
    ])

    answer = AnswerService(documents, generator, Settings()).answer(ChatRequest(question="Give me an overview."))

    generator.generate.assert_called_once()
    supplied = _supplied(generator)
    assert 2 < len(supplied) < page_count
    assert sum(len(passage["text"]) for passage in supplied) <= budget
    page_numbers = {passage["page_number"] for passage in supplied}
    assert min(page_numbers) <= page_count // 3
    assert max(page_numbers) > page_count * 2 // 3
    assert {source.page_number for source in answer.sources} == page_numbers
    assert "selected excerpts" in answer.answer


@pytest.mark.parametrize(("question", "source_pages"), [
    ("Are any dates or deadlines mentioned?", [2, 3]),
    ("Are there important deadlines?", [2]),
    ("ما هي التواريخ المهمة؟", [2, 3]),
])
def test_date_intent_prioritizes_real_dates_and_differentiates_deadlines(demo_corpus, question, source_pages):
    documents = demo_corpus({"trial": (
        "Unit 71 of the router fleet uses firmware version 1.10.2026. The retry delay is 20 minutes.",
        "آخر موعد للتقديم هو ١٥ آذار ٢٠٢٨.",
        "The public trial opens on 16 September 2027 at 09:30.",
    )})
    generator = _mock_generator(lambda passages: [
        _claim(passage, "يرد الموعد في المستند." if question.endswith("؟") else None) for passage in passages
    ])

    answer = AnswerService(documents, generator, Settings()).answer(ChatRequest(question=question))

    assert answer.status == "answered"
    assert [passage["page_number"] for passage in _supplied(generator)] == source_pages
    assert [source.page_number for source in answer.sources] == source_pages


def test_dates_absent_and_no_documents_selected_skip_generation(demo_corpus):
    documents = demo_corpus({"equipment": (
        "The router address is 198.51.100.1. The delay is 20 minutes. Firmware version 1.10.2026 was installed.",
    )})
    generator = Mock(spec=AnswerGenerator)
    answers = AnswerService(documents, generator, Settings())

    dated = answers.answer(ChatRequest(question="What dates should I know?", document_ids=["equipment"]))
    assert dated.status == "insufficient_context"
    assert "dates or times" in dated.answer
    assert not dated.sources
    none = answers.answer(ChatRequest(question="Summarize the main points.", document_ids=[]))
    assert none.status == "insufficient_context"
    assert not none.sources
    generator.generate.assert_not_called()


def test_date_answer_cannot_use_an_unrelated_number_as_a_day(demo_corpus):
    event = "The rollout starts on 14 September 2027."
    identifier = "The next item concerns rack 23."
    documents = demo_corpus({"deployment": (f"{event} {identifier}",)})
    generator = _mock_generator(lambda passages: [{
        "text": "The rollout starts on 23 September 2027.",
        "evidence": [
            {"source_id": passages[0]["id"], "quote": event},
            {"source_id": passages[0]["id"], "quote": identifier},
        ],
    }])

    answer = AnswerService(documents, generator, Settings()).answer(
        ChatRequest(question="What dates should I know?"),
    )

    generator.generate.assert_called_once()
    assert answer.status == "insufficient_context"
    assert not answer.sources


@pytest.mark.parametrize("question", [
    "Are any dates or deadlines mentioned?", "What are the dates?", "what the dates?",
    "What important dates are mentioned, and what does each one represent?",
])
def test_date_answer_pairs_each_value_with_verified_event_instead_of_bare_claims(demo_corpus, question):
    documents = demo_corpus({"manual": (
        "Workshop Notes | Revision B | 7 April 2028\nThe cooperative was founded in 2018.",
        "2. Incident VM-77\nResolved | Priority P1 | 16 March 2028\nService restored: 08:45 UTC",
    )})
    def short_claims(passages):
        by_page = {p["page_number"]: p["id"] for p in passages}
        return [
            {"text": text, "evidence": [{"source_id": by_page[page], "quote": quote}]}
            for text, page, quote in (
                ("The publication deadline was 7 April 2028.", 1, "7 April 2028"),
                ("Yes, dates are mentioned.", 1, "The cooperative was founded in 2018."),
                ("16 March 2028", 2, "16 March 2028"),
            )
        ]
    generator = _mock_generator(short_claims)
    answer = AnswerService(documents, generator, Settings()).answer(
        ChatRequest(question=question, document_ids=["manual"], history=[]),
    )
    instructions = generator.generate.call_args.kwargs["system_prompt"]
    assert '"source_id":1,"quote":"exact source words"' in instructions
    assert "untrusted" in instructions and len(instructions) < 1500
    assert answer.status == "answered"
    assert "Revision B | 7 April 2028" in answer.answer
    assert "cooperative was founded in 2018" in answer.answer
    assert "Incident VM-77" in answer.answer and "16 March 2028" in answer.answer
    assert "publication" not in answer.answer and "Yes, dates" not in answer.answer
    assert [(source.document_id, source.page_number) for source in answer.sources] == [("manual", 1), ("manual", 2)]


@pytest.mark.parametrize(("question", "intent"), [
    ("What is this PDF about?", "summary"),
    ("What are the main topics discussed?", "summary"),
    ("What are the key conclusions?", "summary"),
    ("لخّص هذا الملف.", "summary"),
    ("List the dates mentioned in the document.", "dates"),
    ("When is the deadline to submit the lab report?", None),
    ("Summarize the findings about one ward.", None),
    ("Did the repair meet the restoration target?", None),
])
def test_document_intents_distinguish_whole_document_questions(question, intent):
    assert document_intent(question) == intent

import pytest
from langchain_core.embeddings import Embeddings

from app.rag.answer import _cited_answer
from app.rag.chunking import Chunk
from app.rag.lexical import comparison_queries, retrieval_question
from app.rag.types import RetrievedChunk
from app.rag.vector_store import VectorStore
from app.services.groq_client import GeneratedAnswer


POLICY = "Priority 1 (P1) drills: target restoration within 42 minutes of the incident start."
DURATION = "Recorded outage duration: 33 minutes"
QUESTION = "Did incident LAB-731 meet the P1 restoration target? Explain the calculation and cite the relevant pages."


def verify(text, sources, quotes):
    chunks = [
        RetrievedChunk("chosen", "operations.pdf", page, f"chosen:{page}:1", source, 0.9)
        for page, source in sources
    ]
    result = GeneratedAnswer(status="answered", claims=[{
        "text": text, "evidence": [{"source_id": index, "quote": quote} for index, quote in enumerate(quotes, 1)],
    }])
    return _cited_answer(result, chunks)


def test_comparison_verifies_copied_pdf_labels_and_dimensional_equation():
    # The model added a terminal period to a PDF label, used nonbreaking
    # spaces and a typographic identifier hyphen, and kept units in the math.
    text = "LAB‑731 met the P1 target by 9 minutes (42 minutes – 33 minutes = 9 minutes)."
    result = verify(text, [(1, POLICY), (3, f"Incident LAB-731\n{DURATION}\nSymptoms")], [POLICY, DURATION + "."])
    assert result is not None
    assert [source.page_number for source in result.sources] == [1, 3]
    assert result.sources[1].snippet == DURATION


@pytest.mark.parametrize("text", [
    "It missed the restoration target by 9 minutes (42 minutes - 33 minutes = 9 minutes).",
    "It met the restoration target by 9 hours (42 hours - 33 hours = 9 hours).",
    "It met the restoration target by 9 minutes (42 minutes - 33 seconds = 9 minutes).",
    "It met the restoration target by 4 minutes (42 minutes - 33 minutes = 4 minutes).",
])
def test_citations_cannot_authorize_wrong_verdict_units_or_math(text):
    assert verify(text, [(1, POLICY), (3, DURATION)], [POLICY, DURATION]) is None


def test_natural_language_margin_requires_both_labeled_durations():
    text = "LAB-731 met the restoration target by 9 minutes."
    assert verify(text, [(1, POLICY), (3, DURATION)], [POLICY, DURATION]) is not None
    assert verify(text, [(1, POLICY)], [POLICY]) is None
    assert verify(text, [(3, DURATION)], [DURATION]) is None


@pytest.mark.parametrize("range_text", ["07:30-16:45", "07:30 -16:45", "07:30 ‑ 16:45"])
def test_arabic_conjunction_checks_the_full_clock_value_against_both_range_endpoints(range_text):
    source = f"Support hours: {range_text}. Initial response within 4 hours."
    assert verify("ساعات العمل بين ٠٧:٣٠ و١٦:٤٥ والاستجابة بـ٤ ساعات.", [(1, source)], [source])
    assert verify("ساعات العمل بين ٠٧:٣٠ و٢٠:٠٠ والاستجابة بـ٤ ساعات.", [(1, source)], [source]) is None
    assert verify("ساعات العمل بين ٠٧:٣٠ و١٦:٥٠.", [(1, source)], [source]) is None


def test_a_matching_number_cannot_substitute_another_identifier_or_time_zone():
    source = "LAB-731 was a P1 incident, recorded at 09:00 EET (UTC+2)."
    assert verify("LAB‑731 was recorded at 09:00 UTC+2.", [(1, source)], [source])
    assert verify("LAB‑431 was recorded at 09:00 UTC+2.", [(1, source)], [source]) is None
    assert verify("LAB‑731 was recorded at 09:00 UTC−2.", [(1, source)], [source]) is None


@pytest.mark.parametrize(("source", "bad_quote"), [
    ("Recorded outage duration: 33 minutes longer than the target", DURATION + "."),
    ("Gateway: 198.51.100.1", "Gateway: 198.51.100"),
    ("Gateway: 198.51.100.1", "51.100.1"),
    ("Threshold: -33 minutes", "33 minutes"),
    ("VLAN 42 | 198.51.100.1\nVLAN 30 | 192.0.2.1", "VLAN 42 | 192.0.2.1"),
])
def test_quote_verification_rejects_partial_addresses_qualifiers_and_other_rows(source, bad_quote):
    assert verify("The value is listed.", [(1, source)], [bad_quote]) is None


class CrowdedEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0, 1.0] if "policy" in text else [1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [0.0, 1.0] if text == "first response target" else [1.0, 0.0]


def test_comparison_finds_policy_when_incident_hits_fill_the_original_top_k(tmp_path):
    question = "Did ticket TKT-841 meet the first response target?"
    assert comparison_queries(retrieval_question(question)) == ("first response target", "ticket TKT-841")
    store = VectorStore(directory=tmp_path, model_name="crowded", embeddings=CrowdedEmbeddings())
    chunks = [
        Chunk("selected", "runbook.pdf", 1, "selected:1", "The policy requires an initial reply within 30 minutes."),
        Chunk("other", "other.pdf", 1, "other:1", "The policy permits an initial reply within 90 minutes."),
        *[
            Chunk("selected", "runbook.pdf", 3, f"selected:3:{i}",
                  f"Ticket TKT-841 first response recorded after 26 minutes. Log entry {i}.")
            for i in range(4)
        ],
    ]
    store.add(chunks)
    result = store.retrieve(question, document_ids=["selected"], top_k=2, min_similarity=0.3)
    assert {item.document_id for item in result} == {"selected"}
    assert {item.page_number for item in result} == {1, 3}
    assert len(result) == 2

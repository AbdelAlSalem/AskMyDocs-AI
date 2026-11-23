import json

import pytest

from app.rag.document_wide import document_coverage, document_intent, summary_page_scope
from app.rag.prompt import build_prompt, prepare_summary_prompt
from app.rag.types import RetrievedChunk
from app.services.groq_client import GeneratedSummary, ProviderError


@pytest.mark.parametrize(("question", "expected"), [
    ("Summarize only page 1 of the selected document in three bullet points", ((1, 1),)),
    ("Please summarize pages 2–4 and 8 in 4 bullets, and cite the sources.", ((2, 4), (8, 8))),
    ("What are the main points on pages 1, 3, and 5?", ((1, 1), (3, 3), (5, 5))),
    ("لخّص الصفحة ١ فقط من الملف في ثلاث نقاط.", ((1, 1),)),
    ("لخص الصفحات ٢ إلى ٤ و٦ في ٣ نقاط، مع ذكر المصادر.", ((2, 4), (6, 6))),
    ("Summarize the main points in three bullet points.", None),
    ("Summarize the main points of the entire document in five bullet points, with page citations.", None),
    ("Give me a brief overview of the whole report with page citations in four bullets.", None),
    ("لخص أهم المعلومات الموجودة في المستند كاملًا باللغة العربية.", None),
    ("لخص الملفات كامله في خمس نقاط بالعربية.", None),
])
def test_summary_formatting_and_explicit_page_scope(question, expected):
    assert document_intent(question) == "summary"
    assert summary_page_scope(question) == expected


@pytest.mark.parametrize("question", [
    "What is the maintenance limit mentioned on page 1?",
    "Summarize the findings about one ward.",
    "Summarize the key findings about patients on one ward in five bullets.",
    "لخص المعلومات عن موعد الرحلة باللغة العربية.",
    "ما الموعد المذكور في الصفحة ١؟",
])
def test_specific_facts_keep_their_normal_retrieval_path(question):
    assert document_intent(question) is None
    assert summary_page_scope(question) is None


def _passage(document_id, page, index, text):
    return RetrievedChunk(document_id, f"{document_id}.pdf", page, f"{document_id}:{page}:{index}", text, 0.0)


def test_small_summary_keeps_complete_passages_from_requested_pages_only():
    passages = [
        _passage("alpha", 1, 2, "The findings explain improved reliability."),
        _passage("alpha", 1, 1, "The research describes pump designs."),
        _passage("alpha", 2, 1, "A separate appendix describes calibration."),
        _passage("beta", 1, 1, "The report recommends a new delivery route."),
    ]

    coverage = document_coverage(
        passages, {"alpha": 2, "beta": 1}, "summary", 4500, page_scope=((1, 1),),
    )

    assert coverage.has_documents
    assert not coverage.partial
    assert coverage.chunks == [passages[1], passages[0], passages[3]]


def test_missing_requested_page_returns_empty_evidence():
    cover = document_coverage(
        [_passage("alpha", 1, 1, "An available but unrequested page.")],
        {"alpha": 1},
        "summary",
        4500,
        page_scope=((55, 55),),
    )
    assert cover.has_documents
    assert not cover.chunks


def test_summary_budget_and_citation_coverage_across_same_named_documents():
    passages = [
        RetrievedChunk(document, "Report.pdf", 1, f"{document}:1:{i}",
                       f"The report describes the design of station {i}. The design is ready for the next trial.", 0.0)
        for document in ("alpha", "beta") for i in range(6)
    ]
    wire, evidence = prepare_summary_prompt(build_prompt(
        "Summarize the documents in five bullet points.", [], passages, summary=True,
    ))
    payload = json.loads(wire)
    assert payload["cover_each_page"] and payload["point_count"] == 5
    assert len(evidence.required_pages) == 2
    assert all(len(page["evidence"]) <= 6 for page in payload["retrieved_passages"])
    for source, span in evidence.quotes.values():
        assert span in passages[source - 1].text

    first_page, second_page = evidence.required_pages
    on_first = [
        {"text": evidence.quotes[anchor][1], "evidence_ids": [anchor]} for anchor in sorted(first_page)[:5]
    ]
    on_second = {"text": "The next trial is planned.", "evidence_ids": [min(second_page)]}
    for incomplete in (on_first, [on_first[0], on_second]):
        with pytest.raises(ProviderError) as failure:
            GeneratedSummary(points=incomplete).as_answer(evidence)
        assert failure.value.code == "groq_unverified_evidence"
    assert len(GeneratedSummary(points=on_first[:4] + [on_second]).as_answer(evidence).claims) == 5


def test_summary_checks_arabic_prose_formats_bullets_and_preserves_contiguous_evidence():
    overlap = "a small regional office in the harbor."
    passages = [
        _passage("pilot", 1, 1, f"The main office is downtown. The group also has {overlap}"),
        _passage("pilot", 1, 2, f"{overlap} The first trial lasted 55 minutes. Its limit was 60 minutes."),
    ]
    wire, evidence = prepare_summary_prompt(build_prompt("لخص الملف في نقاط باللغة العربية.", [], passages, summary=True))
    assert json.loads(wire)["summary_language"] == "Arabic"
    assert evidence.quotes[2] == (2, "The first trial lasted 55 minutes. Its limit was 60 minutes.")
    with pytest.raises(ProviderError, match="language"):
        GeneratedSummary(points=[{"text": "The trial took 55 minutes.", "evidence_ids": [2]}]).as_answer(evidence)
    summary = GeneratedSummary(points=[{"text": "استغرقت التجربة 55 دقيقة.", "evidence_ids": [2]}]).as_answer(evidence)
    assert summary.claims[0].text == "- استغرقت التجربة 55 دقيقة."


def test_arabic_summary_keeps_restoration_distinct_from_service_modification():
    question = "لخص الملف باللغة العربية."
    points = [
        {"text": "وتُعَدَّل الخدمة بعد إصلاح العطل.", "evidence_ids": [1]},
        {"text": "استُعيدت الخدمة بعد إصلاح العطل.", "evidence_ids": [1]},
    ]
    passage = _passage("stations", 2, 1, "Service was restored after the fault was repaired.")
    _, evidence = prepare_summary_prompt(build_prompt(question, [], [passage], summary=True))
    verified = GeneratedSummary(points=points).as_answer(evidence)
    assert [p.text for p in verified.claims] == [points[1]["text"]]

    both = _passage("stations", 2, 2, "Service was modified, and service was restored after the fault was repaired.")
    _, evidence = prepare_summary_prompt(build_prompt(question, [], [both], summary=True))
    assert len(GeneratedSummary(points=points).as_answer(evidence).claims) == 2


def test_over_budget_page_keeps_findings_and_separate_parts_with_verbatim_provenance():
    passages = [
        _passage("alpha", 1, 0, "Copyright Example House. All rights reserved."),
        *[
            _passage("alpha", 1, index, f"Section {index}: The process monitors usage and records equipment status.")
            for index in range(1, 30)
        ],
        _passage("alpha", 1, 30, "Version: updated after publication and prepress distribution."),
        _passage("alpha", 2, 1, "The separate appendix is unrequested."),
    ]
    passages[15] = _passage(
        "alpha", 1, 15, "Findings: A revised filter reduced contamination. The report recommends replacing worn filters.",
    )

    coverage = document_coverage(passages, {"alpha": 2}, "summary", 900, page_scope=((1, 1),))

    assert coverage.partial
    assert sum(len(chunk.text) for chunk in coverage.chunks) <= 900
    assert coverage.chunks == [passages[1], passages[15], passages[29]]

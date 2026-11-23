import json
import re
from dataclasses import dataclass
from io import BytesIO
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.core.config import Settings
from app.main import create_app
from app.rag.vector_store import RetrievedChunk, VectorStore
from app.services.catalog import DocumentRecord
from app.services.documents import DocumentService
from app.services.groq_client import AnswerGenerator, GeneratedAnswer


FILENAME = "synthetic-lantern.pdf"
FOUNDING_QUESTION = "What year was Lantern Works founded, and where is its headquarters?"
STAFF_QUESTION = "How many employees work in lighting design and stage operations?"
GATEWAY_QUESTION = "What is the default gateway for VLAN 42?"
INCIDENT_QUESTION = "Did incident LAB-731 meet the P1 restoration target, and by how many minutes?"
CALCULATION_QUESTION = (
    "Did incident LAB-731 meet the P1 restoration target? Explain the calculation and cite the relevant pages."
)
POLICY = "Priority 1 (P1) incidents have a restoration target of 42 minutes from the start."
INCIDENT_DURATION = "Restoration took 33 minutes."


def _pdf(*pages: list[tuple[int, int, str] | tuple[int, int, str, int]]) -> bytes:
    writer = PdfWriter()
    for cells in pages:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        operations = ["BT"]
        for x, y, value, *font_sizes in cells:
            font_size = font_sizes[0] if font_sizes else 11
            escaped = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            operations.append(f"/F1 {font_size} Tf 1 0 0 1 {x} {y} Tm ({escaped}) Tj")
        operations.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(operations).encode("ascii"))
        page.replace_contents(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _handbook(*, include_margin: bool = True, include_policy: bool = True) -> bytes:
    """Construct a fictional workshop handbook entirely in memory."""
    company = [
        (72, 730, "Lantern Works"),
        (72, 705, "Lantern Workshop Notes | 7 April 2028"),
        (72, 673, "Lantern Works was founded in 2018."),
        (72, 655, "Its headquarters is in Galway, Ireland."),
        (72, 607, "15 employees", 18),
        (72, 592, "Lighting design"),
        (340, 607, "11 employees", 18),
        (340, 592, "Box office"),
        (72, 558, "8 employees", 18),
        (72, 543, "Stage operations"),
        (340, 558, "11 employees", 18),
        (340, 543, "Wardrobe"),
    ]
    if include_policy:
        company.append((72, 495, POLICY))
    network = [(72, 730, "Lab network reference")]
    for y, cells in [
        (690, ("Segment", "Network", "Default gateway", "Purpose")),
        (668, ("VLAN 17", "192.0.2.0/24", "192.0.2.1", "Lighting consoles")),
        (646, ("VLAN 42", "198.51.100.0/24", "198.51.100.1", "Ticket printers")),
        (624, ("VLAN 69", "203.0.113.0/24", "203.0.113.1", "Lobby displays")),
    ]:
        network.extend((x, y, text) for x, text in zip((72, 175, 300, 415), cells, strict=True))
    incident = [
        (72, 730, "Incident LAB-731"),
        (72, 708, "Resolved | Priority P1 | 16 March 2028"),
        (72, 670, "The incident began at 08:12 UTC."),
        (72, 650, "Service was restored at 08:45 UTC."),
        (72, 628, "Restoration took 33 minutes, with a margin of 9 minutes." if include_margin else INCIDENT_DURATION),
    ]
    glossary = [(72, 730, "P2 tickets have a standard response goal of three business hours.")]
    return _pdf(company, network, incident, glossary)


class HandbookEmbeddings(Embeddings):
    """Keep exact gateway evidence below the dense similarity threshold."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if "198.51.100." in text:
                vectors.append([0.0, 0.2, 0.0, 0.98])
            elif "LAB-731" in text:
                vectors.append([0.0, 0.0, 1.0, 0.0])
            elif "Lantern Works" in text or "employees" in text or "restoration target" in text:
                vectors.append([0.8, 0.0, 0.6, 0.0])
            else:
                vectors.append([0.0, 0.0, 0.0, 1.0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if "VLAN 42" in text:
            return [0.0, 1.0, 0.0, 0.0]
        if "LAB-731" in text:
            return [0.0, 0.0, 1.0, 0.0]
        return [1.0, 0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def indexed_workspace(tmp_path_factory):
    directory = tmp_path_factory.mktemp("rag-regression")
    settings = Settings(
        upload_dir=directory / "uploads",
        sqlite_path=directory / "documents.sqlite3",
        chroma_dir=directory / "chroma",
        conversations_path=directory / "conversations.sqlite3",
    )
    documents = DocumentService(
        upload_dir=settings.upload_dir,
        sqlite_path=settings.sqlite_path,
        vectors=VectorStore(
            directory=settings.chroma_dir,
            model_name=settings.embedding_model,
            embeddings=HandbookEmbeddings(),
        ),
    )
    with TestClient(create_app(settings, documents=documents)) as client:
        upload = client.post("/api/documents/upload", files=[
            ("files", (FILENAME, _handbook(), "application/pdf")),
            ("files", ("out-of-scope.pdf", _pdf([
                (72, 720, "Lantern Works was founded in 1999."),
                (72, 700, "Its headquarters is in Rome, Italy."),
                (72, 680, "The default gateway for VLAN 42 is 198.51.100.254."),
            ]), "application/pdf")),
        ])
    assert upload.status_code == 200, upload.text
    handbook, distractor = upload.json()["documents"]
    assert handbook["page_count"] == 4
    assert handbook["id"] != distractor["id"]
    return settings, documents, handbook["id"]


def _evidence(passages, page, pattern):
    for passage in passages:
        if passage["page_number"] == page:
            match = re.search(pattern, passage["text"])
            if match:
                return {"source_id": passage["id"], "quote": match.group()}
    pytest.fail(f"Missing page {page} evidence matching {pattern!r} in prompt {passages!r}")


_CARD_SEPARATOR = r"[ \t|:–—-]+"
_LIGHTING_DESIGN_CARD = (
    rf"\b15 employees{_CARD_SEPARATOR}Lighting design|Lighting design{_CARD_SEPARATOR}15 employees\b"
)
_STAGE_OPERATIONS_CARD = (
    rf"\b8 employees{_CARD_SEPARATOR}Stage operations|Stage operations{_CARD_SEPARATOR}8 employees\b"
)


@pytest.mark.parametrize(("question", "claims"), [
    (FOUNDING_QUESTION, [
        ("Lantern Works was founded in 2018.", [(1, r"Lantern Works was founded in 2018\.")]),
        ("Its headquarters is in Galway, Ireland.", [(1, r"Its headquarters is in Galway, Ireland\.")]),
    ]),
    (STAFF_QUESTION, [
        ("Lighting design has 15 employees.", [(1, _LIGHTING_DESIGN_CARD)]),
        ("Stage operations has 8 employees.", [(1, _STAGE_OPERATIONS_CARD)]),
    ]),
    (GATEWAY_QUESTION, [
        ("The default gateway for VLAN 42 is 198.51.100.1.", [
            (2, r"VLAN\s+42[^\n]*198\.51\.100\.1\b"),
        ]),
    ]),
    ("متى تأسست شركة فانوس؟", [
        ("تأسست شركة فانوس عام 2018.", [(1, r"Lantern Works was founded in 2018\.")]),
    ]),
    ("كم عدد موظفي تصميم الإضاءة وتشغيل المسرح؟", [
        ("يعمل ١٥ موظفًا في تصميم الإضاءة.", [(1, _LIGHTING_DESIGN_CARD)]),
        ("يعمل ٨ موظفين في تشغيل المسرح.", [(1, _STAGE_OPERATIONS_CARD)]),
    ]),
    (INCIDENT_QUESTION, [
        ("The P1 restoration target was 42 minutes.", [(1, r"42 minutes from the start\.")]),
        ("LAB-731 began at 08:12 UTC and service was restored at 08:45 UTC.", [
            (3, r"The incident began at 08:12 UTC\.\s+Service was restored at 08:45 UTC\."),
        ]),
        ("Restoration took 33 minutes, so LAB-731 met the target by 9 minutes.", [
            (1, re.escape(POLICY)),
            (3, r"Restoration took 33 minutes, with a margin of 9 minutes\."),
        ]),
    ]),
], ids=[
    "company-founding-and-headquarters", "staff-counts", "vlan-gateway",
    "arabic-founding", "arabic-staff-counts", "cross-page-p1",
])
def test_grounded_chat_uses_original_pages_and_verified_citations(indexed_workspace, question, claims):
    settings, documents, document_id = indexed_workspace
    generator = Mock(spec=AnswerGenerator)
    passages_by_id = {}
    evidence_by_claim = []

    def respond(*, system_prompt, user_prompt):
        prompt = json.loads(user_prompt)
        assert prompt["question"] == question
        assert prompt["conversation_history"] == []
        passages = prompt["retrieved_passages"]
        assert passages
        assert all(passage["filename"] == FILENAME for passage in passages)
        assert all("198.51.100.254" not in passage["text"] for passage in passages)
        passages_by_id.update((passage["id"], passage) for passage in passages)
        for _, facts in claims:
            evidence_by_claim.append([_evidence(passages, page, pattern) for page, pattern in facts])
        return GeneratedAnswer(status="answered", claims=[
            {"text": text, "evidence": evidence}
            for (text, _), evidence in zip(claims, evidence_by_claim, strict=True)
        ])

    generator.generate.side_effect = respond
    with TestClient(create_app(settings, documents=documents, generator=generator)) as client:
        response = client.post("/api/chat", json={
            "question": question,
            "document_ids": [document_id],
            "history": [{"role": "assistant", "content": "The gateway is 198.51.100.254."}],
        })

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "answered", result
    generator.generate.assert_called_once()
    cited_quotes = {}
    for evidence in evidence_by_claim:
        for reference in evidence:
            cited_quotes.setdefault(reference["source_id"], set()).add(" ".join(reference["quote"].split()))
    assert len(result["sources"]) == len(cited_quotes)
    displayed_ids = {}
    for source, (prompt_id, quotes) in zip(result["sources"], cited_quotes.items(), strict=True):
        passage = passages_by_id[prompt_id]
        assert source["filename"] == FILENAME
        assert source["document_id"] == document_id
        assert source["page_number"] == passage["page_number"]
        assert source["chunk_id"].startswith(f"{document_id}:{source['page_number']}:")
        assert set(source["snippet"].splitlines()) == quotes
        displayed_ids[prompt_id] = source["id"]
    for (text, _), evidence in zip(claims, evidence_by_claim, strict=True):
        line = next(line for line in result["answer"].splitlines() if text.rstrip(".") in line)
        assert all(f'[{displayed_ids[reference["source_id"]]}]' in line for reference in evidence)


def test_chat_abstains_when_relevant_document_does_not_contain_the_fact(indexed_workspace):
    settings, documents, document_id = indexed_workspace
    generator = Mock(spec=AnswerGenerator)

    def respond(*, system_prompt, user_prompt):
        passages = json.loads(user_prompt)["retrieved_passages"]
        assert _evidence(passages, 1, r"Lantern Works was founded in 2018\.")
        assert all("catering budget" not in passage["text"] for passage in passages)
        return GeneratedAnswer(status="insufficient_context", claims=[])

    generator.generate.side_effect = respond
    with TestClient(create_app(settings, documents=documents, generator=generator)) as client:
        response = client.post("/api/chat", json={
            "question": "What is the catering budget for Lantern Works?",
            "document_ids": [document_id],
        })

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "insufficient_context"
    assert "enough information" in result["answer"]
    assert result["sources"] == []
    generator.generate.assert_called_once()


@pytest.mark.parametrize("forged_quote", [False, True], ids=["real-quote", "forged-quote"])
def test_chat_rejects_an_invented_staff_count_even_with_a_real_citation(indexed_workspace, forged_quote):
    settings, documents, document_id = indexed_workspace
    generator = Mock(spec=AnswerGenerator)

    def respond(*, system_prompt, user_prompt):
        passages = json.loads(user_prompt)["retrieved_passages"]
        evidence = _evidence(passages, 1, _LIGHTING_DESIGN_CARD)
        if forged_quote:
            evidence["quote"] = "Lighting design has 11 employees."
        return GeneratedAnswer(status="answered", claims=[{
            "text": "Lighting design has 11 employees.",
            "evidence": [evidence],
        }])

    generator.generate.side_effect = respond
    with TestClient(create_app(settings, documents=documents, generator=generator)) as client:
        response = client.post("/api/chat", json={"question": STAFF_QUESTION, "document_ids": [document_id]})

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_context"
    assert "11 employees" not in response.json()["answer"]


@pytest.fixture(scope="module")
def calculation_documents(indexed_workspace):
    settings, documents, _ = indexed_workspace
    with TestClient(create_app(settings, documents=documents)) as client:
        upload = client.post("/api/documents/upload", files=[
            ("files", ("incident-calculation.pdf", _handbook(include_margin=False), "application/pdf")),
            ("files", (
                "incident-missing-policy.pdf",
                _handbook(include_margin=False, include_policy=False),
                "application/pdf",
            )),
        ])
    assert upload.status_code == 200, upload.text
    exact, missing = upload.json()["documents"]
    return settings, documents, exact["id"], missing["id"]


@pytest.mark.parametrize("case", ["verified", "missing-operand", "incorrect-result"])
def test_cross_page_incident_calculation_requires_a_verified_equation(calculation_documents, case):
    settings, documents, complete_id, missing_policy_id = calculation_documents
    document_id = missing_policy_id if case == "missing-operand" else complete_id
    difference = 4 if case == "incorrect-result" else 9
    generator = Mock(spec=AnswerGenerator)

    def respond(*, system_prompt, user_prompt):
        prompt = json.loads(user_prompt)
        assert prompt["question"] == CALCULATION_QUESTION
        passages = prompt["retrieved_passages"]
        assert all(not re.search(r"\b9\s+minutes\b", passage["text"]) for passage in passages)
        evidence = []
        if case == "missing-operand":
            assert all(POLICY not in passage["text"] for passage in passages)
        else:
            evidence.append(_evidence(passages, 1, re.escape(POLICY)))
        evidence.append(_evidence(passages, 3, re.escape(INCIDENT_DURATION)))
        return GeneratedAnswer(status="answered", claims=[{
            "text": f"LAB-731 met the P1 target by {difference} minutes (42 − 33 = {difference}).",
            "evidence": evidence,
        }])

    generator.generate.side_effect = respond
    with TestClient(create_app(settings, documents=documents, generator=generator)) as client:
        response = client.post("/api/chat", json={
            "question": CALCULATION_QUESTION,
            "document_ids": [document_id],
        })

    assert response.status_code == 200, response.text
    output = response.json()
    generator.generate.assert_called_once()
    if case != "verified":
        assert output["status"] == "insufficient_context"
        assert output["sources"] == []
        return
    assert output["status"] == "answered", output
    assert "42 − 33 = 9" in output["answer"]
    assert "by 9 minutes" in output["answer"]
    assert {source["page_number"]: source["snippet"] for source in output["sources"]} == {
        1: POLICY,
        3: INCIDENT_DURATION,
    }
    assert all(source["document_id"] == document_id for source in output["sources"])
    assert all(f'[{source["id"]}]' in output["answer"] for source in output["sources"])


@pytest.mark.parametrize("policy_source_id", [1, 3], ids=["correct-source", "misassigned-source"])
def test_cross_page_calculation_cites_each_page_once_for_multiple_chunks(policy_source_id, make_settings):
    second_policy_quote = "The restoration clock starts when the incident begins."
    source_chunks = [
        RetrievedChunk("incident-id", FILENAME, 1, "incident-id:1:1", POLICY, 0.8),
        RetrievedChunk("incident-id", FILENAME, 1, "incident-id:1:2", second_policy_quote, 0.7),
        RetrievedChunk("incident-id", FILENAME, 1, "incident-id:1:3", "Priority levels are assigned at triage.", 0.65),
        RetrievedChunk("incident-id", FILENAME, 3, "incident-id:3:1", INCIDENT_DURATION, 0.9),
    ]
    documents = Mock(spec=DocumentService)
    documents.retrieve.return_value = source_chunks
    generator = Mock(spec=AnswerGenerator)
    generator.generate.return_value = GeneratedAnswer(status="answered", claims=[{
        "text": "LAB-731 met the P1 target by 9 minutes (42 - 33 = 9).",
        "evidence": [
            {"source_id": policy_source_id, "quote": POLICY},
            {"source_id": 2, "quote": second_policy_quote},
            {"source_id": 4, "quote": INCIDENT_DURATION},
        ],
    }])

    with TestClient(create_app(make_settings(), documents=documents, generator=generator)) as client:
        response = client.post("/api/chat", json={"question": CALCULATION_QUESTION})

    assert response.status_code == 200, response.text
    output = response.json()
    assert output["status"] == "answered", output
    assert [source["page_number"] for source in output["sources"]] == [1, 3]
    assert set(output["sources"][0]["snippet"].splitlines()) == {POLICY, second_policy_quote}
    assert output["sources"][0]["chunk_id"] == "incident-id:1:1"
    assert set(output["sources"][0]["chunk_ids"]) == {"incident-id:1:1", "incident-id:1:2"}
    assert output["sources"][1]["snippet"] == INCIDENT_DURATION
    assert output["sources"][1]["chunk_ids"] == ["incident-id:3:1"]
    assert output["answer"].count("[1]") == output["answer"].count("[2]") == 1
    assert "[3]" not in output["answer"]


@pytest.mark.parametrize(
    ("source_id", "policy_locations"),
    [
        (99, [("incident-id", 1, 2)]),
        (1, []),
        (1, [("other-document-id", 1, 1)]),
        (1, [("incident-id", 2, 1), ("incident-id", 4, 1)]),
    ],
    ids=["unknown-source", "absent-quote", "other-document", "ambiguous-page"],
)
def test_chat_rejects_policy_citations_without_a_safe_exact_match(source_id, policy_locations, make_settings):
    documents = Mock(spec=DocumentService)
    documents.retrieve.return_value = [
        RetrievedChunk("incident-id", FILENAME, 1, "incident-id:1:1", "Priority levels are assigned at triage.", 0.7),
        RetrievedChunk("incident-id", FILENAME, 3, "incident-id:3:1", INCIDENT_DURATION, 0.9),
        *[
            RetrievedChunk(document_id, FILENAME, page, f"{document_id}:{page}:{index}", POLICY, 0.8)
            for document_id, page, index in policy_locations
        ],
    ]
    generator = Mock(spec=AnswerGenerator)
    generator.generate.return_value = GeneratedAnswer(status="answered", claims=[{
        "text": "LAB-731 met the P1 target by 9 minutes (42 - 33 = 9).",
        "evidence": [
            {"source_id": source_id, "quote": POLICY},
            {"source_id": 2, "quote": INCIDENT_DURATION},
        ],
    }])

    with TestClient(create_app(make_settings(), documents=documents, generator=generator)) as client:
        response = client.post("/api/chat", json={"question": CALCULATION_QUESTION})

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "insufficient_context"
    assert response.json()["sources"] == []


@dataclass(frozen=True, slots=True)
class _IndexedCalculation:
    settings: Settings
    documents: DocumentService
    selected: DocumentRecord
    question: str
    expected_passages: dict[int, str]
    expected_claim: str


@pytest.fixture(scope="module", params=[
    pytest.param(("Blue Finch Service Handbook.pdf", "CASE-813", 2, 4, 42, 35), id="met-target"),
    pytest.param(("Maple Operations Guide.pdf", "OUT-572", 4, 2, 32, 40), id="missed-target"),
])
def varied_calculation(indexed_workspace, request):
    settings, documents, _ = indexed_workspace
    filename, case, policy_page, incident_page, limit, duration = request.param
    question = f"Did incident {case} meet the P1 restoration target? Explain the calculation and cite the relevant pages."
    policy = f"Priority 1 (P1) incidents have a restoration target of {limit} minutes."
    observed = f"The P1 incident {case} was restored in {duration} minutes."
    pages = [[(72, 710, "The team publishes quarterly onboarding guides.")] for _ in range(4)]
    pages[policy_page - 1] = [(72, 710, policy)]
    pages[incident_page - 1] = [(72, 710, observed)]
    equation = f"{max(limit, duration)} - {min(limit, duration)} = {abs(limit - duration)}"
    outcome = "met" if duration <= limit else "missed"
    claim = f"Incident {case} {outcome} the target by {abs(limit - duration)} minutes ({equation})."
    distractor_pdf = _pdf([
        (72, 720, f"Incident {case} was restored in 90 minutes."),
        (72, 700, "Priority 1 (P1) incidents have a restoration target of 91 minutes."),
    ])
    selected = documents.ingest(filename, BytesIO(_pdf(*pages)))
    documents.ingest(f"other-{filename}", BytesIO(distractor_pdf))
    return _IndexedCalculation(settings, documents, selected, question, {policy_page: policy, incident_page: observed}, claim)


def test_cross_page_retrieval_finds_both_operands_in_the_selected_pdf(varied_calculation):
    case = varied_calculation
    chunks = case.documents.retrieve(
        case.question,
        [case.selected.id],
        top_k=case.settings.rag_top_k,
        min_similarity=case.settings.rag_min_similarity,
    )

    assert len(chunks) == len(case.expected_passages)
    assert {chunk.page_number: chunk.text for chunk in chunks} == case.expected_passages
    assert {(chunk.document_id, chunk.filename) for chunk in chunks} == {(case.selected.id, case.selected.filename)}


def test_cross_page_calculation_is_independent_of_unrelated_history(varied_calculation):
    case = varied_calculation
    histories = [
        [],
        [
            {"role": "user", "content": "Is the observatory open on Mondays?"},
            {"role": "assistant", "content": "The observatory is open for 90 minutes."},
        ],
    ]
    prompts = []
    generator = Mock(spec=AnswerGenerator)

    def respond(*, system_prompt, user_prompt):
        prompt = json.loads(user_prompt)
        prompts.append(prompt)
        assert prompt["question"] == case.question
        passages = prompt["retrieved_passages"]
        assert len(passages) >= 2
        assert all(passage["filename"] == case.selected.filename for passage in passages)
        return GeneratedAnswer(status="answered", claims=[{
            "text": case.expected_claim,
            "evidence": [
                _evidence(passages, page, re.escape(quote))
                for page, quote in case.expected_passages.items()
            ],
        }])

    generator.generate.side_effect = respond
    with TestClient(create_app(case.settings, documents=case.documents, generator=generator)) as client:
        results = [
            client.post("/api/chat", json={
                "question": case.question, "document_ids": [case.selected.id], "history": history,
            })
            for history in histories
        ]

    assert len(prompts) == len(histories)
    assert prompts[0]["retrieved_passages"] == prompts[1]["retrieved_passages"]
    assert [result.status_code for result in results] == [200, 200]
    first, later = (result.json() for result in results)
    assert first["conversation_id"] != later["conversation_id"]
    assert {key: first[key] for key in ("status", "answer", "sources")} == {
        key: later[key] for key in ("status", "answer", "sources")
    }
    assert first["status"] == "answered", first
    assert case.expected_claim in first["answer"]
    assert {source["page_number"]: source["snippet"] for source in first["sources"]} == case.expected_passages
    assert {(source["document_id"], source["filename"]) for source in first["sources"]} == {
        (case.selected.id, case.selected.filename),
    }
    assert all(source["chunk_ids"] == [source["chunk_id"]] for source in first["sources"])
    assert all(f'[{source["id"]}]' in first["answer"] for source in first["sources"])

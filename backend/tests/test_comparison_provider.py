import json
from unittest.mock import Mock

import groq
import httpx
import pytest

from app.core.config import Settings
from app.rag.answer import _cited_answer
from app.rag.comparison import comparison_evidence_for
from app.rag.prompt import build_prompt
from app.rag.types import RetrievedChunk
from app.services.groq_client import GeneratedComparison, GroqGenerator, ProviderError


QUESTION = "Did incident X-602 meet the P1 restoration target? Explain the calculation and cite the relevant pages."
POLICY = "P1 incidents: maximum service restoration time is 42 minutes."
OBSERVED = "Total service disruption: 35 minutes"


def chunk(page, text, document="chosen"):
    return RetrievedChunk(document, "handbook.pdf", page, f"{document}:{page}:{len(text)}", text, 0.8)


@pytest.mark.parametrize(("observed_page", "output", "error"), [
    (3, {}, None), (1, {}, None), (3, {"met": False}, "groq_unverified_evidence"),
    (3, {"difference": 9}, "groq_unverified_evidence"),
    (3, {"policy_id": 2}, "groq_unverified_evidence"),
    (3, {"observation_id": 1}, "groq_unverified_evidence"),
])
def test_comparison_wire_ids_and_final_page_citations(monkeypatch, observed_page, output, error):
    chunks = [chunk(1, "P1 help desk drills have a two-hour first response target.\n" + POLICY),
              chunk(observed_page, "Incident X-602\nAffected users: 12\n" + OBSERVED)]
    prompt = build_prompt(QUESTION, [], chunks)
    sdk = groq.Groq
    calls = []

    def provider(request):
        body = json.loads(request.content)
        calls.append(body)
        wire = json.loads(body["messages"][1]["content"])
        assert [item["text"] for item in wire["policies"]] == [POLICY]
        assert [item["text"] for item in wire["observations"]] == [OBSERVED]
        schema = body["response_format"]["json_schema"]
        assert schema["strict"] and schema["name"] == "duration_comparison"
        policy_id, observation_id = wire["policies"][0]["id"], wire["observations"][0]["id"]
        properties = schema["schema"]["properties"]
        assert properties["policy_id"]["enum"] == [policy_id]
        assert properties["observation_id"]["enum"] == [observation_id]
        content = {"status": "answered", "policy_id": policy_id, "observation_id": observation_id,
                   "met": True, "difference": 7, **output}
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(content)}}]})

    monkeypatch.setattr(groq, "Groq", lambda **kwargs: sdk(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(provider))))
    generator = GroqGenerator(Settings(groq_api_key="test-key"))
    if error:
        with pytest.raises(ProviderError) as exc:
            generator.generate(system_prompt="", user_prompt=prompt)
        assert exc.value.code == error
    else:
        generated = generator.generate(system_prompt="", user_prompt=prompt)
        result = _cited_answer(generated, chunks, question=QUESTION)
        assert result and "met the P1 restoration target by 7 minutes" in result.answer
        assert "42 - 35 = 7" in result.answer
        assert sorted({source.page_number for source in result.sources}) == sorted({1, observed_page})
        assert {ref.quote for ref in generated.claims[0].evidence} == {POLICY, OBSERVED}
    assert len(calls) == 1


@pytest.mark.parametrize("chunks", [
    [chunk(1, POLICY)], [chunk(3, "Incident X-602\n" + OBSERVED)],
    [chunk(1, POLICY), chunk(3, "Incident X-602"), chunk(3, OBSERVED, document="other")],
])
def test_missing_role_or_other_same_named_pdf_does_not_establish_a_comparison(monkeypatch, chunks):
    provider = Mock()
    monkeypatch.setattr(groq, "Groq", provider)
    result = GroqGenerator(Settings(groq_api_key="test-key")).generate(
        system_prompt="", user_prompt=build_prompt(QUESTION, [], chunks),
    )
    assert result.status == "insufficient_context" and not result.claims
    provider.assert_not_called()


@pytest.mark.parametrize("chunks", [
    [chunk(1, POLICY.replace("42 minutes", "two hours")), chunk(3, "Incident X-602\n" + OBSERVED)],
    [chunk(1, POLICY.replace("42 minutes", "2 hours")), chunk(3, "Incident X-602\n" + OBSERVED)],
    [chunk(1, POLICY), chunk(3, "Incident X-602\nIncident X-999\n" + OBSERVED)],
    [chunk(1, POLICY + "\nP1 incidents: minimum restoration duration is 15 minutes."), chunk(3, "Incident X-602\n" + OBSERVED)],
    [chunk(1, POLICY), chunk(3, "Incident X-602 began at 08:12. Service restored: 08:45.")],
    [chunk(1, "P1 incidents must be back online within 42 minutes."), chunk(3, "Incident X-602\n" + OBSERVED)],
    [chunk(1, POLICY), chunk(3, "Incident X-602\nThe outage lasted thirty minutes.")],
])
def test_ambiguous_or_unparsed_comparison_keeps_full_passage_generation(chunks):
    assert comparison_evidence_for(build_prompt(QUESTION, [], chunks)) is None


@pytest.mark.parametrize("subject", ["it", "that one", "the Riverbend deployment"])
def test_subjects_without_a_source_bound_identifier_retain_the_full_prompt(subject):
    question = f"Did {subject} meet the P1 restoration target?"
    assert comparison_evidence_for(build_prompt(question, [], [chunk(1, POLICY), chunk(3, OBSERVED)])) is None


@pytest.mark.parametrize(("duration", "met", "margin"), [(42, True, 0), (47, False, 5)])
def test_equal_or_missed_limit_is_rendered_from_verified_values(duration, met, margin):
    chunks = [chunk(1, POLICY), chunk(3, f"Incident X-602\nTotal service disruption: {duration} minutes")]
    _, plan = comparison_evidence_for(build_prompt(QUESTION, [], chunks))
    answer = GeneratedComparison(status="answered", policy_id=1, observation_id=2, met=met, difference=margin).as_answer(plan)
    verified = _cited_answer(answer, chunks, question=QUESTION)
    assert verified and f"{'met' if met else 'missed'} the P1 restoration target by {margin} minutes" in verified.answer

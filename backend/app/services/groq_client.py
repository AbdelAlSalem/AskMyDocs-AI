import json
import logging
import re
import unicodedata
from decimal import Decimal
from typing import Protocol

import groq
from groq.types.completion_usage import CompletionUsage
from groq.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from app.core.config import DemoSettings, Settings
from app.models.schemas import ChatStatus
from app.rag.comparison import COMPARISON_PROMPT, ComparisonEvidence, comparison_evidence_for
from app.rag.debug import trace_fields, trace_wire_prompt
from app.rag.prompt import ARABIC_SUMMARY_INSTRUCTIONS, SummaryEvidence, prepare_summary_prompt
from app.services.errors import ServiceError


SUPPORTED_MODELS = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b"})
_PROVIDER_CODES = frozenset({
    "context_length_exceeded", "invalid_request_error", "invalid_value", "json_validate_failed",
    "rate_limit_exceeded", "server_error", "service_unavailable",
})
logger = logging.getLogger("uvicorn.error.askmydocs.groq")
ANSWER_FORMAT: ResponseFormat = {
    "type": "json_schema",
    "json_schema": {
        "name": "document_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["answered", "insufficient_context"]},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "evidence": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_id": {"type": "integer"},
                                        "quote": {"type": "string"},
                                    },
                                    "required": ["source_id", "quote"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["text", "evidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["status", "claims"],
            "additionalProperties": False,
        },
    },
}


def _summary_format(evidence: SummaryEvidence) -> ResponseFormat:
    properties = {}
    for key, ids, limit in evidence.groups() or [("points", set(evidence.quotes), 10)]:
        anchor = {"type": "integer", **({"enum": sorted(ids)} if ids else {})}
        properties[key] = {
            "type": "array", "maxItems": limit, "minItems": int(bool(evidence.required_pages)),
            "items": {
                "type": "object", "additionalProperties": False, "required": ["text", "evidence_ids"],
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {"type": "array", "minItems": 1, "maxItems": 8, "items": anchor},
                },
            },
        }
    return {"type": "json_schema", "json_schema": {
        "name": "document_summary", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "properties": properties, "required": list(properties)},
    }}


class ProviderError(ServiceError):
    """A Groq operation failure."""

    def __init__(
        self, status_code: int, code: str, message: str, *,
        retry_after: int | None = None, output_failure: bool = False,
    ) -> None:
        super().__init__(status_code, code, message)
        self.retry_after = retry_after
        self.output_failure = output_failure


class EvidenceQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_id: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=1000)


class GeneratedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=1000)
    evidence: list[EvidenceQuote] = Field(min_length=1, max_length=8)


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ChatStatus
    claims: list[GeneratedClaim] = Field(max_length=12)


class GeneratedComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ChatStatus
    policy_id: int = Field(strict=True)
    observation_id: int = Field(strict=True)
    met: bool = Field(strict=True)
    difference: Decimal = Field(ge=0)

    def as_answer(self, plan: ComparisonEvidence) -> GeneratedAnswer:
        if self.status == "insufficient_context":
            return GeneratedAnswer(status=self.status, claims=[])
        try:
            text, facts = plan.resolve(self.policy_id, self.observation_id, self.met, self.difference)
        except (KeyError, ValueError):
            trace_fields(schema_validation="passed", verification_reason="invalid_duration_comparison")
            raise ProviderError(502, "groq_unverified_evidence",
                                "The generated comparison could not be verified.", output_failure=True) from None
        references = [EvidenceQuote(source_id=fact.passage_id, quote=fact.quote) for fact in facts]
        return GeneratedAnswer(status="answered", claims=[GeneratedClaim(text=text, evidence=references)])


class SummaryCandidates(GeneratedAnswer):
    required_sources: list[set[int]]
    point_count: int | None


class SummaryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[int] = Field(min_length=1, max_length=8)


_SUMMARY_GROUPS = TypeAdapter(dict[str, list[SummaryPoint]])
_RESTORED_SERVICE = re.compile(
    r"\b(?:service (?:was |fully |was fully )?restored|restor(?:e|ed|ation of) (?:the )?service)\b", re.I,
)
_CHANGED_SERVICE = re.compile(
    r"\b(?:service (?:was |has been )?(?:modified|amended|adjusted|changed)|"
    r"(?:modif(?:y|ied)|alter(?:ed)?|chang(?:e|ed)|adjust(?:ed)?) (?:the )?service)\b", re.I,
)
_AR_CHANGED_SERVICE = re.compile(r"(?<!\w)[وف]?(?:تعديل|ت?عدل(?:ت)?|تغيرت|غيرت)\s+الخدمة\b")


class GeneratedSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points: list[SummaryPoint] = Field(max_length=10)

    @classmethod
    def parse_completion(cls, content: str, evidence: SummaryEvidence):
        groups = evidence.groups()
        if not groups:
            return cls.model_validate_json(content)
        parsed = _SUMMARY_GROUPS.validate_json(content)
        if set(parsed) != {key for key, _, _ in groups} or any(
            not 1 <= len(parsed[key]) <= limit for key, _, limit in groups
        ):
            trace_fields(schema_validation="failed", output_error="invalid_page_groups")
            raise ProviderError(502, "groq_invalid_response", "The summary page groups were invalid.", output_failure=True)
        points = [
            point for key, ids, _ in groups for point in parsed[key] if ids.issuperset(point.evidence_ids)
        ]
        discarded = sum(map(len, parsed.values())) - len(points)
        if discarded:
            trace_fields(invalid_summary_group_candidates=discarded)
        return cls(points=points)

    def as_answer(self, evidence: SummaryEvidence) -> GeneratedAnswer:
        claims = []
        cited = set()
        omitted = []
        for point in self.points:
            references = []
            if not set(point.evidence_ids).issubset(evidence.quotes):
                omitted.append("unknown_evidence_id")
                continue
            for evidence_id in dict.fromkeys(point.evidence_ids):
                source_id, quote = evidence.quotes[evidence_id]
                references.append(EvidenceQuote(source_id=source_id, quote=quote))
            text = point.text
            letters = [c for c in text if c.isalpha()]
            arabic = sum(unicodedata.name(c, "").startswith("ARABIC LETTER") for c in letters)
            if (evidence.language == "Arabic" and (not letters or arabic * 4 < len(letters))) or (
                evidence.language == "English" and arabic * 2 > len(letters)
            ):
                omitted.append("summary_language_mismatch")
                continue
            if evidence.language == "Arabic":
                plain = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
                source = " ".join(reference.quote for reference in references)
                if _AR_CHANGED_SERVICE.search(plain) and _RESTORED_SERVICE.search(source) and not _CHANGED_SERVICE.search(source):
                    omitted.append("summary_translation_changes_event")
                    continue
            if evidence.bullets:
                text = "- " + re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text)
            claims.append(GeneratedClaim(text=text, evidence=references))
            cited.update(point.evidence_ids)
        if omitted:
            trace_fields(omitted_summary_candidates=omitted)
        if (self.points and not claims) or (evidence.point_count and len(claims) < evidence.point_count) or any(
            not cited.intersection(page) for page in evidence.required_pages
        ):
            trace_fields(schema_validation="passed", verification_reason=omitted[0] if omitted else "summary_coverage_incomplete")
            raise ProviderError(
                502, "groq_unverified_evidence", "The requested summary language, coverage or citations could not be verified.",
                output_failure=True,
            )
        return SummaryCandidates(
            status="answered" if claims else "insufficient_context",
            claims=claims,
            required_sources=[{evidence.quotes[number][0] for number in page} for page in evidence.required_pages],
            point_count=evidence.point_count,
        )


class AnswerGenerator(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str, summary: bool = False) -> GeneratedAnswer: ...


def _context_size(user_prompt: str) -> int | None:
    try:
        passages = json.loads(user_prompt)["retrieved_passages"]
        return sum(len(passage["text"]) for passage in passages)
    except (KeyError, TypeError, ValueError):
        return None


def _provider_code(error: groq.APIError) -> str | None:
    body = error.body
    if not isinstance(body, dict):
        return None
    details = body.get("error", body)
    if not isinstance(details, dict):
        return None
    code = details.get("code")
    return code if isinstance(code, str) and code in _PROVIDER_CODES else None


def _token_count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


class GroqGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        is_demo = isinstance(settings, DemoSettings)
        self._completion_tokens = settings.demo_max_completion_tokens if is_demo else 2500
        self._reasoning_effort = "low" if is_demo else "medium"

    def generate(self, *, system_prompt: str, user_prompt: str, summary: bool = False) -> GeneratedAnswer:
        if not self.settings.groq_configured:
            raise ProviderError(
                503, "groq_not_configured", "Set GROQ_API_KEY in backend/.env and restart the backend."
            )
        if self.settings.groq_model not in SUPPORTED_MODELS:
            raise ProviderError(
                503,
                "groq_model_invalid",
                "Set GROQ_MODEL to openai/gpt-oss-20b or openai/gpt-oss-120b and restart the backend.",
            )

        reasoning_effort = "low" if summary else self._reasoning_effort
        context_chars = _context_size(user_prompt)
        summary_evidence = None
        comparison = None
        response_format = ANSWER_FORMAT
        if summary:
            user_prompt, summary_evidence = prepare_summary_prompt(user_prompt)
            if summary_evidence.language == "Arabic":
                system_prompt += ARABIC_SUMMARY_INSTRUCTIONS
            response_format = _summary_format(summary_evidence)
            trace_fields(evidence_id_to_passage={i: source_id for i, (source_id, _) in summary_evidence.quotes.items()})
        elif prepared := comparison_evidence_for(user_prompt):
            question, comparison = prepared
            trace_fields(comparison_policy_count=len(comparison.policies),
                         comparison_observation_count=len(comparison.observations))
            if not comparison.policies or not comparison.observations:
                trace_fields(evidence="incomplete", provider_requested=False)
                return GeneratedAnswer(status="insufficient_context", claims=[])
            system_prompt = COMPARISON_PROMPT
            user_prompt = comparison.wire(question)
            response_format = comparison.response_format()
            reasoning_effort = "low"
            trace_fields(evidence_id_to_passage={key: row.passage_id for group in
                         (comparison.policies, comparison.observations) for key, row in group.items()})

        def record_failure(
            reason: str, *, upstream_status: int | None = None, upstream_code: str | None = None,
            usage: CompletionUsage | None = None, validation_errors: int | None = None,
        ) -> None:
            # Provider errors and validation failures can echo private prompt or
            # completion text. Record only fixed labels and numeric measurements.
            logger.warning(
                "Groq generation failed: reason=%s upstream_status=%s upstream_code=%s model=%s summary=%s "
                "context_chars=%s user_chars=%d system_chars=%d max_completion_tokens=%d reasoning_effort=%s "
                "response_format=%s "
                "prompt_tokens=%s completion_tokens=%s reasoning_tokens=%s validation_errors=%s",
                reason, upstream_status, upstream_code, self.settings.groq_model, summary,
                context_chars, len(user_prompt), len(system_prompt), self._completion_tokens,
                reasoning_effort, "json_schema",
                _token_count(getattr(usage, "prompt_tokens", None)),
                _token_count(getattr(usage, "completion_tokens", None)),
                _token_count(getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)),
                validation_errors,
            )

        try:
            trace_wire_prompt(user_prompt, self.settings.groq_api_key.get_secret_value())
            trace_fields(
                provider_requested=True, model=self.settings.groq_model, temperature=0,
                max_completion_tokens=self._completion_tokens, reasoning_effort=reasoning_effort,
                include_reasoning=False, response_format="json_schema",
                response_schema=response_format["json_schema"]["name"],
                timeout_seconds=self.settings.groq_timeout_seconds,
            )
            with groq.Groq(
                api_key=self.settings.groq_api_key.get_secret_value(),
                base_url="https://api.groq.com",
                timeout=self.settings.groq_timeout_seconds,
                max_retries=0,
            ) as client:
                result = client.chat.completions.create(
                    model=self.settings.groq_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=self._completion_tokens,
                    reasoning_effort=reasoning_effort,
                    include_reasoning=False,
                    response_format=response_format,
                    temperature=0,
                )
        except groq.RateLimitError as error:
            trace_fields(upstream_status=error.status_code, outcome="provider_request_failed")
            header = error.response.headers.get("Retry-After", "")
            retry_after = int(header) if header.isascii() and header.isdecimal() and len(header) <= 5 else 20
            raise ProviderError(
                429, "groq_rate_limit", "Groq's rate limit was reached. Wait briefly and retry.",
                retry_after=max(1, min(retry_after, 86400)),
            ) from None
        except groq.APITimeoutError:
            record_failure("timeout")
            raise ProviderError(504, "groq_timeout", "Groq took too long to respond. Please retry.") from None
        except groq.APIConnectionError:
            record_failure("connection")
            raise ProviderError(
                502, "groq_network_error", "Could not reach Groq. Check your connection and retry."
            ) from None
        except (groq.AuthenticationError, groq.PermissionDeniedError) as error:
            trace_fields(upstream_status=error.status_code, outcome="provider_request_failed")
            raise ProviderError(
                503, "groq_auth_error", "Groq refused access. Check the API key and account permissions."
            ) from None
        except groq.NotFoundError as error:
            trace_fields(upstream_status=error.status_code, outcome="provider_request_failed")
            raise ProviderError(
                503, "groq_model_invalid", "Groq could not find the configured model. Check GROQ_MODEL."
            ) from None
        except groq.APIError as error:
            provider_code = _provider_code(error)
            trace_fields(upstream_status=getattr(error, "status_code", None), provider_code=provider_code)
            if provider_code == "json_validate_failed":
                trace_fields(schema_validation="rejected_by_provider")
            record_failure(
                "provider_error", upstream_status=getattr(error, "status_code", None),
                upstream_code=provider_code,
            )
            raise ProviderError(
                502, "groq_request_failed", "Groq could not complete the request. Please retry.",
                output_failure=provider_code == "json_validate_failed",
            ) from None

        choice = result.choices[0] if result.choices else None
        finish = choice.finish_reason if choice is not None else None
        trace_fields(
            upstream_status=200, finish_reason=finish, generation_completed=finish == "stop",
            prompt_tokens=_token_count(getattr(result.usage, "prompt_tokens", None)),
            completion_tokens=_token_count(getattr(result.usage, "completion_tokens", None)),
            reasoning_tokens=_token_count(getattr(getattr(result.usage, "completion_tokens_details", None), "reasoning_tokens", None)),
        )
        if choice is not None and choice.finish_reason == "length":
            record_failure("length", upstream_status=200, usage=result.usage)
            message = (
                "Groq reached its output limit. Try requesting fewer summary points or a smaller page range."
                if summary else "Groq reached its output limit. Try requesting a shorter answer."
            )
            raise ProviderError(502, "groq_output_limit", message, output_failure=True)
        reason = "missing_choice" if choice is None else "incomplete_output"
        validation_errors = None
        if choice is not None and choice.finish_reason == "stop":
            reason = "empty_output"
            if choice.message.content:
                try:
                    if summary_evidence is not None:
                        answer = GeneratedSummary.parse_completion(choice.message.content, summary_evidence).as_answer(summary_evidence)
                    elif comparison is not None:
                        answer = GeneratedComparison.model_validate_json(choice.message.content).as_answer(comparison)
                    else:
                        answer = GeneratedAnswer.model_validate_json(choice.message.content)
                    trace_fields(schema_validation="passed")
                    return answer
                except ValidationError as error:
                    reason = "invalid_schema"
                    validation_errors = error.error_count()
                    trace_fields(schema_validation="failed", validation_errors=validation_errors)
        trace_fields(output_error=reason)
        record_failure(reason, upstream_status=200, usage=result.usage, validation_errors=validation_errors)
        raise ProviderError(
            502, "groq_invalid_response", "Groq returned an incomplete answer. Please retry.", output_failure=True,
        )

import re
import unicodedata
from collections.abc import Callable
from fractions import Fraction
from itertools import combinations
from typing import Protocol

from app.core.config import DemoSettings, Settings
from app.models.schemas import ChatRequest, ChatResponse, SourceReference
from app.rag.debug import log_context, request_trace, trace_fields
from app.rag.grounding import wording_supported
from app.rag.document_wide import (
    LOCAL_CONTEXT_CHARS,
    DocumentCoverage,
    DocumentIntent,
    SummaryPageScope,
    document_intent,
    summary_excerpts,
    summary_page_scope,
)
from app.rag.prompt import DOCUMENT_INSTRUCTIONS, SUMMARY_SYSTEM_PROMPT, SYSTEM_PROMPT, build_prompt
from app.rag.temporal import render_temporal_row, temporal_evidence_rows
from app.rag.types import RetrievedChunk
from app.services.groq_client import AnswerGenerator, GeneratedAnswer, ProviderError, SummaryCandidates


INLINE_MARKER = re.compile(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")
NUMBER = re.compile(r"(?<!\w)(?<![^\W\d_]-)(?<!\d[.:/,])[+-]?\d+(?:[.:/,]\d+)*")
DECIMAL = r"[+-]?\d+(?:\.\d+)?"
TIME_UNIT = r"(?:hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|دقائق|دقيقة|ثواني|ثانية|ساعات|ساعة)"
EQUATION = re.compile(
    rf"(?<![\w.:/,])({DECIMAL})\s*({TIME_UNIT})?\s*([+-]|minus\b|plus\b)\s*"
    rf"({DECIMAL})\s*({TIME_UNIT})?\s*(?:=|equals\b)\s*({DECIMAL})(?!\w|[.:/,]\d)"
    rf"(?:\s+({TIME_UNIT})\b)?",
    re.IGNORECASE,
)
NUMERIC_CHARACTERS = str.maketrans("٫٬−–‐‑", ".,----")
ARABIC_NUMBER_PREFIX = re.compile(r"(?<!\w)(?:[وف]?[بكل]?ال|[وف]?[بكل]|[وف])ـ*(?=[+-]?\d)")
CLOCK_RANGE = re.compile(r"(\b\d{1,2}:\d{2})\s*-\s*(?=\d{1,2}:\d{2}\b)")
TIME_ZONE_OFFSET = re.compile(r"\b(UTC|GMT)\s*([+-])\s*(?=\d)", re.IGNORECASE)
NUMBERED_LABEL = re.compile(r"(?<![a-z\d-])([a-z]+-?)(\d+)(?![a-z\d])", re.IGNORECASE | re.ASCII)
QUOTE_TYPOGRAPHY = str.maketrans({"‐": "-", "‑": "-", "’": "'", "‘": "'", "”": '"', "“": '"'})
QUANTITY = re.compile(rf"(?<![\w.:/-])({DECIMAL})[\s-]+({TIME_UNIT})\b", re.IGNORECASE)
UPPER_BOUND = re.compile(r"\b(?:within|at most|no more than|maximum|limit)\b", re.IGNORECASE)
TIME_TARGET = re.compile(r"\btarget\b", re.IGNORECASE)
ELAPSED = re.compile(r"\b(?:restor\w*|response|duration|disruption|elapsed|completed|took|latency)\b", re.IGNORECASE)
MARGIN = re.compile(rf"\bby\s+({DECIMAL})\s+({TIME_UNIT})\b", re.IGNORECASE)
_LIMIT_OBJECT = r"(?:\s+[\w-]+){0,4}\s+(?:target|limit)\b"
MISSED_LIMIT = re.compile(
    rf"\b(?:missed|exceeded|failed to meet|did not meet|didn't meet){_LIMIT_OBJECT}"
    r"|\b(?:limit|target) was (?:not met|missed|exceeded)\b", re.IGNORECASE,
)
MET_LIMIT = re.compile(rf"\b(?:met|within){_LIMIT_OBJECT}|\b(?:limit|target) was met\b", re.IGNORECASE)
SNIPPET_LENGTH = 360


def _in_arabic(question: str) -> bool:
    letters = [char for char in question if char.isalpha()]
    arabic_letters = sum(unicodedata.name(char, "").startswith("ARABIC ") for char in letters)
    return arabic_letters * 2 > len(letters)


def insufficient_answer(question: str) -> str:
    if _in_arabic(question):
        return "لا أجد معلومات كافية في المستندات المحددة للإجابة عن هذا السؤال."
    return "I couldn’t find enough information in the selected documents to answer this question."


def _verification_failure(question: str) -> str:
    if _in_arabic(question):
        return "وجدت مقاطع ذات صلة، لكن تعذّر التحقق من الإجابة بالاستناد إليها."
    return "I found relevant passages, but couldn’t verify the generated answer against them."


def make_snippet(text: str) -> str:
    return text if len(text) <= SNIPPET_LENGTH else text[: SNIPPET_LENGTH - 1].rstrip() + "…"


def _normalized_text(text: str) -> str:
    return " ".join(text.split())


def _numeric_text(text: str) -> str:
    digits = "".join(str(unicodedata.decimal(char)) if char.isdecimal() else char for char in text)
    # The conjunction in و١٨:٠٠ and the hyphen in 09:00-18:00 introduce
    # another complete value. Keep both endpoints independently checkable.
    normalized = ARABIC_NUMBER_PREFIX.sub(" ", digits.translate(NUMERIC_CHARACTERS))
    normalized = CLOCK_RANGE.sub(r"\1 ", normalized)
    return TIME_ZONE_OFFSET.sub(r"\1 \2", normalized)


def _numbers(text: str) -> set[str]:
    # Compare measured values and addresses. Digits in priority codes and
    # incident IDs are labels, even when a quote contains only the matched value.
    return set(NUMBER.findall(_numeric_text(text)))


def _time_unit(unit: str) -> str:
    unit = unit.casefold()
    for name, aliases in (
        ("minute", {"minute", "minutes", "min", "mins", "دقيقة", "دقائق"}),
        ("second", {"second", "seconds", "sec", "secs", "ثانية", "ثواني"}),
        ("hour", {"hour", "hours", "hr", "hrs", "ساعة", "ساعات"}),
    ):
        if unit in aliases:
            return name
    return "day"


def _figures_supported(text: str, quoted_numbers: set[str], quotes: list[str] | None = None) -> bool:
    normalized = _numeric_text(text.replace("**", ""))
    derived: set[str] = set()
    labels: dict[str, set[str]] = {}
    for quote in quotes or []:
        for family, number in NUMBERED_LABEL.findall(_numeric_text(quote).casefold()):
            labels.setdefault(family, set()).add(number)
    if any(family in labels and number not in labels[family]
           for family, number in NUMBERED_LABEL.findall(normalized.casefold())):
        return False
    quantities = {
        (value, _time_unit(unit))
        for quote in quotes or [] for value, unit in QUANTITY.findall(_numeric_text(quote))
    }
    for equation in EQUATION.finditer(normalized):
        left, left_unit, operator, right, right_unit, result, result_unit = equation.groups()
        if left not in quoted_numbers or right not in quoted_numbers:
            return False
        units = {_time_unit(unit) for unit in (left_unit, right_unit, result_unit) if unit}
        if len(units) > 1:
            return False
        if units and quotes:
            unit = next(iter(units))
            if not {(left, unit), (right, unit)} <= quantities:
                return False
        a, b = Fraction(left), Fraction(right)
        expected = a + b if operator.lower() in {"+", "plus"} else a - b
        if expected != Fraction(result):
            return False
        derived.add(result)
    # The equation has already been checked, including operators that could be
    # read as signs by the standalone numeric literal matcher.
    direct_figures = _numbers(EQUATION.sub(" ", normalized))
    # For two separately cited durations, a correct subtraction alone doesn't
    # establish that the measured duration was on the claimed side of the limit.
    if quotes and re.search(r"\b(?:limit|target)\b", normalized, re.IGNORECASE):
        limits, observations = set(), set()
        for quote in quotes:
            measures = QUANTITY.findall(_numeric_text(quote))
            if len(measures) == 1:
                value, unit = measures[0]
                if re.search(r"\b(?:minimum|at least|no less than)\b", quote, re.IGNORECASE):
                    continue
                if UPPER_BOUND.search(quote) or (TIME_TARGET.search(quote) and ELAPSED.search(quote)):
                    limits.add((Fraction(value), _time_unit(unit)))
                elif ELAPSED.search(quote):
                    observations.add((Fraction(value), _time_unit(unit)))
        if len(limits) == len(observations) == 1:
            (limit, unit), = limits
            (measured, measured_unit), = observations
            if unit == measured_unit:
                if MISSED_LIMIT.search(normalized) and measured <= limit:
                    return False
                if not MISSED_LIMIT.search(normalized) and MET_LIMIT.search(normalized) and measured > limit:
                    return False
                if margin := MARGIN.search(normalized):
                    value, margin_unit = margin.groups()
                    if _time_unit(margin_unit) != unit or Fraction(value) != abs(limit - measured):
                        return False
                    derived.add(value)
    return direct_figures <= quoted_numbers | derived


def _matching_quote(chunk: RetrievedChunk, passage: str, quote: str) -> str | None:
    # Only typographic equivalents are interchangeable. Return the original
    # source substring so the UI always displays the document's own spelling.
    quote = quote.translate(QUOTE_TYPOGRAPHY)
    canonical = passage.translate(QUOTE_TYPOGRAPHY)
    if not any(char.isalnum() for char in quote):
        return None
    added_period = quote.endswith(".") and len(quote) > 1 and quote[-2].isalnum()
    body = quote[:-1] if added_period else quote
    start = r"(?<!\w)" if quote[0].isalnum() else ""
    end = r"(?!\w)" if body[-1].isalnum() else ""
    if quote[0].isdecimal():
        start += r"(?<!\d[.:/,])(?<![+-])"
    if body[-1].isdecimal():
        end += r"(?![.:/,]\d)"
    pattern = start + re.escape(body) + end
    for match in re.finditer(pattern, canonical):
        finish = match.end()
        if not added_period:
            return passage[match.start():finish]
        if canonical[finish:finish + 1] == ".":
            return passage[match.start():finish + 1]
        # PDF tables and label/value lists often omit terminal punctuation.
        # A model-added period is harmless only at an actual source line end.
        offset = -1
        for line in chunk.text.splitlines():
            if normalized := _normalized_text(line):
                offset += len(normalized) + 1
                if offset == finish:
                    return passage[match.start():finish]
    return None


def _verified_quote(
    source_id: int, quote: str, passages: dict[int, RetrievedChunk], texts: dict[int, str],
) -> tuple[int, str] | None:
    supplied = passages.get(source_id)
    if supplied is None or not quote:
        return None
    if actual := _matching_quote(supplied, texts[source_id], quote):
        return source_id, actual

    # A model can confuse two numbered chunks from the same document. Repair
    # only when the verbatim quote identifies a single supplied document page.
    matches = [
        (candidate_id, actual) for candidate_id, chunk in passages.items()
        if chunk.document_id == supplied.document_id
        and (actual := _matching_quote(chunk, texts[candidate_id], quote))
    ]
    if len({passages[candidate_id].page_number for candidate_id, _ in matches}) == 1:
        return matches[0]
    return None


def _cited_answer(
    generated: GeneratedAnswer, chunks: list[RetrievedChunk], *, intent: DocumentIntent | None = None,
    arabic: bool = False, question: str = "", report_rejection: Callable[..., None] = trace_fields,
) -> ChatResponse | None:
    if generated.status != "answered" or not generated.claims:
        return None

    passages = dict(enumerate(chunks, start=1))
    normalized_passages = {source_id: _normalized_text(chunk.text) for source_id, chunk in passages.items()}
    excerpts: dict[int, list[str]] = {}
    chunk_ids: dict[int, list[str]] = {}
    page_sources: dict[tuple[str, int], int] = {}
    claims: list[tuple[str, list[int]]] = []
    temporal = intent in ("dates", "deadlines")
    for claim_index, claim in enumerate(generated.claims, start=1):
        text = INLINE_MARKER.sub("", claim.text).strip()
        quoted_numbers: set[str] = set()
        claim_quotes: list[str] = []
        source_ids: list[int] = []
        temporal_rows = []
        for reference in claim.evidence:
            quote = _normalized_text(reference.quote)
            match = _verified_quote(reference.source_id, quote, passages, normalized_passages)
            if match is None:
                report_rejection(
                    rejection="unknown_source" if reference.source_id not in passages else "quote_mismatch",
                    rejected_claim=claim_index, rejected_source_id=reference.source_id,
                )
                return None
            verified_id, quote = match
            chunk = passages[verified_id]
            rows = temporal_evidence_rows(chunk.text, quote, deadlines_only=intent == "deadlines") if temporal else []
            if temporal and not rows:
                continue
            source_id = page_sources.setdefault((chunk.document_id, chunk.page_number), verified_id)
            previous = excerpts.setdefault(source_id, [])
            for excerpt in [row.text for row in rows] if temporal else [quote]:
                claim_quotes.append(excerpt)
                quoted_numbers.update(_numbers(excerpt))
                if excerpt not in previous:
                    previous.append(excerpt)
            temporal_rows.extend((row, source_id) for row in rows)
            contributing = chunk_ids.setdefault(source_id, [])
            if chunk.chunk_id not in contributing:
                contributing.append(chunk.chunk_id)
            if source_id not in source_ids:
                source_ids.append(source_id)
        if not claim_quotes or not text or not _figures_supported(text, quoted_numbers, claim_quotes):
            report_rejection(rejection="unsupported_figures" if claim_quotes and text else "missing_evidence",
                         rejected_claim=claim_index)
            return None
        if temporal:
            for row, source_id in temporal_rows:
                rendered = render_temporal_row(row, arabic=arabic, generated_text=text if len(temporal_rows) == 1 else "")
                line = (rendered, [source_id])
                if line not in claims:
                    claims.append(line)
        else:
            if not wording_supported(text, claim_quotes, question, figures=bool(_numbers(text))):
                report_rejection(rejection="unrelated_claim", rejected_claim=claim_index)
                return None
            claims.append((text, source_ids))

    display_ids = {source_id: number for number, source_id in enumerate(excerpts, start=1)}
    paragraphs = [
        f"{text} " + " ".join(f"[{display_ids[source_id]}]" for source_id in source_ids)
        for text, source_ids in claims
    ]
    sources = [
        SourceReference(
            id=display_ids[source_id],
            document_id=passages[source_id].document_id,
            filename=passages[source_id].filename,
            page_number=passages[source_id].page_number,
            chunk_id=passages[source_id].chunk_id,
            chunk_ids=chunk_ids[source_id],
            snippet="\n".join(quotes) if intent == "summary" else make_snippet("\n".join(quotes)),
        )
        for source_id, quotes in excerpts.items()
    ]
    return ChatResponse(status="answered", answer="\n\n".join(paragraphs), sources=sources)


def _verified_summary(generated: SummaryCandidates, chunks: list[RetrievedChunk], question: str) -> ChatResponse | None:
    candidates = []
    seen = set()
    rejected = []
    for index, claim in enumerate(generated.claims, start=1):
        result = {}
        one_claim = GeneratedAnswer(status="answered", claims=[claim])
        answer = _cited_answer(one_claim, chunks, intent="summary", question=question, report_rejection=result.update)
        if answer is None:
            rejected.append({"candidate": index, "reason": result.get("rejection")})
            continue
        content = _normalized_text(claim.text).casefold()
        if content not in seen:
            seen.add(content)
            candidates.append((claim, {(source.document_id, source.page_number) for source in answer.sources}))
    trace_fields(summary_candidates=len(generated.claims), rejected_summary_candidates=rejected)
    required = {
        (chunks[source_id - 1].document_id, chunks[source_id - 1].page_number)
        for page in generated.required_sources for source_id in page
    }
    count = generated.point_count or min(5, len(candidates))
    # At most ten candidates arrive in a single completion. Choose the earliest
    # verified combination that covers the most distinct in-scope pages.
    best = None
    best_coverage = -1
    for group in combinations(candidates, count):
        pages = {page for _, claim_pages in group for page in claim_pages}
        if pages.issuperset(required) and len(pages) > best_coverage:
            best, best_coverage = group, len(pages)
    if not best:
        trace_fields(rejection="summary_coverage_incomplete", required_count=count, verified_candidates=len(candidates))
        return None
    trace_fields(selected_summary_points=count)
    chosen = GeneratedAnswer(status="answered", claims=[claim for claim, _ in best])
    return _cited_answer(chosen, chunks, intent="summary", question=question)


def _excerpt_answer(question: str, chunks: list[RetrievedChunk]) -> ChatResponse | None:
    generated = GeneratedAnswer(status="answered", claims=[
        {"text": f"- {quote}", "evidence": [{"source_id": source_id, "quote": quote}]}
        for source_id, quote in summary_excerpts(chunks)
    ])
    answer = _cited_answer(generated, chunks, intent="summary")
    if answer is not None:
        intro = (
            "تعذّر التحقق من الملخّص المولّد. إليك مقتطفات موثّقة بلغتها الأصلية."
            if _in_arabic(question) else
            "A generated summary couldn’t be verified. Here are cited excerpts in their original language."
        )
        answer.answer = f"{intro}\n\n{answer.answer}"
    return answer


class Retriever(Protocol):
    def retrieve(
        self, question: str, document_ids: list[str] | None, *, top_k: int, min_similarity: float,
    ) -> list[RetrievedChunk]: ...

    def retrieve_document_wide(
        self, document_ids: list[str] | None, *, intent: DocumentIntent, max_context_chars: int,
        page_scope: SummaryPageScope | None = None,
    ) -> DocumentCoverage: ...


class AnswerService:
    def __init__(self, documents: Retriever, generator: AnswerGenerator, settings: Settings) -> None:
        self.documents = documents
        self.generator = generator
        self.settings = settings

    def answer(self, request: ChatRequest) -> ChatResponse:
        debug = self.settings.rag_debug_context and not (self.settings.demo_mode or isinstance(self.settings, DemoSettings))
        with request_trace(debug, request.document_ids, self.settings.groq_api_key.get_secret_value()):
            return self._answer(request, debug)

    def _answer(self, request: ChatRequest, debug: bool) -> ChatResponse:
        intent = document_intent(request.question)
        coverage = None
        page_scope = summary_page_scope(request.question) if intent == "summary" else None
        trace_fields(question=request.question, intent=intent or "factual",
                     strategy="document_wide" if intent else "hybrid", page_scope=page_scope)
        system_prompt = SYSTEM_PROMPT
        if intent is None:
            chunks = self.documents.retrieve(
                request.question,
                request.document_ids,
                top_k=self.settings.rag_top_k,
                min_similarity=self.settings.rag_min_similarity,
            )
        else:
            coverage = self.documents.retrieve_document_wide(
                request.document_ids, intent=intent, max_context_chars=LOCAL_CONTEXT_CHARS,
                page_scope=page_scope,
            )
            chunks = coverage.chunks
            system_prompt = (
                SUMMARY_SYSTEM_PROMPT if intent == "summary" else DOCUMENT_INSTRUCTIONS[intent]
            )

        insufficient_text = insufficient_answer(request.question)
        if coverage is not None and coverage.has_documents and not coverage.partial:
            if page_scope is not None and not chunks:
                insufficient_text = (
                    "لم أجد نصًا في الصفحات المطلوبة من المستندات المختارة."
                    if _in_arabic(request.question) else
                    "I couldn’t find text on the requested pages of the selected documents."
                )
            elif intent == "dates":
                insufficient_text = (
                    "لم أجد تواريخ أو أوقاتًا محددة ذات صلة في المستندات المختارة."
                    if _in_arabic(request.question) else
                    "I couldn’t verify any relevant dates or times in the selected documents."
                )
            elif intent == "deadlines":
                insufficient_text = (
                    "لم أجد مواعيد نهائية محددة في المستندات المختارة."
                    if _in_arabic(request.question) else
                    "I couldn’t verify any explicit deadlines in the selected documents."
                )
        insufficient = ChatResponse(
            status="insufficient_context",
            answer=insufficient_text,
            sources=[],
        )
        # Summary scope is explicit. Earlier answers can bias which sections
        # get covered and consume tokens without adding any source evidence.
        history = [] if intent == "summary" else request.history
        if debug:
            trace_fields(evidence="empty" if not chunks else "partial" if coverage and coverage.partial else "retrieved",
                         context_chunks=len(chunks), history_messages=len(history),
                         document_ids_in_context=sorted({chunk.document_id for chunk in chunks}),
                         pages_in_context=sorted({chunk.page_number for chunk in chunks}))
        prompt = build_prompt(request.question, history, chunks, summary=intent == "summary")
        if debug:
            log_context(prompt, chunks, self.settings.groq_api_key.get_secret_value())
        if not chunks:
            trace_fields(outcome="retrieval_no_evidence")
            return insufficient

        options = {"summary": True} if intent == "summary" else {}
        try:
            generated = self.generator.generate(system_prompt=system_prompt, user_prompt=prompt, **options)
        except ProviderError as error:
            failure = {
                "groq_output_limit": "provider_truncated",
                "groq_invalid_response": "structured_output_invalid",
                "groq_unverified_evidence": "verification_rejected",
            }.get(error.code, "structured_output_invalid" if error.output_failure else "provider_request_failed")
            trace_fields(outcome=failure, provider_error=error.code)
            if error.code == "groq_unverified_evidence":
                trace_fields(verification="failed", schema_validation="passed")
                if intent != "summary":
                    insufficient.answer = _verification_failure(request.question)
                    return insufficient
            if intent != "summary" or not error.output_failure:
                raise
            answer = None
        else:
            trace_fields(generator_status=generated.status, claim_count=len(generated.claims))
            if intent == "summary" and isinstance(generated, SummaryCandidates):
                answer = _verified_summary(generated, chunks, request.question)
            else:
                answer = _cited_answer(generated, chunks, intent=intent, arabic=_in_arabic(request.question),
                                       question=request.question)
            if answer is not None:
                trace_fields(outcome="generated", verification="passed", delivery="generated_answer")
            elif generated.status == "insufficient_context":
                trace_fields(outcome="evidence_incomplete", verification="not_run")
            else:
                trace_fields(outcome="verification_rejected", verification="failed")
                insufficient.answer = _verification_failure(request.question)
        if answer is None and intent == "summary":
            answer = _excerpt_answer(request.question, chunks)
            if answer is not None:
                trace_fields(delivery="labeled_excerpts")
        if answer is None:
            return insufficient
        if coverage is not None and coverage.partial:
            note = (
                "تغطي الإجابة مقتطفات مختارة ضمن حد السياق."
                if _in_arabic(request.question) else
                "This answer covers selected excerpts within the context limit."
            )
            answer.answer += f"\n\n{note}"
        return answer

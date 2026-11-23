"""Select bounded, citable evidence for document-wide questions."""

import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from heapq import nlargest
from itertools import islice
from typing import Literal

from app.rag.temporal import temporal_priority
from app.rag.types import RetrievedChunk


DocumentIntent = Literal["summary", "dates", "deadlines"]
SummaryPageScope = tuple[tuple[int, int], ...]
LOCAL_CONTEXT_CHARS = 12000
MAX_CONTEXT_CHUNKS = 24
MAX_SUMMARY_SECTIONS = 12

_EN_DOCUMENT = r"(?:(?:this|the|these|my) )?(?:(?:entire|whole|full|selected) ){0,2}(?:documents?|pdfs?|files?|reports?|text)"
_EN_SUMMARY = re.compile(
    rf"(?:(?:please )?(?:summari[sz]e|recap)(?: {_EN_DOCUMENT}"
    rf"| (?:the )?(?:main points|key points|key findings)(?: (?:of|in|from) {_EN_DOCUMENT})?)?"
    rf"|what (?:is|are) {_EN_DOCUMENT} about"
    r"|what are (?:the )?(?:main points|main topics|key points|key conclusions|key takeaways)"
    rf"(?: (?:discussed|covered))?(?: (?:in|of) {_EN_DOCUMENT})?"
    r"|(?:give me|provide) (?:a brief |a short |an? |the )?(?:overview|summary)"
    rf"(?: of {_EN_DOCUMENT})?)"
)
_AR_DOCUMENT = r"(?:(?:هذا|هذه) )?(?:المستند|الملف|الوثيقه|الوثائق|المستندات|الملفات)(?: (?:كامل[اه]?|بكامله[ا]?|كله[ا]?))?"
_AR_SUMMARY = re.compile(
    rf"(?:لخص(?: {_AR_DOCUMENT}"
    rf"| (?:اهم |ابرز )?(?:المعلومات|النقاط|النتائج|المواضيع)(?: (?:الموجوده|الوارده))?(?: في {_AR_DOCUMENT})?)?"
    r"|ما (?:هي )?(?:اهم|ابرز|خلاصه) (?:النقاط|المواضيع|الموضوعات|النتائج|الاستنتاجات)"
    r"(?: في (?:هذا |هذه )?(?:المستند|الملف|الوثيقه|الملفات|المستندات))?"
    r"|(?:اعطني|قدم) (?:ملخصا?|نظره عامه)(?: (?:عن|لـ?) (?:هذا |هذه )?(?:المستند|الملف|الوثيقه))?)"
)
_PAGE_RANGE_SEPARATOR = r"(?:[-–—−]|\b(?:to|through|الى|حتى)\b)"
_PAGE_RANGE = rf"\d+(?:\s*{_PAGE_RANGE_SEPARATOR}\s*\d+)?"
_PAGE_LIST = rf"{_PAGE_RANGE}(?:\s*(?:[,،]\s*(?:\band\b|و)?|&|\band\b|و)\s*{_PAGE_RANGE})*"
_PAGE_NUMBERS = re.compile(rf"(\d+)(?:\s*{_PAGE_RANGE_SEPARATOR}\s*(\d+))?")
_EN_PAGE_SUMMARY = re.compile(
    r"(?:(?:please )?(?:summari[sz]e|recap) (?:the (?:main points|key points) (?:of|on|from) )?"
    r"|(?:give me|provide) (?:an? |the )?(?:summary|overview) (?:of|for|on) "
    r"|what are (?:the )?(?:main points|main topics|key conclusions) (?:in|on|of) )"
    rf"(?:(?:only|just) )?(?:pages?|pp?\.?) (?P<pages>{_PAGE_LIST})"
    r"(?: only)?"
    r"(?: (?:of|in|from) (?:this|the|these|selected|the selected|my) (?:documents?|pdfs?|files?|reports?))?"
    r"(?: (?:only|just))?"
)
_AR_PAGE_SUMMARY = re.compile(
    r"(?:تلخيص |لخص |(?:اعطني|قدم) (?:ملخصا?|نظره عامه) (?:عن |ل ?)?"
    r"|ما (?:هي )?(?:اهم|ابرز) (?:النقاط|المواضيع|النتائج) في )"
    rf"(?:فقط )?(?:ال)?صفح(?:ه|ات) (?P<pages>{_PAGE_LIST})"
    r"(?: فقط)?"
    r"(?: من (?:هذا |هذه )?(?:المستند|الملف|الوثيقه)(?: (?:المحدد|المختار))?)?(?: فقط)?"
)
_SUMMARY_FORMAT = re.compile(
    r"(?: (?:in|using) (?:(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|a few) )?"
    r"(?:short |concise )?(?:bullet points?|bullets?|sentences?|paragraphs?)"
    r"| (?:في|ضمن|على شكل) (?:(?:\d+|واحده?|نقطتين|ثلاثه?|اربعه?|خمسه?|سته?|عده) )?"
    r"(?:نقاط|جمل|فقرات)(?: موجزه| قصيره)?| في نقطه واحده| باختصار)$"
)
_SUMMARY_CITATIONS = re.compile(
    r"(?: with (?:page |source )?citations| and cite (?:the )?(?:relevant )?(?:sources|pages)| مع ذكر المصادر)$"
)
_SUMMARY_LANGUAGE = re.compile(r"(?: in (?:english|arabic)| باللغه (?:العربيه|الانجليزيه)| ب(?:العربيه|الانجليزيه))$")
_EN_TEMPORAL = re.compile(
    r"(?:(?:what|which) (?:are )?|are there |are any |list |show me |tell me |give me )"
    r"(?:any |the |all |about )*(?:important |key )?(?:dates|deadlines|times)"
    r"(?: (?:and|or) (?:dates|deadlines|times))*"
    r"(?: (?:should (?:i|we) know|(?:i|we) should know(?: about)?|(?:are )?(?:mentioned|listed|included)|appear))?"
    r"(?: in (?:this|the|these|the selected) (?:documents?|pdfs?|files?|reports?))?"
    r"(?:,? and (?:what (?:does each(?: one)?|do they) (?:represent|mean|refer to)"
    r"|(?:explain|describe) (?:what each(?: one)? (?:represents|means)|(?:the )?events?)))?"
)
_AR_TEMPORAL = re.compile(
    r"(?:(?:ما (?:هي )?|هل (?:هناك|توجد) |اذكر |اعرض |حدد )"
    r"(?:(?:اهم|ابرز|اي) )?(?:التواريخ|تواريخ|المواعيد|مواعيد|المهل|مهل))"
    r"(?: و(?:التواريخ|المواعيد|المهل))?"
    r"(?: (?:المهمه|الهامه|الرئيسيه|النهائيه|نهائيه|المذكوره|الوارده))*"
    r"(?: في (?:هذا |هذه )?(?:المستند|الملف|الوثيقه|المستندات|الملفات))?"
)
_WORD = re.compile(r"[^\W\d_]{3,}")
_ARABIC_VARIANTS = str.maketrans("أإآٱة", "ااااه")
_METADATA_LINE = re.compile(
    r"(?:\d+|(?:page|صفحه|الصفحه)\s*\d+(?:\s*(?:of|من|/)\s*\d+)?|"
    r"(?:copyright|all rights reserved|حقوق النشر|جميع الحقوق محفوظه|isbn|issn)\b.*|"
    r"(?:https?://|www\.)\S+|(?:editor|publisher|published|last updated|version|"
    r"publication date|المحرر|الناشر|تاريخ النشر|اخر تحديث|الاصدار)\s*[:：].*)",
    re.IGNORECASE,
)
_TOPIC_HEADING = re.compile(
    r"^[ \t]*(?:(?:executive )?summary|key (?:findings|conclusions|results)|findings|conclusions?|"
    r"recommendations?|results|ملخص(?: تنفيذي)?|النتائج|الاستنتاجات|التوصيات)[ \t]*(?:[:：]|$)",
    re.IGNORECASE | re.MULTILINE,
)
_EXCERPT_BOUNDARY = re.compile(r"(?<=[.!?؟])\s+|\n+")


def _normalize(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", question).casefold().translate(_ARABIC_VARIANTS)
    return " ".join(
        "".join(char for char in normalized if not unicodedata.combining(char) and char != "ـ").split()
    ).strip(" .?!؟")


def _summary_subject(normalized: str) -> str:
    # Presentation qualifiers may appear in either order. Leave any substantive
    # topic or named entity in place so a focused summary keeps normal retrieval.
    subject = normalized
    while True:
        for modifier in (_SUMMARY_CITATIONS, _SUMMARY_FORMAT, _SUMMARY_LANGUAGE):
            subject = modifier.sub("", subject).rstrip(" ,،؛")
        if subject == normalized:
            return subject
        normalized = subject


def _summary_page_scope(subject: str) -> SummaryPageScope | None:
    match = _EN_PAGE_SUMMARY.fullmatch(subject) or _AR_PAGE_SUMMARY.fullmatch(subject)
    if not match:
        return None
    ranges: list[tuple[int, int]] = []
    for first, last in _PAGE_NUMBERS.findall(match["pages"]):
        start, end = int(first), int(last or first)
        ranges.append((min(start, end), max(start, end)))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


def summary_page_scope(question: str) -> SummaryPageScope | None:
    """Find explicit inclusive page numbers in a supported summary request."""
    return _summary_page_scope(_summary_subject(_normalize(question)))


def document_intent(question: str) -> DocumentIntent | None:
    normalized = _normalize(question)
    subject = _summary_subject(normalized)
    if _EN_SUMMARY.fullmatch(subject) or _AR_SUMMARY.fullmatch(subject) or _summary_page_scope(subject):
        return "summary"
    if _EN_TEMPORAL.fullmatch(normalized) or _AR_TEMPORAL.fullmatch(normalized):
        deadline = re.search(r"\b(?:deadlines|المهل|مهل|النهائيه|نهائيه)\b", normalized)
        date = re.search(r"\b(?:dates|times|التواريخ|تواريخ)\b", normalized)
        if deadline and not date:
            return "deadlines"
        return "dates"
    return None


@dataclass(frozen=True, slots=True)
class DocumentCoverage:
    chunks: list[RetrievedChunk]
    has_documents: bool
    partial: bool = False


def _page_order(chunk: RetrievedChunk) -> tuple[int, int, str]:
    index = chunk.chunk_id.rsplit(":", 1)[-1]
    return chunk.page_number, int(index) if index.isascii() and index.isdecimal() else 0, chunk.chunk_id


def _bounded(chunks: Iterable[RetrievedChunk], max_chars: int) -> list[RetrievedChunk]:
    chosen = []
    for chunk in chunks:
        if len(chunk.text) > max_chars:
            continue
        chosen.append(chunk)
        max_chars -= len(chunk.text)
        if len(chosen) == MAX_CONTEXT_CHUNKS:
            break
    return chosen


def _section_excerpts(chunks: Iterable[RetrievedChunk], max_chars: int) -> list[RetrievedChunk]:
    limit = min(MAX_CONTEXT_CHUNKS, max(1, max_chars // 250))
    sections = list(islice(chunks, limit))
    if sum(len(chunk.text) for chunk in sections) <= max_chars:
        return sections

    # Share the remaining budget across sections so a long first passage
    # cannot consume the entire prompt. Keep contiguous, verbatim excerpts.
    excerpts = []
    for index, chunk in enumerate(sections):
        share = max_chars // (len(sections) - index)
        text = chunk.text
        if len(text) > share:
            # A source may begin with publication metadata. If clipping is
            # needed, start the excerpt at the first usable line instead.
            for line in text.splitlines(keepends=True):
                if line.strip() and not _METADATA_LINE.fullmatch(_normalize(line)):
                    break
                text = text[len(line):]
            if not text:
                text = chunk.text
            text = text[:share]
            boundary = max(text.rfind(" "), text.rfind("\n"))
            if boundary >= share // 2:
                text = text[:boundary]
            text = text.rstrip()
        excerpts.append(replace(chunk, text=text))
        max_chars -= len(text)
    return excerpts


def _scope_pages(page_count: int, page_scope: SummaryPageScope | None) -> SummaryPageScope:
    if page_scope is None:
        return ((1, page_count),)
    return tuple(
        (max(1, start), min(page_count, end))
        for start, end in page_scope
        if start <= page_count and end >= 1
    )


def _page_position(page: int, ranges: SummaryPageScope) -> int | None:
    position = 0
    for start, end in ranges:
        if page < start:
            break
        if page <= end:
            return position + page - start
        position += end - start + 1
    return None


def _summary_salience(text: str) -> tuple[bool, int, int]:
    body = "\n".join(
        line for line in text.splitlines()
        if line.strip() and not _METADATA_LINE.fullmatch(_normalize(line))
    )
    prose, headings = _TOPIC_HEADING.subn("", body)
    return bool(prose.strip()), headings, len(set(_WORD.findall(prose.casefold())))


def summary_excerpts(chunks: list[RetrievedChunk], *, limit: int = 4) -> list[tuple[int, str]]:
    """Choose useful verbatim lines across the supplied pages."""
    candidates = []
    seen: set[str] = set()
    for source_id, chunk in enumerate(chunks, start=1):
        for line in _EXCERPT_BOUNDARY.split(chunk.text):
            quote = line.strip()
            if len(quote) > 300:
                quote = quote[:301].rsplit(maxsplit=1)[0]
            score = _summary_salience(quote)
            if not score[0] or score[2] < 4 or quote in seen or len(quote) > 300:
                continue
            seen.add(quote)
            candidates.append((score, source_id, quote))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    seen_pages: set[tuple[str, int]] = set()
    remainder = []
    for _, source_id, quote in candidates:
        chunk = chunks[source_id - 1]
        page = (chunk.document_id, chunk.page_number)
        if page in seen_pages:
            remainder.append((source_id, quote))
        else:
            seen_pages.add(page)
            selected.append((source_id, quote))
            if len(selected) == limit:
                break
    return sorted((selected + remainder)[:limit], key=lambda item: item[0])


@dataclass(slots=True)
class _PageBand:
    primary: RetrievedChunk
    score: tuple[bool, int, int]
    first: RetrievedChunk | None = None
    last: RetrievedChunk | None = None

    def consider(self, chunk: RetrievedChunk, score: tuple[bool, int, int]) -> None:
        order = _page_order(chunk)
        if score > self.score or (score == self.score and order < _page_order(self.primary)):
            self.primary = chunk
            self.score = score
        if score[0]:
            if self.first is None or order < _page_order(self.first):
                self.first = chunk
            if self.last is None or order > _page_order(self.last):
                self.last = chunk


def _band_excerpts(bands: Iterable[_PageBand]) -> Iterator[RetrievedChunk]:
    bands = list(bands)
    seen: set[tuple[str, str]] = set()
    for position in ("primary", "first", "last"):
        for band in bands:
            chunk = getattr(band, position)
            if chunk is not None and (key := (chunk.document_id, chunk.chunk_id)) not in seen:
                seen.add(key)
                yield chunk


def _summary_chunks(
    chunks: Iterable[RetrievedChunk], page_counts: Mapping[str, int], max_chars: int,
    page_scope: SummaryPageScope | None,
) -> DocumentCoverage:
    complete: list[RetrievedChunk] = []
    total_chars = count = 0
    # Map each selected PDF into contiguous page ranges. Keep a representative
    # excerpt per range while scanning so the whole index never enters the prompt.
    total_sections = min(MAX_SUMMARY_SECTIONS, max(1, max_chars // 750))
    requested_ranges = {doc_id: _scope_pages(count, page_scope) for doc_id, count in page_counts.items()}
    requested_page_counts = {
        doc_id: sum(end - start + 1 for start, end in ranges) for doc_id, ranges in requested_ranges.items()
    }
    covered_documents = sum(count > 0 for count in requested_page_counts.values())
    if covered_documents == 0:
        return DocumentCoverage([], True)
    sections_per_document = max(1, total_sections // covered_documents)
    sections: dict[str, dict[int, _PageBand]] = {doc_id: {} for doc_id in page_counts}
    for chunk in chunks:
        position = _page_position(chunk.page_number, requested_ranges[chunk.document_id])
        if position is None:
            continue
        count += 1
        total_chars += len(chunk.text)
        if count <= MAX_CONTEXT_CHUNKS and total_chars <= max_chars:
            complete.append(chunk)
        page_count = requested_page_counts[chunk.document_id]
        band_count = min(sections_per_document, page_count)
        band = position * band_count // page_count
        score = _summary_salience(chunk.text)
        group = sections[chunk.document_id].get(band)
        if group is None:
            group = _PageBand(chunk, score)
            sections[chunk.document_id][band] = group
        group.consider(chunk, score)

    if count == len(complete):
        doc_order = {doc_id: index for index, doc_id in enumerate(page_counts)}
        complete.sort(key=lambda chunk: (doc_order[chunk.document_id], _page_order(chunk)))
        return DocumentCoverage(complete, True)

    # Reduce the regional excerpts into one bounded prompt. Give each selected
    # document a turn before taking additional excerpts from any one document.
    bands = (
        entries[band]
        for band in range(sections_per_document)
        for entries in sections.values()
        if band in entries
    )
    excerpts = _section_excerpts(_band_excerpts(bands), max_chars)
    doc_order = {doc_id: index for index, doc_id in enumerate(page_counts)}
    excerpts.sort(key=lambda chunk: (doc_order[chunk.document_id], _page_order(chunk)))
    return DocumentCoverage(excerpts, True, partial=True)


def document_coverage(
    chunks: Iterable[RetrievedChunk],
    page_counts: Mapping[str, int],
    intent: DocumentIntent,
    max_chars: int,
    *,
    page_scope: SummaryPageScope | None = None,
) -> DocumentCoverage:
    if not page_counts:
        return DocumentCoverage([], False)
    if intent == "summary":
        return _summary_chunks(chunks, page_counts, max_chars, page_scope)

    count = 0

    def candidates():
        nonlocal count
        for chunk in chunks:
            score = temporal_priority(chunk.text, deadlines_only=intent == "deadlines")
            if score:
                count += 1
                yield score, chunk

    ranked = nlargest(MAX_CONTEXT_CHUNKS, candidates(), key=lambda item: item[0])
    chosen = _bounded((chunk for _, chunk in ranked), max_chars)
    return DocumentCoverage(chosen, True, partial=len(chosen) < count)

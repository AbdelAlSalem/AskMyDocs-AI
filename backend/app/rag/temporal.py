"""Find explicit dates and their source labels without inferring event descriptions."""

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date


MONTHS = (
    ("january", "jan", "يناير", "كانون الثاني", "جانفي"),
    ("february", "feb", "فبراير", "شباط", "فيفري"),
    ("march", "mar", "مارس", "اذار"),
    ("april", "apr", "ابريل", "نيسان", "افريل"),
    ("may", "مايو", "ايار"),
    ("june", "jun", "يونيو", "حزيران", "جوان"),
    ("july", "jul", "يوليو", "تموز", "جويلية"),
    ("august", "aug", "اغسطس", "اب", "اوت", "غشت"),
    ("september", "sep", "sept", "سبتمبر", "ايلول", "شتنبر"),
    ("october", "oct", "اكتوبر", "تشرين الاول"),
    ("november", "nov", "نوفمبر", "تشرين الثاني", "نونبر"),
    ("december", "dec", "ديسمبر", "كانون الاول", "دجنبر"),
)
MONTH_NUMBER = {name: number for number, names in enumerate(MONTHS, start=1) for name in names}
MONTH_NAMES = "|".join(
    re.escape(name).replace(r"\ ", r"\s+") for name in sorted(MONTH_NUMBER, key=len, reverse=True)
)
CHARACTERS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹ـى‐‑–⁄", "01234567890123456789 ي---/",
)
DAY = r"[0-9]{1,2}(?:st|nd|rd|th)?"
YEAR = r"[1-9][0-9]{3}"
DAY_MONTH = re.compile(
    rf"(?<![\w./-])(?P<day>{DAY})\s+(?:(?:of|من)\s+)?(?P<month>{MONTH_NAMES})\b\.?"
    rf"(?:[,،]?\s+(?:(?:عام|سنة)\s+)?(?P<year>{YEAR})\b)?"
)
MONTH_DAY = re.compile(
    rf"\b(?P<month>{MONTH_NAMES})\b\.?\s+(?P<day>{DAY})\b"
    rf"(?:[,،]?\s+(?:(?:عام|سنة)\s+)?(?P<year>{YEAR})\b)?"
)
MONTH_YEAR = re.compile(rf"\b(?:{MONTH_NAMES})\b\.?\s+(?:(?:عام|سنة)\s+)?{YEAR}\b(?![./:-][0-9])")
DAY_BEFORE_MONTH = re.compile(rf"\b{DAY}\s+(?:(?:of|من)\s+)?$")
HIJRI_DATE = re.compile(
    r"(?<![\w./-])(?:[1-9]|[12][0-9]|30)\s+(?:محرم|صفر|ربيع\s+(?:الاول|الاخر|الثاني)|"
    r"جمادي\s+(?:الاولي|الاول|الاخرة|الثانية)|رجب|شعبان|رمضان|شوال|ذ[وي]\s+(?:القعدة|الحجة))\b"
)
NUMERIC_DATE = re.compile(
    rf"(?<![\w./:-])(?P<a>{YEAR}|[0-9]{{1,2}})\s*(?P<sep>[-/.])\s*(?P<b>[0-9]{{1,2}})\s*(?P=sep)\s*"
    rf"(?P<c>{YEAR}|[0-9]{{1,2}})(?=t[0-9]{{2}}:|[^\w./:-]|[.:](?![0-9])|$)"
)
NON_DATE_LABEL = re.compile(r"\b(?:version|firmware|software build|رقم الاصدار|نطاق)\s*[:=]?\s*$")
NON_CLOCK_LABEL = re.compile(r"\b(?:ratio|reference|port|vlan|version|النسبة|المنفذ)\s*(?:is\s+|هي\s+)?[:=]?\s*$")
CONTEXTUAL_YEAR = re.compile(
    r"\b(?:in|since|during|through|until|year|founded|established|published|q[1-4]|[ولب]?عام|سنة)\s*:?\s*"
    r"(?P<year>1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b(?![./:-][0-9])"
)
CLOCK = re.compile(
    r"(?<![\w:./-])(?:[01]?[0-9]|2[0-3]):[0-5][0-9]\s*-\s*(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?![\w:]|[./-][0-9])"
    r"|(?<![\w:./-])(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?(?![\w:]|[./-][0-9])"
    r"|(?<![\w:./-])(?:1[0-2]|0?[1-9])\s*[ap]\.?m\b\.?"
    r"|\b(?:at|by|before|until|after)\s+(?:noon|midnight)\b"
    r"|\bالساعة\s+(?:[01]?[0-9]|2[0-3])\b(?!:[0-9])"
    r"|\bالساعة\s+(?:الواحدة|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|"
    r"التاسعة|العاشرة|الحادية عشرة|الثانية عشرة)\b"
)
ISO_TIME = re.compile(
    r"t(?:[01][0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9](?:\.[0-9]+)?)?"
    r"(?:z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])?\b"
)
WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
ARABIC_WEEKDAYS = "الاثنين|الثلاثاء|الاربعاء|الخميس|الجمعة|السبت|الاحد"
RELATIVE_DATE = re.compile(
    rf"\b(?:today|tomorrow|yesterday|اليوم|غدا|امس)\b"
    rf"|\b(?:next|last|this|every)\s+(?:{WEEKDAYS}|week|month|year)\b"
    rf"|\b(?:on|by|before|until|after)\s+(?:{WEEKDAYS})\b"
    rf"|\b(?:يوم|حتى|قبل|بحلول)\s+(?:{ARABIC_WEEKDAYS})\b"
    r"|\b(?:الاسبوع|الشهر|العام|السنة)\s+(?:المقبل[ة]?|القادم[ة]?|الماضي[ة]?)\b"
)
RELATIVE_INTERVAL = re.compile(
    r"\b(?:within|in)\s+[1-9][0-9]*\s+(?:minutes?|hours?|days?|weeks?|months?|years?)\b"
    r"|\bخلال\s+[1-9][0-9]*\s+(?:دقيقة|دقائق|ساعة|ساعات|يوم|ايام|اسبوع|اسابيع|شهر|اشهر|عام|سنوات)\b"
)
DEADLINE_CUE = re.compile(
    r"\b(?:deadlines?|cut[- ]?offs?|due(?!\s+to\b)|no later than|expires?|expiration|"
    r"closing date|(?:submit|apply|file|register|renew|pay|deliver)\s+(?:by|before|within)|"
    r"(?:registration|applications?|submissions?)\s+(?:will\s+)?clos(?:e|es)|"
    r"الموعد النهائي|موعد نهائي|اخر موعد|مهلة|بحلول|تنتهي صلاحية|انتهاء الصلاحية|"
    r"(?:قدم|ارسل|سدد|جدد|سجل)\s+(?:\w+\s+){0,3}(?:قبل|خلال)|"
    r"(?:يغلق|ينتهي|يقفل)\s+(?:باب\s+)?(?:التسجيل|التقديم))\b"
)
DEADLINE_HEADING = re.compile(
    r"(?:(?:application|filing|payment|registration|renewal|submission)\s+)?(?:deadline|closing date)"
    r"|(?:الموعد النهائي|اخر موعد)(?:\s+(?:للتسجيل|للتقديم|للسداد))?"
)
EVENT_CUE = re.compile(
    r"\b(?:launch(?:es|ed)?|releases?|meetings?|scheduled|starts?|opens?|workshops?|"
    r"effective(?: date)?|اطلاق|انطلاق|اجتماع|ورشة|يبدا|تبدا|افتتاح|سريان)\b"
)
_NEW_CLAUSE = r"[,،]\s+(?=(?:but|while|whereas|and|لكن|بينما|أما)\b)"
BOUNDARY = re.compile(r"[.!?؟؛;]\s+|\n|" + _NEW_CLAUSE)


def _valid_date(year: int, month: int, day: int) -> bool:
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold()).translate(CHARACTERS)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _calendar_matches(text: str) -> Iterator[re.Match[str]]:
    for match in NUMERIC_DATE.finditer(text):
        if NON_DATE_LABEL.search(text[max(0, match.start() - 40):match.start()]):
            continue
        a, b, c = (match[name] for name in ("a", "b", "c"))
        if len(a) == 4 and len(c) < 4:
            if _valid_date(int(a), int(b), int(c)):
                yield match
        elif len(c) == 4 and len(a) < 4:
            if _valid_date(int(c), int(a), int(b)) or _valid_date(int(c), int(b), int(a)):
                yield match

    for pattern in (DAY_MONTH, MONTH_DAY):
        for match in pattern.finditer(text):
            day = int(re.sub(r"[a-z]+", "", match["day"]))
            month = MONTH_NUMBER[" ".join(match["month"].split())]
            year = int(match["year"] or 2000)
            if _valid_date(year, month, day):
                yield match


def _context(text: str, match: re.Match[str]) -> str:
    before = BOUNDARY.split(text[max(0, match.start() - 90):match.start()])
    after = BOUNDARY.split(text[match.end():match.end() + 90], maxsplit=1)[0]
    preceding = before[-1]
    # Extractors can place a field label immediately above its value.
    if not preceding.strip() and len(before) > 1:
        label = before[-2].strip().removesuffix(":")
        if DEADLINE_HEADING.fullmatch(label) or DEADLINE_CUE.fullmatch(label) or EVENT_CUE.fullmatch(label):
            preceding = label + " "
    return preceding + match[0] + after


def _clock_matches(text: str) -> Iterator[re.Match[str]]:
    for match in CLOCK.finditer(text):
        if not NON_CLOCK_LABEL.search(text[max(0, match.start() - 30):match.start()]):
            yield match


def _temporal_matches(text: str) -> Iterator[tuple[str, int, re.Match[str]]]:
    month_year = (
        match for match in MONTH_YEAR.finditer(text)
        if not DAY_BEFORE_MONTH.search(text[max(0, match.start() - 18):match.start()])
    )
    groups = (
        ("date", 3, _calendar_matches(text)),
        ("date", 3, HIJRI_DATE.finditer(text)),
        ("date", 2, month_year),
        ("year", 2, CONTEXTUAL_YEAR.finditer(text)),
        ("time", 1, _clock_matches(text)),
        ("date", 1, RELATIVE_DATE.finditer(text)),
        ("interval", 0, RELATIVE_INTERVAL.finditer(text)),
    )
    for kind, weight, matches in groups:
        for match in matches:
            context = _context(text, match)
            if DEADLINE_CUE.search(context):
                yield "deadline", weight + 3, match
            elif weight:
                yield kind, weight + bool(EVENT_CUE.search(context)), match
                if kind == "date" and (stamp := ISO_TIME.match(text, match.end())):
                    yield "time", 1, stamp


def temporal_priority(text: str, *, deadlines_only: bool = False) -> int:
    """Rank explicit temporal evidence without turning ordinary numbers into dates."""
    return max((
        weight for kind, weight, _ in _temporal_matches(_normalize(text))
        if not deadlines_only or kind == "deadline"
    ), default=0)


@dataclass(frozen=True, slots=True)
class TemporalRow:
    text: str
    kind: str
    labeled: bool


_ROW_BOUNDARY = re.compile(r"(?<=[.!?؟;؛])\s+|\n+|" + _NEW_CLAUSE, re.IGNORECASE)
_MONTH_ABBREVIATION = re.compile(r"\b(?:jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\.$", re.IGNORECASE)
_FIELD_NAME = re.compile(r"(?:[\w-]+ ){0,4}(?:date|year|time|deadline|hours)|(?:تاريخ|سنة|وقت|مهلة)(?: \w+){0,4}")
_VALUE_AT_START = re.compile(rf"^(?:[0-9]|(?:{MONTH_NAMES})\b)")
_GENERIC_DATE_WORD = re.compile(r"\b(?:the|a|on|in|at|of|date|day|year|time|is|was|بتاريخ|تاريخ|عام|سنة|في|يوم)\b")
_TEMPORAL_LABELS = {
    "date": ("Date", "تاريخ"), "year": ("Year", "سنة"), "time": ("Time", "وقت"),
    "deadline": ("Deadline", "موعد نهائي"), "date_time": ("Date and time", "تاريخ ووقت"),
}


def _row_spans(passage: str) -> Iterator[tuple[int, int]]:
    start = 0
    for boundary in _ROW_BOUNDARY.finditer(passage):
        if _MONTH_ABBREVIATION.search(passage[:boundary.start()]) and passage[boundary.end():boundary.end() + 1].isdigit():
            continue
        yield start, boundary.start()
        start = boundary.end()
    yield start, len(passage)


def temporal_evidence_rows(passage: str, quote: str, *, deadlines_only: bool = False) -> list[TemporalRow]:
    """Expand a verified temporal quote to its contiguous source sentence or table row."""
    tokens = quote.split()
    if not tokens:
        return []
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    if tokens[0][0].isalnum():
        pattern = r"(?<!\w)" + pattern
    if tokens[-1][-1].isalnum():
        pattern += r"(?!\w)"
    citations = list(re.finditer(pattern, passage))
    if not citations:
        return []

    rows = []
    for start, end in _row_spans(passage):
        segment = passage[start:end]
        overlaps = [
            passage[max(start, citation.start()):min(end, citation.end())]
            for citation in citations if start < citation.end() and end > citation.start()
        ]
        if overlaps:
            line = segment.strip()
            preceding = passage[:start].rstrip().splitlines()
            heading = preceding[-1].strip() if preceding else ""
            # Retain the printed label above a value or the heading of a table row.
            value_led = _VALUE_AT_START.match(_normalize(line))
            field_above = value_led and (heading.endswith((":", "：")) or _FIELD_NAME.fullmatch(_normalize(heading)))
            table_heading = "|" in line and re.match(r"^\d+[.)]\s+\S", heading)
            if len(heading) <= 100 and (field_above or table_heading):
                line = heading + "\n" + line
            normalized = _normalize(" ".join(line.split()))
            cited = [_normalize(" ".join(overlap.split())) for overlap in overlaps]
            matches = [
                (kind, match) for kind, _, match in _temporal_matches(normalized)
                if (not deadlines_only or kind == "deadline") and any(
                    match[0] in excerpt
                    or kind == "year" and match["year"] == excerpt.strip(".,: ")
                    for excerpt in cited
                )
            ]
            if matches:
                kinds = {kind for kind, _ in matches}
                if "deadline" in kinds:
                    kind = "deadline"
                elif "time" in kinds and "date" in kinds:
                    kind = "date_time"
                else:
                    kind = matches[0][0]
                label = normalized
                for matched_kind, match in matches:
                    label = label.replace(match["year"] if matched_kind == "year" else match[0], " ")
                labeled = bool(re.search(r"[^\W\d_]{2,}", _GENERIC_DATE_WORD.sub(" ", label)))
                rows.append(TemporalRow(line, kind, labeled))
    return rows


def render_temporal_row(row: TemporalRow, *, arabic: bool = False, generated_text: str = "") -> str:
    """Keep the event and its value together, including for a bare-date provider answer."""
    label = _TEMPORAL_LABELS[row.kind][int(arabic)]
    original = " — ".join(" ".join(line.split()) for line in row.text.splitlines())
    if generated_text.lstrip("- *").casefold() == original.casefold():
        original = generated_text.lstrip("- *")
    if not row.labeled:
        note = "لم يذكر المصدر الحدث المرتبط به" if arabic else "the source gives no event description"
        return f"- {label} — {original} ({note})"
    return f"- {label} — {original}"

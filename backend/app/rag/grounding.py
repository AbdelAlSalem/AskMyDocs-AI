"""Conservative checks for unrelated prose attached to otherwise valid citations."""

import re
import unicodedata


_WORDS = re.compile(r"[^\W\d_]+")
_FUNCTION_WORDS = frozenset("""
    the this that they their there these those from with which what when were have
    has had are was for and but also into its according document passage passages
    source sources says stated answer based mentions mentioned yes only information
    هي هو هذا هذه ذلك تلك على عن إلى الى التي الذي في من أن ان مع كما المستند
""".split())
_LOCATION_ROLE = re.compile(
    r"(?P<headquarters>\b(?:headquarter(?:s|ed)|hq|head office|main office)\b|"
    r"(?:ال)?مقر(?:ها|ه| الشركة)?(?: الرئيسي)?)|"
    r"(?P<office>\b(?:office|branch)\b|(?:ال)?مكتب(?:ها|ه)?|(?:ال)?فرع(?:ها|ه)?)"
)
_LOCATION_START = re.compile(
    r"^\s*(?:(?:(?:is|are|was|were|located|situated|based)\s+)*(?:in|at)\s+|"
    r"[^,،.;؛!?؟]{0,60}?\bفي\s+|[:：]\s*)"
)
_CLAUSE_END = re.compile(r"[.;؛!?؟]|\b(?:and|where|which|with|while|employing)\b|،\s*و|\s+مع\s+")
_MISSED = re.compile(
    r"\b(?:missed|exceeded|did not meet|failed to meet)\b(?:[\w' -]+?\s)?(?:limit|target)\b|"
    r"\b(?:limit|target) was (?:missed|exceeded|not met)\b|"
    r"(?<!لم )(?:متجاوزا?|تجاوزت?|يتجاوز|تخطي|تخطى|لم يحقق|لم تحقق)\s+(?:ال)?(?:هدف|حد)"
)
_MET = re.compile(
    r"\b(?:met|within)\b(?:[\w' -]+?\s)?(?:limit|target)\b|"
    r"\b(?:limit|target) was met\b|"
    r"(?:حقق(?:ت)?|ضمن|دون|قبل|لم يتجاوز|لم تتجاوز)\s+(?:ال)?(?:هدف|حد)"
)
_PREDICATE = re.compile(r"\b(?:has|have|offers?|provides?|pays?|includes?|owns?)\s+([^.;!?]+)")
_EQUIVALENTS = {
    form: concept for concept, forms in {
        "restore": ("restored", "restoration", "recovery", "recovered"),
        "found": ("founded", "formed", "established", "incorporated"),
        "approve": ("accepted", "approved", "approval"),
    }.items() for form in forms
}


def _clean(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", text.casefold())
                   if not unicodedata.combining(char))


def _words(text: str) -> set[str]:
    return {_EQUIVALENTS.get(word, word) for word in _WORDS.findall(text)
            if len(word) > 1 and word not in _FUNCTION_WORDS}


def _arabic(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return sum(unicodedata.name(char, "").startswith("ARABIC") for char in letters) * 2 > len(letters)


def _locations(text: str) -> dict[str, list[str]]:
    """Bind a place to its own role, stopping before a subsequent role or clause."""
    roles = list(_LOCATION_ROLE.finditer(text))
    result: dict[str, list[str]] = {}
    for index, role in enumerate(roles):
        tail = text[role.end():roles[index + 1].start() if index + 1 < len(roles) else len(text)]
        locations = result.setdefault(role.lastgroup, [])
        if start := _LOCATION_START.match(tail):
            location = _CLAUSE_END.split(tail[start.end():], maxsplit=1)[0].strip(" ,،")
            if location:
                locations.append(location)
    return result


def _relations_supported(claim: str, quotes: list[str]) -> bool:
    observed = _locations(claim)
    if observed:
        locations = [_locations(quote) for quote in quotes]
        for role, places in observed.items():
            if not places:
                continue
            candidates = [place for item in locations for place in item.get(role, [])]
            if not candidates:
                return False
            for place in places:
                # Compare the first named place, retaining country qualifiers in
                # the evidence. Literal place matching applies within a script.
                name = re.split(r"[,،]", place, maxsplit=1)[0].strip()
                comparable = [candidate for candidate in candidates if _arabic(name) == _arabic(candidate)]
                if comparable and not any(
                    re.search(rf"(?<!\w){re.escape(name)}(?!\w)", candidate) for candidate in comparable
                ):
                    return False
    # An Arabic 'exceeded the target' can imply a missed maximum. Require a
    # consistent sign whenever the cited material gives an explicit verdict.
    evidence = " ".join(quotes)
    if _MISSED.search(claim) and not _MET.search(claim) and _MET.search(evidence) and not _MISSED.search(evidence):
        return False
    return not (_MET.search(claim) and not _MISSED.search(claim)
                and _MISSED.search(evidence) and not _MET.search(evidence))


def _shares_term(words: set[str], evidence: set[str]) -> bool:
    # Allow common inflections without substring matches between short names.
    return bool(words & evidence or {w[:5] for w in words if len(w) >= 5}
                & {w[:5] for w in evidence if len(w) >= 5})


def wording_supported(claim: str, quotes: list[str], question: str, *, figures: bool) -> bool:
    claim = _clean(claim)
    quotes = [_clean(quote) for quote in quotes]
    if not _relations_supported(claim, quotes):
        return False
    evidence = _words(" ".join(quotes))
    words = _words(claim)
    if not words:
        return figures
    # Literal word overlap is meaningful only when claim and evidence use the
    # same script. Arabic translations of English evidence may share no words.
    if _arabic(" ".join(words)) != _arabic(" ".join(evidence)):
        return True
    if not evidence:
        return figures
    if not _shares_term(words, evidence):
        return False
    # Sharing only a subject ("Acme offers ...") cannot establish an unrelated
    # benefit or property. Compare the object independently on short assertions.
    if predicate := _PREDICATE.search(claim):
        objects = _words(predicate[1])
        if objects and not figures and not _shares_term(objects, evidence):
            return False
    # Repeating the queried entity doesn't support a newly invented property.
    asserted = words - _words(_clean(question))
    return figures or not asserted or _shares_term(asserted, evidence)

import re
from dataclasses import dataclass
from itertools import pairwise

from app.rag.types import RetrievedChunk

TOKEN = re.compile(r"[^\W_]+(?:[.-][^\W_]+)*")
SOURCE_INSTRUCTION = r"(?:cite|include)\s+(?:the\s+)?(?:relevant\s+)?(?:pages|sources|citations)"
PRESENTATION_SUFFIX = re.compile(
    r"(?<=[?؟])\s+(?:please\s+)?(?:"
    r"(?:explain|show)\s+(?:the|your)\s+(?:calculation|reasoning|work)"
    rf"(?:\s+and\s+{SOURCE_INSTRUCTION})?"
    rf"|{SOURCE_INSTRUCTION})[.!]?\s*\Z",
    re.IGNORECASE,
)
QUESTION_WORDS = frozenset(
    """
    a an and are as at be by did do does for from had has have how in is it
    many of on or that the their them there these this to was were what when
    where which who whose why will with would
    أو أين أن إلى التي الذي على عن في كم كيف ما ماذا متى من هل هو هي و
    """.split()
)


def retrieval_question(question: str) -> str:
    # Keep the complete question for generation. Search only trims a trailing
    # format instruction that does not name any additional evidence to find.
    return PRESENTATION_SUFFIX.sub("", question)


_LIMIT_COMPARISON = re.compile(
    r"^(?:did|does|will|would|can)\s+(.+?)\s+"
    r"(?:meet|satisfy|exceed|stay within|fall within|comply with)\s+(?:the\s+|its\s+)?"
    r"([^?,]+?\b(?:targets?|limits?|requirements?|thresholds?|deadlines?)\b[^?,]*)"
    r"(?:,.*)?[?!.]?$",
    re.IGNORECASE,
)


def comparison_queries(search_text: str) -> tuple[str, ...]:
    """Look up the rule and measured entity independently for limit comparisons."""
    match = _LIMIT_COMPARISON.fullmatch(search_text.strip())
    return (match[2].strip(), match[1].strip()) if match else ()


def comparison_evidence(
    original: list[RetrievedChunk], focused: list[list[RetrievedChunk]], top_k: int,
) -> list[RetrievedChunk]:
    # Reserve one qualified passage for each independently searched part.
    # Preserve the original prompt order if both were already retrieved.
    required: dict[str, RetrievedChunk] = {}
    for hits in focused:
        winner = next((hit for hit in hits if hit.chunk_id not in required), None)
        if winner is not None:
            required[winner.chunk_id] = winner
    chosen = dict(required)
    for hit in original:
        if len(chosen) >= top_k:
            break
        chosen.setdefault(hit.chunk_id, hit)
    ranked = {hit.chunk_id: hit for hit in original if hit.chunk_id in chosen}
    for chunk_id, hit in required.items():
        ranked.setdefault(chunk_id, hit)
    return list(ranked.values())[:top_k]


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass(frozen=True, slots=True)
class LexicalQuery:
    terms: frozenset[str]
    identifiers: frozenset[str]
    phrases: frozenset[tuple[str, str]]

    @classmethod
    def from_text(cls, text: str) -> "LexicalQuery":
        words = _tokens(text)
        terms = frozenset(words) - QUESTION_WORDS
        return cls(
            terms=terms,
            identifiers=frozenset(term for term in terms if any(char.isdecimal() for char in term)),
            phrases=frozenset((a, b) for a, b in pairwise(words) if a in terms and b in terms),
        )

    def document_filter(self) -> dict[str, str]:
        alternatives = "|".join(re.escape(term) for term in sorted(self.terms))
        return {"$regex": rf"(?i)\b(?:{alternatives})\b"}

    def score(self, text: str) -> float:
        words = _tokens(text)
        matched = self.terms.intersection(words)
        if len(matched) < 2 or len(matched) * 2 < len(self.terms):
            return 0.0
        if self.identifiers and not self.identifiers.intersection(matched):
            return 0.0

        # Exact phrases prioritize the associated row over nearby mentions of its terms.
        weight = len(self.terms) + len(self.identifiers) + len(self.phrases)
        evidence = (
            len(matched)
            + len(self.identifiers.intersection(matched))
            + len(self.phrases.intersection(pairwise(words)))
        )
        return evidence / weight

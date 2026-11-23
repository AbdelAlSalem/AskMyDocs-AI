import json
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from math import isclose, isfinite, sqrt
from pathlib import Path

from app.rag.demo_embeddings import (
    EMBEDDING_DIMENSIONS,
    MODEL_ID,
    MODEL_REVISION,
    DemoQueryEncoder,
    passage_fingerprint,
)
from app.rag.document_wide import DocumentCoverage, DocumentIntent, SummaryPageScope, document_coverage
from app.rag.lexical import QUESTION_WORDS, LexicalQuery, comparison_evidence, comparison_queries, retrieval_question
from app.rag.types import RetrievedChunk
from app.services.errors import DocumentError


CORPUS_PATH = Path(__file__).resolve().parents[1] / "demo_data" / "corpus.json"
EMBEDDINGS_PATH = CORPUS_PATH.with_name("embeddings.json")
DEFAULT_TOP_K = 4
LEXICAL_RANK_WEIGHT = 0.8
ENGLISH_QUERY_FORMS = (
    (
        re.compile(r"how\s+big\s+(is|are)\s+([^.?!؟]+?)\s*\??", re.IGNORECASE),
        "How large {} {}?",
    ),
    (
        re.compile(r"where\s+(?:should|can)\s+(?:i|we)\s+go\s+for\s+([^.?!؟]+?)\s*\??", re.IGNORECASE),
        "Where is {}?",
    ),
)
MINIMUM_QUERY_COVERAGE = 0.6
WORD = re.compile(r"[^\W_]+", re.UNICODE)
ARABIC_LETTERS = str.maketrans("ىة", "يه")
TERM_GROUPS = {
    "launch": "launch launched launches launching kickoff start starts begins اطلاق انطلاق ينطلق يطلق تدشين",
    "date": "date dates day days when موعد مواعيد تاريخ تواريخ يوم متى",
    "pilot": "pilot pilots trial experimental تجريبي تجريبيه تجريب",
    "team": "team teams group groups فريق فرق مجموعه",
    "volunteer": "volunteer volunteers volunteering متطوع متطوعون متطوعين متطوعه",
    "workshop": "workshop workshops session sessions training ورشه ورش جلسه جلسات تدريب",
    "location": "location located where venue place held address اين مكان تقام ستقام موقع عنوان مقر",
    "duration": "duration long length last lasts hours hour takes مده تستمر تستغرق يستغرق ساعه ساعتين ساعات زمن",
    "library": "library libraries مكتبه مكتبات",
    "registration": "register registration registering تسجيل يسجل",
    "reading": "reading read قراءه مطالعه",
    "system": "system systems نظام انظمه",
    "summary": "summary summarize summarise overview highlights points recap ملخص لخص نقاط ابرز اهم",
    "orbit": "orbit اوربت اوربيت",
}
EXTRA_QUESTION_WORDS = frozenset(
    """
    about advice according all answer can could details document documents fact facts get
    give important information know list main me much name need number pdf please project
    report s say should size tell they through under us want we you your
    اذكر اعطني اعرف اول بيانات تقرير توضيح ذلك سوف سيتم عام عدد عن في ما مستند
    مستندات ملف مشروع نعرف هذا وهذه هذه هنا يمكن يجب
    """.split()
)


@dataclass(frozen=True, slots=True)
class DemoDocument:
    id: str
    filename: str
    page_count: int
    chunk_count: int
    size_bytes: int
    uploaded_at: datetime
    status: str = "ready"


def _words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold()).translate(ARABIC_LETTERS)
    normalized = "".join(
        str(unicodedata.decimal(char)) if char.isdecimal() else char
        for char in normalized
        if not unicodedata.combining(char) and char != "ـ"
    )
    words = []
    for word in WORD.findall(normalized):
        # Arabic queries and passages can differ in attached articles and conjunctions.
        if len(word) > 5 and word.startswith(("وال", "بال", "فال", "كال")):
            word = word[3:]
        elif len(word) > 4 and word.startswith(("ال", "لل")):
            word = word[2:]
        words.append(word)
    return words


ALIASES = {word: concept for concept, variants in TERM_GROUPS.items() for word in _words(variants)}
STOP_WORDS = set(_words(" ".join(QUESTION_WORDS | EXTRA_QUESTION_WORDS)))


def _terms(text: str) -> frozenset[str]:
    return frozenset(
        ALIASES.get(word, word)
        for word in _words(text)
        if word in ALIASES or word not in STOP_WORDS
    )


def _alternate_english_query(question: str) -> str | None:
    for pattern, template in ENGLISH_QUERY_FORMS:
        if match := pattern.fullmatch(question.strip()):
            return template.format(*match.groups())
    return None


class DemoRetriever:
    """Read-only retrieval over the bundled synthetic PDFs."""

    def __init__(
        self, *,
        max_context_chars: int = 4500,
        use_semantic: bool = True,
        embed_query: Callable[[str], Sequence[float]] | None = None,
    ) -> None:
        corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        self._max_context_chars = max_context_chars
        self._documents = {
            item["id"]: DemoDocument(
                id=item["id"],
                filename=item["filename"],
                page_count=item["page_count"],
                chunk_count=item["chunk_count"],
                size_bytes=item["size_bytes"],
                uploaded_at=datetime.fromisoformat(item["uploaded_at"]),
            )
            for item in corpus["documents"]
        }
        self._passages = tuple(
            RetrievedChunk(
                document_id=item["document_id"],
                filename=self._documents[item["document_id"]].filename,
                page_number=item["page_number"],
                chunk_id=item["chunk_id"],
                text=item["text"],
                similarity=0.0,
            )
            for item in corpus["passages"]
        )
        self._embed_query = None
        if use_semantic:
            self._vectors = _load_vectors(corpus["passages"])
            self._embed_query = DemoQueryEncoder().embed_query if embed_query is None else embed_query
        else:
            # The previous demo retriever is available for the offline comparison.
            self._lexical_terms = {
                passage.chunk_id: _terms(passage.text) for passage in self._passages
            }

    def list_documents(self) -> list[DemoDocument]:
        return list(self._documents.values())

    def retrieve_document_wide(
        self, document_ids: list[str] | None, *, intent: DocumentIntent, max_context_chars: int,
        page_scope: SummaryPageScope | None = None,
    ) -> DocumentCoverage:
        ids = self._documents if document_ids is None else document_ids
        if set(ids).difference(self._documents):
            raise DocumentError(404, "invalid_document_scope", "A selected document is no longer available.")
        scope = {doc_id: self._documents[doc_id].page_count for doc_id in ids}
        passages = (passage for passage in self._passages if passage.document_id in scope)
        return document_coverage(
            passages, scope, intent, min(max_context_chars, self._max_context_chars), page_scope=page_scope,
        )

    def retrieve(
        self,
        question: str,
        document_ids: list[str] | None = None,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = 0.30,
    ) -> list[RetrievedChunk]:
        scope = set(self._documents if document_ids is None else document_ids)
        if scope.difference(self._documents):
            raise DocumentError(404, "invalid_document_scope", "A selected document is no longer available.")
        if not scope or top_k <= 0:
            return []
        passages = [passage for passage in self._passages if passage.document_id in scope]
        if not passages:
            return []
        search_text = retrieval_question(question)
        if self._embed_query is None:
            ranked = self._rank_lexical(search_text, scope, passages, min_similarity)
        else:
            ranked = self._rank_semantic(
                search_text, passages, self._embed_query(search_text), top_k, min_similarity,
            )
            if top_k >= 2 and (parts := comparison_queries(search_text)):
                focused = [
                    self._rank_semantic(part, passages, self._embed_query(part), top_k, min_similarity)
                    for part in parts
                ]
                ranked = comparison_evidence(ranked, focused, top_k)
            # A second search can recover a paraphrase without lowering the cutoff
            # or replacing passages already qualified by the original question.
            if not ranked and (alternate := _alternate_english_query(search_text)):
                ranked = self._rank_semantic(
                    search_text, passages, self._embed_query(alternate), top_k, min_similarity,
                )
        return _spread_pages(ranked, top_k, self._max_context_chars)

    def _rank_semantic(
        self,
        search_text: str,
        passages: list[RetrievedChunk],
        embedding: Sequence[float],
        top_k: int,
        min_similarity: float,
    ) -> list[RetrievedChunk]:
        query = LexicalQuery.from_text(search_text)
        scores = []
        for passage in passages:
            vector = self._vectors[passage.chunk_id]
            similarity = sum(a * b for a, b in zip(embedding, vector, strict=True))
            scores.append(replace(
                passage, similarity=max(-1.0, min(1.0, similarity)), lexical_score=query.score(passage.text),
            ))

        dense = sorted(scores, key=lambda match: match.similarity, reverse=True)[:top_k]
        lexical = sorted(
            (match for match in scores if match.lexical_score > 0),
            key=lambda match: match.lexical_score, reverse=True,
        )[:top_k]
        lexical_ids = {match.chunk_id for match in lexical}
        candidates = {match.chunk_id: match for match in dense + lexical}
        qualified = [
            replace(match, lexical_score=match.lexical_score if match.chunk_id in lexical_ids else 0.0)
            for match in candidates.values()
            if match.similarity >= min_similarity or match.chunk_id in lexical_ids
        ]
        qualified.sort(
            key=lambda match: max(0.0, match.similarity) + LEXICAL_RANK_WEIGHT * match.lexical_score,
            reverse=True,
        )
        return qualified

    def _rank_lexical(
        self, search_text: str, scope: set[str], passages: list[RetrievedChunk], min_similarity: float,
    ) -> list[RetrievedChunk]:
        # A filename identifies a document, not evidence for any of its claims.
        references = set().union(*(_terms(Path(self._documents[doc_id].filename).stem) for doc_id in scope))
        terms = _terms(search_text) - references
        if terms == {"summary"}:
            return [replace(passage, lexical_score=1.0) for passage in passages]
        if not terms:
            return []

        terms_by_passage = [(passage, self._lexical_terms[passage.chunk_id]) for passage in passages]
        matched = terms.intersection(set().union(*(words for _, words in terms_by_passage)))
        if len(matched) / len(terms) < MINIMUM_QUERY_COVERAGE:
            return []

        ranked = []
        for passage, words in terms_by_passage:
            score = len(words.intersection(terms)) / len(terms)
            if score > 0 and score >= min_similarity:
                ranked.append(replace(passage, lexical_score=score))
        ranked.sort(key=lambda item: (-item.lexical_score, item.filename, item.page_number, item.chunk_id))
        return ranked


def _load_vectors(passages: list[dict]) -> dict[str, list[float]]:
    if not EMBEDDINGS_PATH.is_file():
        raise RuntimeError("Demo vectors are missing. Run scripts/build_demo_embeddings.py.")
    index = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
    vectors = index["vectors"]
    if (
        index["model"] != MODEL_ID
        or index["revision"] != MODEL_REVISION
        or index["passage_sha256"] != passage_fingerprint(passages)
        or vectors.keys() != {passage["chunk_id"] for passage in passages}
    ):
        raise RuntimeError("The demo corpus and vectors differ. Run scripts/build_demo_embeddings.py.")
    if any(
        len(vector) != EMBEDDING_DIMENSIONS
        or not all(isfinite(value) for value in vector)
        or not isclose(sqrt(sum(value * value for value in vector)), 1.0, abs_tol=1e-5)
        for vector in vectors.values()
    ):
        raise RuntimeError("Invalid demo vectors. Run scripts/build_demo_embeddings.py.")
    return vectors


def _spread_pages(passages: list[RetrievedChunk], limit: int, context_chars: int) -> list[RetrievedChunk]:
    unique, remaining = [], []
    seen: set[tuple[str, int]] = set()
    for passage in passages:
        page = (passage.document_id, passage.page_number)
        if page in seen:
            remaining.append(passage)
        else:
            unique.append(passage)
            seen.add(page)
    selected = []
    for passage in unique + remaining:
        length = len(passage.text)
        if length <= context_chars:
            selected.append(passage)
            context_chars -= length
            if len(selected) == limit:
                break
    return selected

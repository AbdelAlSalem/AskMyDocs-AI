from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.pdf import PDFPage


@dataclass(frozen=True, slots=True)
class Chunk:
    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    text: str


@dataclass(frozen=True, slots=True)
class ChunkTokenBudget:
    count: Callable[[str], int]
    maximum: int


def split_pages(
    pages: Sequence[PDFPage],
    *,
    document_id: str,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
    token_budget: ChunkTokenBudget | None = None,
) -> list[Chunk]:
    length = len
    if token_budget is not None:
        count_tokens = lru_cache(maxsize=512)(token_budget.count)

        def length(text: str) -> int:
            # Use the larger fraction of the character and encoder limits.
            scaled_tokens = (count_tokens(text) * chunk_size + token_budget.maximum - 1) // token_budget.maximum
            return max(len(text), scaled_tokens)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length,
        keep_separator="end",
        separators=["\n\n", "\n", "؟ ", ". ", "! ", "؛ ", "، ", " ", ""],
    )
    return [
        Chunk(document_id, filename, page.number, f"{document_id}:{page.number}:{index}", text)
        for page in pages
        for index, text in enumerate(_split_page(page.text, splitter, length, chunk_size), start=1)
    ]


def _split_page(
    text: str, splitter: RecursiveCharacterTextSplitter, length: Callable[[str], int], maximum: int,
) -> Iterator[str]:
    for chunk in splitter.split_text(text):
        yield from _verify_length(chunk, length, maximum)


def _verify_length(text: str, length: Callable[[str], int], maximum: int) -> Iterator[str]:
    if length(text) <= maximum:
        yield text
        return
    # Token counts of combined text can differ from the sum of individual pieces.
    # Recheck the final chunk and split at a predictable boundary if it is too large.
    midpoint = len(text) // 2
    if midpoint == 0:
        raise ValueError("A single character exceeds the embedding model's chunk token budget.")
    left, right = text[:midpoint].rstrip(), text[midpoint:].lstrip()
    if left:
        yield from _verify_length(left, length, maximum)
    if right:
        yield from _verify_length(right, length, maximum)

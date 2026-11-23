from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    text: str
    similarity: float
    lexical_score: float = 0.0

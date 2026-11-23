"""Build the public demo corpus from the explicit synthetic PDF allowlist."""

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

BACKEND = Path(__file__).resolve().parents[1]
SAMPLES = BACKEND.parent / "samples"
OUTPUT = BACKEND / "app" / "demo_data" / "corpus.json"
APPROVED_SOURCES = {"project-orbit": "project-orbit.pdf"}
MAX_CHUNK_CHARACTERS = 800

sys.path.insert(0, str(BACKEND))

from app.services.pdf import extract_pages


def split_passage(text: str) -> Iterator[str]:
    remaining = text.strip()
    while len(remaining) > MAX_CHUNK_CHARACTERS:
        boundary = remaining.rfind("\n", 0, MAX_CHUNK_CHARACTERS + 1)
        if boundary <= MAX_CHUNK_CHARACTERS // 2:
            boundary = remaining.rfind(" ", 0, MAX_CHUNK_CHARACTERS + 1)
        if boundary < 1:
            boundary = MAX_CHUNK_CHARACTERS
        yield remaining[:boundary].rstrip()
        remaining = remaining[boundary:].lstrip()
    if remaining:
        yield remaining


def build_corpus() -> str:
    documents = []
    passages = []
    for document_id, filename in APPROVED_SOURCES.items():
        source = (SAMPLES / filename).resolve(strict=True)
        if source.parent != SAMPLES.resolve():
            raise ValueError("Approved demo PDFs must reside directly in samples.")
        raw = source.read_bytes()
        pages = extract_pages(raw)
        document_passages = [
            {
                "document_id": document_id,
                "page_number": page.number,
                "chunk_id": f"{document_id}:{page.number}:{index}",
                "text": text,
            }
            for page in pages
            for index, text in enumerate(split_passage(page.text), start=1)
        ]
        metadata = PdfReader(BytesIO(raw)).metadata
        created_at = metadata.creation_date if metadata is not None else None
        documents.append(
            {
                "id": document_id,
                "filename": filename,
                "page_count": len(pages),
                "chunk_count": len(document_passages),
                "size_bytes": len(raw),
                "uploaded_at": (created_at or datetime.fromtimestamp(0, UTC)).isoformat(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        passages.extend(document_passages)
    return json.dumps({"documents": documents, "passages": passages}, ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the checked-in corpus without writing it.")
    args = parser.parse_args()
    corpus = build_corpus()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != corpus:
            parser.exit(1, "The demo corpus is out of date. Run build_demo_corpus.py.\n")
        print(f"Demo corpus verified: {OUTPUT.relative_to(BACKEND.parent)}")
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(corpus, encoding="utf-8", newline="\n")
    print(f"Demo corpus built: {OUTPUT.relative_to(BACKEND.parent)}")


if __name__ == "__main__":
    main()

"""Compare retrieval on the approved Project Orbit PDF without calling Groq."""

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


BACKEND = Path(__file__).resolve().parents[1]
SOURCE = BACKEND.parent / "samples" / "project-orbit.pdf"
CASES = BACKEND / "tests" / "fixtures" / "demo_retrieval_cases.json"
DEMO_DOCUMENT_ID = "project-orbit"
ENGINES = ("local", "demo", "demo-lexical")
TOP_K = 4
MIN_SIMILARITY = 0.30
CONTEXT_CHARS = 4500

sys.path.insert(0, str(BACKEND))


def _result(cases, retrieve, document_id):
    rows = []
    for case in cases:
        matches = retrieve(
            case["question"], [document_id], top_k=TOP_K, min_similarity=MIN_SIMILARITY,
        )
        evidence = case["evidence"]
        missing = [
            item for item in evidence
            if not any(
                passage.filename == SOURCE.name
                and passage.page_number == item["page"]
                and item["quote"] in " ".join(passage.text.split())
                for passage in matches
            )
        ]
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "expected_pages": sorted({item["page"] for item in evidence}),
            "missing_evidence": missing,
            "passed": not missing if evidence else not matches,
            "matches": [
                {
                    "page": passage.page_number,
                    "filename": passage.filename,
                    "chunk_id": passage.chunk_id,
                    "similarity": round(passage.similarity, 4),
                    "lexical_score": round(passage.lexical_score, 4),
                }
                for passage in matches
            ],
        })
    return rows


def _evaluate_local(cases):
    from app.core.config import Settings
    from app.rag.vector_store import VectorStore
    from app.services.documents import DocumentService
    from app.services.errors import DocumentError

    config = Settings()
    with TemporaryDirectory(prefix="askmydocs-retrieval-eval-", ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        service = DocumentService(
            upload_dir=root / "uploads",
            sqlite_path=root / "catalog.sqlite3",
            vectors=VectorStore(directory=root / "chroma", model_name=config.embedding_model),
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        try:
            with SOURCE.open("rb") as pdf:
                document = service.ingest(SOURCE.name, pdf)
            return _result(cases, service.retrieve, document.id)
        except DocumentError as error:
            if error.code == "embedding_unavailable":
                raise RuntimeError(
                    f"Install the local backend dependencies and cache {config.embedding_model} before running "
                    "the offline comparison, or point HF_HOME to an existing cache."
                ) from error
            raise


def _evaluate_demo(cases, *, lexical: bool):
    from app.rag.demo_retrieval import DemoRetriever

    options = {"use_semantic": False} if lexical else {}
    retriever = DemoRetriever(max_context_chars=CONTEXT_CHARS, **options)
    return _result(cases, retriever.retrieve, DEMO_DOCUMENT_ID)


def _print_report(cases, outcomes) -> None:
    column_width = 22
    case_width = max(len(case["id"]) for case in cases) + 2
    print(f"{'Case':{case_width}}{'Target pages':{column_width}}" + "".join(
        f"{engine:{column_width}}" for engine in outcomes
    ))
    for index, case in enumerate(cases):
        expected = ", ".join(str(page) for page in sorted({item["page"] for item in case["evidence"]}))
        values = []
        for rows in outcomes.values():
            row = rows[index]
            found = ", ".join(str(page) for page in dict.fromkeys(item["page"] for item in row["matches"]))
            values.append(f"{'pass' if row['passed'] else 'miss'} ({found or 'none'})")
        print(f"{case['id']:{case_width}}{(expected or 'none'):{column_width}}" + "".join(
            f"{value:{column_width}}" for value in values
        ))
    print()
    for engine, rows in outcomes.items():
        supported = [row for row in rows if row["expected_pages"]]
        unsupported = [row for row in rows if not row["expected_pages"]]
        print(
            f"{engine}: found complete evidence for {sum(row['passed'] for row in supported)}/{len(supported)} "
            f"supported cases, returned no context for {sum(row['passed'] for row in unsupported)}/"
            f"{len(unsupported)} unsupported cases."
        )
    print("Unsupported cases measure retrieval abstention. A retrieved passage may still lead to a grounded refusal.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("all", *ENGINES), default="all")
    parser.add_argument("--json", action="store_true", help="Print the cases and ranked scores as JSON.")
    args = parser.parse_args()
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    engines = ENGINES if args.engine == "all" else (args.engine,)
    os.environ["HF_HUB_OFFLINE"] = "1"
    outcomes = {}
    for engine in engines:
        try:
            outcomes[engine] = (
                _evaluate_local(cases) if engine == "local"
                else _evaluate_demo(cases, lexical=engine == "demo-lexical")
            )
        except (ImportError, OSError, RuntimeError) as error:
            parser.exit(2, f"{engine} evaluation unavailable: {error}\n")
    if args.json:
        print(json.dumps(outcomes, indent=2))
    else:
        _print_report(cases, outcomes)


if __name__ == "__main__":
    main()

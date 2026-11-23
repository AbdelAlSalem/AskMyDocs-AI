"""Build the public corpus vectors offline with the full-precision local model."""

import argparse
import json
import sys
from math import isclose
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_OFFLINE


BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.rag.demo_embeddings import MODEL_ID, MODEL_REVISION, passage_fingerprint
from app.rag.demo_retrieval import CORPUS_PATH, EMBEDDINGS_PATH
from app.rag.embeddings import LocalEmbeddings


def build_embeddings() -> dict:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    passages = corpus["passages"]
    snapshot = snapshot_download(
        MODEL_ID, revision=MODEL_REVISION,
        allow_patterns=["*.json", "model.safetensors"],
        local_files_only=HF_HUB_OFFLINE, token=False,
    )
    encoder = LocalEmbeddings(snapshot)
    budget = encoder.chunk_token_budget()
    if any(budget.count(passage["text"]) > budget.maximum for passage in passages):
        raise ValueError("A demo passage exceeds the encoder token limit. Split it before embedding.")
    vectors = encoder.embed_documents([passage["text"] for passage in passages])
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "passage_sha256": passage_fingerprint(passages),
        "vectors": {
            passage["chunk_id"]: [round(value, 8) for value in vector]
            for passage, vector in zip(passages, vectors, strict=True)
        },
    }


def _matches(existing: dict, expected: dict) -> bool:
    if existing.keys() != expected.keys():
        return False
    if any(existing[key] != expected[key] for key in expected if key != "vectors"):
        return False
    stored = existing["vectors"]
    generated = expected["vectors"]
    return stored.keys() == generated.keys() and all(
        len(stored[chunk_id]) == len(vector)
        and all(isclose(a, b, abs_tol=1e-6) for a, b in zip(stored[chunk_id], vector, strict=True))
        for chunk_id, vector in generated.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the committed vectors without writing.")
    args = parser.parse_args()
    if args.check and not EMBEDDINGS_PATH.is_file():
        parser.exit(1, "Demo vectors are missing. Run build_demo_embeddings.py.\n")
    vectors = build_embeddings()
    if args.check:
        existing = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
        if not _matches(existing, vectors):
            parser.exit(1, "Demo vectors are out of date. Run build_demo_embeddings.py.\n")
    else:
        EMBEDDINGS_PATH.write_text(
            json.dumps(vectors, indent=2) + "\n", encoding="utf-8", newline="\n",
        )
    print(f"Demo vectors {'verified' if args.check else 'built'}: {EMBEDDINGS_PATH.relative_to(BACKEND.parent)}")


if __name__ == "__main__":
    main()

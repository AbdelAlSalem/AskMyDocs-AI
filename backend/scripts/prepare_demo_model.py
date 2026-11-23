"""Fetch and verify pinned public ONNX assets for the demo backend."""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.rag.demo_embeddings import MODEL_DIR, MODEL_FILES, MODEL_ID, MODEL_REVISION


def _verified(path: Path, expected_sha256: str) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == expected_sha256


def _download(target: Path, asset: str, sha256: str) -> None:
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{asset}?download=true"
    with NamedTemporaryFile(dir=target.parent, prefix=target.name, suffix=".part", delete=False) as output:
        temporary_path = Path(output.name)
    try:
        with urlopen(url, timeout=120) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if not _verified(temporary_path, sha256):
            raise ValueError(f"SHA-256 verification failed for {target.name}.")
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify local assets without downloading.")
    args = parser.parse_args()
    for filename, (asset, sha256) in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not _verified(path, sha256):
            if args.check:
                parser.exit(1, f"Missing or invalid {filename}. Run prepare_demo_model.py.\n")
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Downloading pinned demo model asset {filename}...", flush=True)
            try:
                _download(path, asset, sha256)
            except (OSError, ValueError):
                parser.exit(1, f"Could not download or verify {filename}. Retry prepare_demo_model.py.\n")
        print(f"Verified {filename}.", flush=True)


if __name__ == "__main__":
    main()

"""Pinned lightweight query encoder for the public demo."""

import hashlib
import json
import re
from pathlib import Path
from threading import Lock


MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MODEL_DIR = Path(__file__).resolve().parents[1] / "demo_model"
EMBEDDING_DIMENSIONS = 384
MAX_SEQUENCE_LENGTH = 128
MODEL_FILES = {
    "encoder.onnx": (
        "onnx/model_quint8_avx2.onnx",
        "98a01d88b7de996cdea58c32ca71208c09968d143798814b2ea09d3439dc334f",
    ),
    "sentencepiece.bpe.model": (
        "sentencepiece.bpe.model",
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    ),
}
SPECIAL_TOKENS = re.compile(r"(<s>|</s>|<pad>|<unk>|<mask>)")


def passage_fingerprint(passages: list[dict]) -> str:
    serialized = json.dumps(passages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class DemoQueryEncoder:
    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        if any(not (model_dir / name).is_file() for name in MODEL_FILES):
            raise RuntimeError("Demo encoder assets are missing. Run scripts/prepare_demo_model.py.")

        import numpy
        import onnxruntime
        from sentencepiece import SentencePieceProcessor

        self._numpy = numpy
        self._tokenizer = SentencePieceProcessor(model_file=str(model_dir / "sentencepiece.bpe.model"))
        self._special_ids = {
            "<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": self._tokenizer.vocab_size() + 1,
        }
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        self._session = onnxruntime.InferenceSession(
            str(model_dir / "encoder.onnx"), sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._lock = Lock()

    def embed_query(self, text: str) -> list[float]:
        # Serialize inference to bound peak memory under concurrent chat requests.
        with self._lock:
            return self._embed_query(text)

    def _token_ids(self, text: str) -> list[int]:
        # XLM-R uses fairseq IDs in the exported model. SentencePiece reserves 0
        # for unknown, while regular piece IDs are shifted by one in fairseq.
        ids = [self._special_ids["<s>"]]
        unknown = self._tokenizer.unk_id()
        normalized = " ".join(text.split())
        for part in SPECIAL_TOKENS.split(normalized):
            if part in self._special_ids:
                ids.append(self._special_ids[part])
            elif part:
                ids.extend(
                    self._special_ids["<unk>"] if piece == unknown else piece + 1
                    for piece in self._tokenizer.encode_as_ids(part)
                )
        return ids[:MAX_SEQUENCE_LENGTH - 1] + [self._special_ids["</s>"]]

    def _embed_query(self, text: str) -> list[float]:
        np = self._numpy
        input_ids = np.asarray([self._token_ids(text)], dtype=np.int64)
        mask = np.ones_like(input_ids)
        (hidden,) = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": mask,
            "token_type_ids": np.zeros_like(input_ids),
        })
        # In a single unpadded sentence all attention weights are one.
        pooled = hidden[0].mean(axis=0)
        norm = np.linalg.norm(pooled)
        if pooled.shape != (EMBEDDING_DIMENSIONS,) or not np.isfinite(norm) or norm <= 0:
            raise RuntimeError("The demo encoder returned an invalid embedding.")
        return (pooled / norm).tolist()

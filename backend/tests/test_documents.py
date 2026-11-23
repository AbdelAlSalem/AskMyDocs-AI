import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import chromadb
import pytest
from chromadb.config import Settings
from langchain_core.embeddings import Embeddings
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.rag.chunking import ChunkTokenBudget, split_pages
from app.rag.embeddings import LocalEmbeddings
from app.rag.vector_store import COLLECTION_NAME, VectorStore, VectorStoreError
from app.services.documents import DocumentError, DocumentService
from app.services.pdf import PDFPage, _clean_text, extract_pages, read_pdf_upload


def make_pdf(*page_texts: str | None) -> bytes:
    writer = PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            continue
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
        })
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
        page.replace_contents(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_extract_preserves_original_page_numbers_and_cleans_text():
    pages = extract_pages(make_pdf("The  station opened in  2021.", None, "Annual review."))

    assert pages == [
        PDFPage(1, "The station opened in 2021."),
        PDFPage(2, ""),
        PDFPage(3, "Annual review."),
    ]


def test_extract_sample_preserves_english_and_arabic_reading_order():
    sample = Path(__file__).resolve().parents[2] / "samples" / "project-orbit.pdf"

    launch, workshop = extract_pages(sample.read_bytes())

    assert launch.number == 1
    assert "18 June 2027" in launch.text
    assert workshop.number == 2
    assert "مكتبة النور" in workshop.text
    assert "تستمر الورشة ساعتين." in workshop.text
    assert "PROJECT ORBIT" in workshop.text


def test_normalizes_arabic_presentation_forms_only():
    assert _clean_text("\ufefb \ufb50 \u06cc \ufb01") == "لا ٱ ی ﬁ"


def make_positioned_pdf(content: str) -> bytes:
    writer = PdfWriter(clone_from=PdfReader(BytesIO(make_pdf("Text"))))
    stream = DecodedStreamObject()
    stream.set_data(content.encode("ascii"))
    writer.pages[0].replace_contents(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize("label_offset", [0, 36])
def test_metric_card_labels_are_joined_only_with_matching_coordinates(label_offset):
    content = make_positioned_pdf(
        "BT /F1 18 Tf 1 0 0 1 72 740 Tm (23 seats) Tj "
        f"/F1 12 Tf 1 0 0 1 {72 + label_offset} 720 Tm (South room) Tj "
        "/F1 18 Tf 1 0 0 1 300 740 Tm (46 seats) Tj "
        "/F1 12 Tf 1 0 0 1 300 720 Tm (North room) Tj ET"
    )
    page = extract_pages(content)[0]

    if label_offset:
        assert "23 seats\nSouth room" in page.text
        assert "46 seats\nNorth room" in page.text
    else:
        assert "South room: 23 seats" in page.text
        assert "North room: 46 seats" in page.text


def test_aligned_table_rows_keep_column_names_after_chunking():
    content = make_positioned_pdf(
        "BT /F1 12 Tf 1 0 0 1 72 740 Tm (Room) Tj 1 0 0 1 300 740 Tm (Capacity) Tj "
        "1 0 0 1 72 720 Tm (South room) Tj 1 0 0 1 300 720 Tm (23 seats) Tj "
        "1 0 0 1 72 700 Tm (North room) Tj 1 0 0 1 300 700 Tm (46 seats) Tj ET"
    )
    pages = extract_pages(content)
    chunks = split_pages(
        pages, document_id="sample", filename="workshop.pdf", chunk_size=42, chunk_overlap=0,
    )

    assert chunks[-2].text == "Room: South room | Capacity: 23 seats"
    assert chunks[-1].text == "Room: North room | Capacity: 46 seats"
    assert all(chunk.page_number == 1 for chunk in chunks)


@pytest.mark.parametrize(
    ("filename", "contents", "maximum", "code", "status_code"),
    [
        ("report.txt", b"%PDF-1.7", 100, "invalid_file_type", 415),
        ("report.pdf", b"plain text", 100, "invalid_pdf_signature", 415),
        ("report.PDF", b"%PDF-1.7", 5, "file_too_large", 413),
        ("bad\x00name.pdf", b"%PDF-1.7", 100, "invalid_filename", 400),
    ],
)
def test_reject_invalid_uploads(filename, contents, maximum, code, status_code):
    with pytest.raises(DocumentError) as exc:
        read_pdf_upload(filename, BytesIO(contents), maximum)

    assert exc.value.code == code
    assert exc.value.status_code == status_code


def test_reject_unreadable_and_textless_pdfs():
    with pytest.raises(DocumentError) as malformed:
        extract_pages(b"%PDF-1.7\nmalformed PDF")
    assert malformed.value.code == "invalid_pdf"

    with pytest.raises(DocumentError, match="OCR") as textless:
        extract_pages(make_pdf(None, None))
    assert textless.value.code == "no_extractable_text"


def test_reject_image_only_pdf_with_ocr_hint():
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    image = DecodedStreamObject()
    image.set_data(b"q 40 0 0 40 72 720 cm BI /W 1 /H 1 /CS /RGB /BPC 8 ID \x00\x00\x00 EI Q")
    page.replace_contents(image)
    output = BytesIO()
    writer.write(output)

    with pytest.raises(DocumentError, match="OCR") as scan:
        extract_pages(output.getvalue())

    assert scan.value.code == "no_extractable_text"


def test_split_uses_overlap_and_preserves_arabic_and_per_page_metadata():
    arabic = "القاهرة مدينة"
    chunks = split_pages(
        [PDFPage(1, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"), PDFPage(2, ""), PDFPage(3, arabic)],
        document_id="sample",
        filename="تقرير.pdf",
        chunk_size=16,
        chunk_overlap=3,
    )

    assert [chunk.page_number for chunk in chunks] == [1, 1, 3]
    assert chunks[0].text[-3:] == chunks[1].text[:3]
    assert chunks[-1].text == arabic
    assert chunks[-1].filename == "تقرير.pdf"
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.document_id == "sample" and len(chunk.text) <= 16 for chunk in chunks)


def test_split_keeps_short_rows_intact_under_the_encoder_and_character_limits():
    rows = [
        "Storage capacity: 18 crates",
        "Reserved capacity: 12 crates",
        "East entrance: 4.5 meters wide",
    ]
    token_budget = ChunkTokenBudget(count=lambda text: 3 * len(text.split()), maximum=15)
    pages = [PDFPage(1, "\n".join(rows)), PDFPage(2, " ".join(["long"] * 20))]

    chunks = split_pages(
        pages, document_id="sample", filename="workshop.pdf",
        chunk_size=55, chunk_overlap=6, token_budget=token_budget,
    )

    assert [chunk.text for chunk in chunks if chunk.page_number == 1] == rows
    assert sum(chunk.text.count("long") for chunk in chunks if chunk.page_number == 2) == 20
    assert all(len(chunk.text) <= 55 and token_budget.count(chunk.text) <= 15 for chunk in chunks)


def test_token_budget_and_both_embedding_paths_reuse_one_model(monkeypatch):
    loaded_models = []
    encoded = []
    tokenizer = SimpleNamespace(
        num_special_tokens_to_add=lambda *, pair: 2,
        encode=lambda text, **_kwargs: text.split(),
    )

    def encode(texts, *, normalize_embeddings, show_progress_bar):
        assert normalize_embeddings and not show_progress_bar
        encoded.extend(texts)
        return SimpleNamespace(tolist=lambda: [[1.0, 0.0] for _ in texts])

    def make_model(name, *, device, trust_remote_code):
        loaded_models.append((name, device, trust_remote_code))
        return SimpleNamespace(max_seq_length=12, tokenizer=tokenizer, encode=encode)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=make_model))
    embedder = LocalEmbeddings("test-model")

    budget = embedder.chunk_token_budget()
    assert budget.maximum == 10
    assert budget.count("one\ntwo three") == 3
    assert embedder.embed_documents(["first\nchunk", "second chunk"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert embedder.embed_query("test\nquestion") == [1.0, 0.0]
    assert loaded_models == [("test-model", "cpu", False)]
    assert encoded == ["first chunk", "second chunk", "test question"]


class TopicEmbeddings(Embeddings):
    """Vectors with known similarities for offline tests of the real Chroma index."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if "saffron" in text.casefold() or "زعفران" in text:
            return [1.0, 0.0, 0.0]
        if "observatory" in text.casefold():
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def make_store(path: Path, model_name: str = "test-embeddings") -> VectorStore:
    return VectorStore(directory=path / "chroma", model_name=model_name, embeddings=TopicEmbeddings())


def make_service(path: Path, vectors: VectorStore | None = None) -> DocumentService:
    return DocumentService(
        upload_dir=path / "uploads",
        sqlite_path=path / "documents.sqlite3",
        vectors=vectors if vectors is not None else make_store(path),
    )


def collection_metadata(path: Path, document_id: str):
    client = chromadb.PersistentClient(
        path=str(path / "chroma"), settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    return collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])


def test_ingestion_persists_pdf_record_and_index_and_rejects_renamed_duplicates(tmp_path):
    content = make_pdf("The saffron stock is 29 units.", "The observatory was opened in 2021.")
    service = make_service(tmp_path)
    record = service.ingest("../../تقرير.PDF", BytesIO(content))

    assert str(UUID(record.id)) == record.id
    assert record.filename == "تقرير.PDF"
    assert record.page_count == record.chunk_count == 2
    assert record.status == "ready"
    assert record.size_bytes == len(content)
    assert (tmp_path / "uploads" / f"{record.id}.pdf").read_bytes() == content
    assert make_service(tmp_path).list_documents() == [record]

    stored = collection_metadata(tmp_path, record.id)
    assert len(stored["ids"]) == 2
    for chunk_id, metadata, text in zip(stored["ids"], stored["metadatas"], stored["documents"], strict=True):
        assert metadata == {
            "document_id": record.id,
            "filename": "تقرير.PDF",
            "page_number": int(chunk_id.split(":")[1]),
            "chunk_id": chunk_id,
        }
        assert text in {"The saffron stock is 29 units.", "The observatory was opened in 2021."}

    with pytest.raises(DocumentError) as duplicate:
        service.ingest("new-name.pdf", BytesIO(content))
    assert duplicate.value.status_code == 409


def test_retrieval_ranks_by_similarity_filters_before_top_k_and_skips_unrelated_chunks(tmp_path):
    service = make_service(tmp_path)
    first = service.ingest("first.pdf", BytesIO(make_pdf("The saffron stock is 29 units.")))
    second = service.ingest("second.pdf", BytesIO(make_pdf(None, "The saffron price is 8 coins.")))
    unrelated = service.ingest("other.pdf", BytesIO(make_pdf("The observatory opened in 2021.")))

    question = "كم يبلغ سعر الزعفران؟"
    all_matches = service.retrieve(question, None, top_k=4, min_similarity=0.3)
    assert {match.document_id for match in all_matches} == {first.id, second.id}
    assert all(match.similarity == pytest.approx(1.0) for match in all_matches)

    selected = service.retrieve(question, [second.id], top_k=1, min_similarity=0.3)
    assert [(match.filename, match.page_number, match.text) for match in selected] == [
        ("second.pdf", 2, "The saffron price is 8 coins.")
    ]
    assert service.retrieve(question, [], top_k=4, min_similarity=0.3) == []
    assert service.retrieve(question, [unrelated.id], top_k=4, min_similarity=0.3) == []

    with pytest.raises(DocumentError, match="no longer available"):
        service.retrieve(question, ["missing"], top_k=1, min_similarity=0.3)


def test_delete_removes_file_catalog_and_vectors_and_allows_reupload(tmp_path):
    service = make_service(tmp_path)
    content = make_pdf("The saffron stock is 29 units.")
    document = service.ingest("first.pdf", BytesIO(content))

    service.delete(document.id)

    assert service.list_documents() == []
    assert collection_metadata(tmp_path, document.id)["ids"] == []
    assert not (tmp_path / "uploads" / f"{document.id}.pdf").exists()
    assert service.retrieve("saffron", None, top_k=4, min_similarity=0.3) == []
    assert service.ingest("first.pdf", BytesIO(content)).id != document.id


def test_upload_failure_rolls_back_partially_written_vectors_and_file(tmp_path, monkeypatch):
    vectors = make_store(tmp_path)
    service = make_service(tmp_path, vectors)
    original_add = vectors.add
    uploaded_ids = []

    def fail_after_write(chunks):
        uploaded_ids.append(chunks[0].document_id)
        original_add(chunks[:1])
        raise VectorStoreError

    monkeypatch.setattr(vectors, "add", fail_after_write)
    with pytest.raises(DocumentError) as failed:
        service.ingest("first.pdf", BytesIO(make_pdf("The observatory opened in 2021.")))

    assert failed.value.code == "storage_unavailable"
    assert service.list_documents() == []
    assert list((tmp_path / "uploads").iterdir()) == []
    assert collection_metadata(tmp_path, uploaded_ids[0])["ids"] == []


def test_interrupted_delete_is_hidden_from_retrieval_and_cleaned_on_restart(tmp_path, monkeypatch):
    vectors = make_store(tmp_path)
    service = make_service(tmp_path, vectors)
    document = service.ingest("first.pdf", BytesIO(make_pdf("The saffron stock is 29 units.")))

    def unavailable(document_id):
        raise VectorStoreError

    monkeypatch.setattr(vectors, "delete", unavailable)
    with pytest.raises(DocumentError) as failed:
        service.delete(document.id)

    assert failed.value.status_code == 503
    assert service.list_documents()[0].status == "deleting"
    assert service.retrieve("saffron", None, top_k=4, min_similarity=0.3) == []
    assert make_service(tmp_path).list_documents() == []
    assert not (tmp_path / "uploads" / f"{document.id}.pdf").exists()
    assert collection_metadata(tmp_path, document.id)["ids"] == []


def test_prevents_using_a_different_embedding_model_with_an_existing_index(tmp_path):
    original = make_service(tmp_path)
    document = original.ingest("first.pdf", BytesIO(make_pdf("The saffron stock is 29 units.")))
    changed = make_service(tmp_path, make_store(tmp_path, "different-embeddings"))

    with pytest.raises(DocumentError) as failed:
        changed.retrieve("saffron", None, top_k=4, min_similarity=0.3)
    assert failed.value.code == "embedding_model_mismatch"

    changed.delete(document.id)
    assert collection_metadata(tmp_path, document.id)["ids"] == []

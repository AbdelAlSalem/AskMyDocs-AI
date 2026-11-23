import re
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from itertools import pairwise
from math import hypot
from typing import BinaryIO
from unicodedata import normalize

from pypdf import PageObject, PdfReader

from app.services.errors import DocumentError

ARABIC_PRESENTATION_FORMS = re.compile(r"[\ufb50-\ufdff\ufe70-\ufefc]+")
METRIC_VALUE = re.compile(r"[+−-]?\d[\d.,:/%-]*(?: [^\W\d_]+){0,4}")
METRIC_LABEL = re.compile(r"[^\W\d_][\w -]{1,63}")


@dataclass(frozen=True, slots=True)
class PDFPage:
    number: int
    text: str


@dataclass(frozen=True, slots=True)
class _TextLine:
    text: str
    x: float
    y: float
    font_size: float


def read_pdf_upload(filename: str, file: BinaryIO, max_bytes: int) -> tuple[str, bytes]:
    # The basename is display metadata only. The storage layer generates the actual path.
    filename = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not filename.casefold().endswith(".pdf") or len(filename) <= 4:
        raise DocumentError(415, "invalid_file_type", "Choose a file with a .pdf extension.")
    if len(filename) > 255 or re.search(r"[\x00-\x1f\x7f]", filename):
        raise DocumentError(400, "invalid_filename", "The PDF filename is invalid.")

    contents = file.read(max_bytes + 1)
    if len(contents) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise DocumentError(413, "file_too_large", f"Each PDF must be {limit_mb:g} MB or smaller.")
    if not contents.startswith(b"%PDF-"):
        raise DocumentError(415, "invalid_pdf_signature", "The file does not have a valid PDF header.")
    return filename, contents


def extract_pages(contents: bytes) -> list[PDFPage]:
    try:
        reader = PdfReader(BytesIO(contents))
        if reader.is_encrypted:
            raise DocumentError(422, "encrypted_pdf", "Upload an unlocked copy of the PDF.")
        pages = [
            PDFPage(number, _extract_page_text(page))
            for number, page in enumerate(reader.pages, start=1)
        ]
    except DocumentError:
        raise
    except Exception as exc:
        # Malformed PDF structures can fail at several levels of the parser.
        raise DocumentError(422, "invalid_pdf", "The PDF could not be read. Try another copy.") from exc

    if not any(page.text for page in pages):
        raise DocumentError(
            422,
            "no_extractable_text",
            "This PDF has no extractable text. Scanned and image-only PDFs need OCR before upload.",
        )
    return pages


def _extract_page_text(page: PageObject) -> str:
    lines: list[_TextLine] = []

    def visit(text: str, user_matrix: list[float], text_matrix: list[float], _font, font_size: float) -> None:
        value = _clean_text(text)
        if not value:
            return
        a, b, c, d, e, f = user_matrix
        _, _, horizontal_scale, vertical_scale, x, y = text_matrix
        size = font_size * hypot(horizontal_scale * a + vertical_scale * c, horizontal_scale * b + vertical_scale * d)
        lines.append(_TextLine(value, x * a + y * c + e, x * b + y * d + f, size))

    text = _clean_text(page.extract_text(visitor_text=visit) or "")
    if not text:
        return text
    return _label_table_rows(page, _join_metric_cards(text, lines))


def _join_metric_cards(text: str, lines: list[_TextLine]) -> str:
    # Multi-column metric cards can appear as alternating values and labels in
    # plain text. Use their positions so a neighbouring card's value stays separate.
    rows: dict[tuple[int, int], list[tuple[_TextLine, _TextLine]]] = defaultdict(list)
    for value, label in pairwise(lines):
        if (
            METRIC_VALUE.fullmatch(value.text)
            and METRIC_LABEL.fullmatch(label.text)
            and value.font_size > label.font_size
            and abs(value.x - label.x) < 2
            and 0 < value.y - label.y <= 3 * value.font_size
        ):
            rows[(round(value.y / 2), round(label.y / 2))].append((value, label))

    pairs = {
        (value.text, label.text)
        for row in rows.values()
        if max(value.x for value, _ in row) - min(value.x for value, _ in row) >= 12
        for value, label in row
    }
    if not pairs:
        return text

    remaining = iter(text.splitlines())
    previous = next(remaining)
    output = []
    for current in remaining:
        if (previous, current) in pairs:
            previous = f"{current}: {previous}"
            continue
        output.append(previous)
        previous = current
    output.append(previous)
    return "\n".join(output)


def _label_table_rows(page: PageObject, text: str) -> str:
    plain_lines = text.splitlines()
    if sum(any(char.isdecimal() for char in line) for line in plain_lines) < 2:
        return text
    try:
        layout = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False) or ""
    except Exception:
        return text

    # Layout can reverse Arabic or combine reading columns. Use its column
    # boundaries only for complete, consecutive rows verified against plain text.
    cells_by_line = [re.split(r"\s{2,}", line.strip()) for line in layout.splitlines()]
    plain_positions: dict[str, list[int]] = defaultdict(list)
    for index, line in enumerate(plain_lines):
        plain_positions[line].append(index)

    for index, headers in enumerate(cells_by_line[:-2]):
        if len(headers) < 2 or len(set(headers)) != len(headers):
            continue
        if not all(cell.replace(" ", "").replace("-", "").isalpha() for cell in headers):
            continue
        for start in plain_positions.get(_clean_text(" ".join(headers)), ()):
            rows = []
            for cells, plain in zip(cells_by_line[index + 1:], plain_lines[start + 1:]):
                if len(cells) != len(headers) or _clean_text(" ".join(cells)) != plain:
                    break
                rows.append(cells)
            if sum(any(char.isdecimal() for cell in row for char in cell) for row in rows) < 2:
                continue
            for offset, cells in enumerate(rows, start=1):
                plain_lines[start + offset] = " | ".join(
                    f"{header}: {value}" for header, value in zip(headers, cells, strict=True)
                )
    return "\n".join(plain_lines)


def _clean_text(text: str) -> str:
    # Some PDF fonts expose shaped Arabic glyphs as presentation forms. Fold them
    # to their underlying characters so embedded text matches ordinary questions.
    text = ARABIC_PRESENTATION_FORMS.sub(lambda match: normalize("NFKC", match[0]), text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    # Keep paragraph breaks for chunking while leaving Unicode character order intact.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

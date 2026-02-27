from __future__ import annotations

import csv
import hashlib
import io
import uuid
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from app.models import Chunk


def parse_content(path: str, mime_type: str) -> tuple[str, dict]:
    p = Path(path)
    ext = p.suffix.lower()

    if ext in {".txt", ".md", ".csv"} or mime_type.startswith("text/"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        details = {"parser": "text", "pages": 1}
        if ext == ".csv":
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                rows = list(csv.reader(f))
            details["rows"] = max(len(rows) - 1, 0)
        return text, details

    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(path)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages), {"parser": "pypdf", "pages": len(pages)}
        except Exception:
            return p.read_text(encoding="utf-8", errors="ignore"), {"parser": "fallback", "pages": 1}

    if ext == ".docx":
        try:
            from docx import Document  # type: ignore

            doc = Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs), {"parser": "python-docx", "paragraphs": len(paragraphs)}
        except Exception:
            return "", {"parser": "docx-unavailable", "paragraphs": 0}

    if ext in {".xlsx", ".xls"}:
        try:
            import openpyxl  # type: ignore

            wb = openpyxl.load_workbook(path)
            texts = []
            total_rows = 0
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    vals = [str(c) for c in row if c is not None]
                    if vals:
                        texts.append(" | ".join(vals))
                        total_rows += 1
            return "\n".join(texts), {"parser": "openpyxl", "rows": total_rows}
        except Exception:
            return "", {"parser": "xlsx-unavailable", "rows": 0}

    if ext in {".png", ".jpg", ".jpeg", ".heic"}:
        return "[Image content placeholder: OCR/VQA pipeline required in production]", {"parser": "image-placeholder", "ocr": False}

    return p.read_text(encoding="utf-8", errors="ignore"), {"parser": "fallback", "pages": 1}


def semantic_units(text: str) -> Iterable[tuple[str, int, int, int]]:
    pos = 0
    para_idx = 0
    for raw in text.split("\n\n"):
        chunk = raw.strip()
        if not chunk:
            pos += len(raw) + 2
            continue
        start = text.find(raw, pos)
        end = start + len(raw)
        pos = end + 2
        yield chunk, start, end, para_idx
        para_idx += 1


def embed(text: str, dim: int = 32) -> list[float]:
    vals = [0.0] * dim
    tokens = text.lower().split()
    if not tokens:
        return vals
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        vals[h % dim] += 1.0
    scale = float(len(tokens))
    return [v / scale for v in vals]


def chunk_document(file_id: str, filename: str, upload_ts: datetime, text: str, page: int = 1) -> list[Chunk]:
    chunks: list[Chunk] = []
    for ctext, start, end, pidx in semantic_units(text):
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                file_id=file_id,
                text=ctext,
                start_offset=start,
                end_offset=end,
                page=page,
                paragraph_index=pidx,
                embedding=embed(ctext),
                metadata={
                    "filename": filename,
                    "page": page,
                    "paragraphIndex": pidx,
                    "charOffsets": [start, end],
                    "uploadTimestamp": upload_ts.isoformat() + "Z",
                },
            )
        )
    return chunks

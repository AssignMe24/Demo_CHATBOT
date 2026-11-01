from __future__ import annotations

import csv
import html
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.parse import unquote

import fitz

LOADER_VERSION = "loader-v7-clean-link-units"

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".markdown",
    ".csv", ".xlsx", ".xls", ".html", ".htm", ".pptx",
}

URL_RE = re.compile(r"https?://[^\s\]\)<>\"']+")
_QUESTION_START_RE = re.compile(
    r"^(?:(?:I|We|My|Our|An employee|The employee|Employees|Staff|A staff member)\b[^?]{0,140}?[.!]\s+)?"
    r"(?:How|What|Where|When|Whom|Who|Why|Which|Can|Could|Should|Would|Will|Is|Are|Do|Does|Did)\b",
    re.I,
)
_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[\s.\):-]+\S|(?:chapter|section|part|clause|article)\s+\d+|[A-Z][A-Z0-9 &/()'\-]{3,})$",
    re.I,
)


def clean_text(text: str, preserve_newlines: bool = False) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"-\s*\n\s*", "", text)
    if preserve_newlines:
        lines = [re.sub(r"[\t\r ]+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()
    return re.sub(r"\s+", " ", text).strip()


def _unit(page: int, text: str, label: str, source_type: str, filename: str, **metadata) -> Dict | None:
    text = clean_text(text, preserve_newlines=True)
    if not text:
        return None
    value = {
        "page": int(page), "text": text, "label": label,
        "source_type": source_type, "filename": filename,
    }
    value.update({k: v for k, v in metadata.items() if v is not None})
    return value


def _read_text_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            pass
    return path.read_text(errors="ignore")


def _nice_link_label(raw: str, url: str) -> str:
    raw = clean_text(raw)
    if raw:
        return raw[:100]
    cleaned = unquote(url).replace("https://", "").replace("http://", "")
    return cleaned.split("/")[0] or "Link"


def _extract_pdf_links(page) -> List[str]:
    links, seen = [], set()
    for link in page.get_links():
        url = (link.get("uri") or "").strip()
        if not url or not URL_RE.match(url):
            continue
        label = ""
        rect = link.get("from")
        if rect is not None:
            try:
                label = page.get_textbox(rect)
            except Exception:
                pass
        md = f"[{_nice_link_label(label, url)}]({url})"
        if md not in seen:
            seen.add(md)
            links.append(md)
    return links


def _is_repeating_pdf_header_or_footer(text: str) -> bool:
    compact = clean_text(text).casefold()
    if not compact:
        return True
    patterns = (
        r"^title:.*\bpage[: ]*\d+$",
        r"^hrd\d+\s+issue\s+[0-9.]+$",
        r"^\(c\).*company confidential$",
        r"^page[: ]*\d+$",
        r"^issue\s+[0-9.]+$",
    )
    return any(re.match(pattern, compact) for pattern in patterns)


def _process_pdf_page(args) -> List[Dict]:
    page_index, page_bytes, filename = args
    doc = fitz.open(stream=page_bytes, filetype="pdf")
    page = doc[0]
    page_text = clean_text(page.get_text("text"), preserve_newlines=True)
    page_low = page_text.casefold()
    page_role = "toc" if "table of contents" in page_low else (
        "front_matter" if page_index < 3 and any(
            term in page_low for term in (
                "document control", "version history", "distribution list",
                "authorisation", "authorization",
            )
        ) else None
    )

    units: List[Dict] = []
    blocks = sorted(page.get_text("blocks"), key=lambda b: (round(float(b[1]), 1), round(float(b[0]), 1)))
    for block_no, block in enumerate(blocks, start=1):
        text = clean_text(str(block[4] or ""), preserve_newlines=True)
        if not text or _is_repeating_pdf_header_or_footer(text):
            continue
        item = _unit(
            page_index + 1, text, f"Page {page_index + 1}", "pdf", filename,
            block_index=block_no, block_type="pdf_block", page_role=page_role,
        )
        if item:
            units.append(item)

    links = _extract_pdf_links(page)
    if links:
        item = _unit(
            page_index + 1, "Document links:\n" + "\n".join(links),
            f"Page {page_index + 1}", "pdf", filename,
            block_index=len(units) + 1, block_type="links", page_role=page_role,
        )
        if item:
            units.append(item)
    doc.close()
    return units


def load_pdf_text(pdf_path: str, filename: str | None = None, max_workers: int = 4) -> List[Dict]:
    path = Path(pdf_path)
    display_name = filename or path.name
    with fitz.open(pdf_path) as doc:
        work = []
        for i in range(len(doc)):
            single = fitz.open()
            single.insert_pdf(doc, from_page=i, to_page=i)
            work.append((i, single.tobytes(), display_name))
            single.close()
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, 4))) as executor:
        page_units = list(executor.map(_process_pdf_page, work))
    units = [u for group in page_units for u in group]
    if not units:
        raise ValueError("PDF contains no extractable text. OCR is required for an image-only PDF.")
    return units


def _docx_para_text_with_links(paragraph) -> str:
    try:
        from docx.oxml.ns import qn
    except Exception:
        return clean_text(paragraph.text)
    rels = paragraph.part.rels
    parts: List[str] = []
    for child in paragraph._p:
        if child.tag == qn("w:hyperlink"):
            rid = child.get(qn("r:id"))
            anchor = child.get(qn("w:anchor"))
            label = "".join(t.text or "" for t in child.iter(qn("w:t")))
            url = ""
            if rid and rid in rels:
                try:
                    url = rels[rid].target_ref
                except Exception:
                    pass
            elif anchor:
                url = f"#{anchor}"
            label = clean_text(label)
            if label and url:
                parts.append(f"[{label}]({url})")
            elif label:
                parts.append(label)
        elif child.tag == qn("w:r"):
            parts.append("".join(t.text or "" for t in child.iter(qn("w:t"))))
    return clean_text("".join(parts)) or clean_text(paragraph.text)


def _docx_cell_text_with_links(cell) -> str:
    return clean_text("\n".join(filter(None, (_docx_para_text_with_links(p) for p in cell.paragraphs))), preserve_newlines=True)


def _looks_like_question(text: str) -> bool:
    text = clean_text(text)
    return 5 <= len(text) <= 350 and text.endswith("?") and bool(_QUESTION_START_RE.match(text))


def _looks_like_heading(text: str, style_name: str = "") -> bool:
    text = clean_text(text)
    if not text or len(text) > 180:
        return False
    if style_name.lower().startswith("heading"):
        return True
    if _HEADING_RE.match(text):
        return True
    letters = [c for c in text if c.isalpha()]
    return bool(letters and len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.88)


def _parse_docx_faq_records(paragraphs: Sequence[dict], filename: str) -> List[Dict]:
    records: List[Dict] = []
    current_q: str | None = None
    answer_parts: List[str] = []
    question_index = 0

    def flush() -> None:
        nonlocal current_q, answer_parts, question_index
        if current_q and answer_parts:
            answer = clean_text("\n".join(answer_parts), preserve_newlines=True)
            if answer:
                question_index += 1
                item = _unit(
                    question_index, f"Question: {current_q}\nAnswer: {answer}",
                    f"FAQ {question_index}", "docx", filename,
                    record_type="faq", question=current_q, answer=answer,
                    section="FAQs", section_title="FAQs",
                    parent_id=f"{filename}::faq::{question_index}", block_type="faq_record",
                )
                if item:
                    records.append(item)
        current_q = None
        answer_parts = []

    for para in paragraphs:
        text = para["text"]
        if not text or text.casefold() in {"faq", "faqs", "frequently asked questions"}:
            continue
        if _looks_like_question(text):
            flush()
            current_q = text
        elif current_q:
            answer_parts.append(text)
    flush()
    return records if len(records) >= 3 else []


def load_docx_text(path: Path, filename: str | None = None) -> List[Dict]:
    from docx import Document
    display_name = filename or path.name
    doc = Document(str(path))
    paragraphs: List[dict] = []
    for index, para in enumerate(doc.paragraphs, start=1):
        text = _docx_para_text_with_links(para)
        if text:
            paragraphs.append({"text": text, "style": getattr(para.style, "name", "") or "", "index": index})

    faq_records = _parse_docx_faq_records(paragraphs, display_name)
    if faq_records:
        return faq_records

    units: List[Dict] = []
    current_heading: str | None = None
    current_section_number: str | None = None
    for para in paragraphs:
        text = para["text"]
        is_heading = _looks_like_heading(text, para["style"])
        if is_heading:
            current_heading = text
            match = re.match(r"^(\d+(?:\.\d+)*)\b", text)
            current_section_number = match.group(1) if match else None
        item = _unit(
            para["index"], text, f"Paragraph {para['index']}", "docx", display_name,
            block_index=para["index"], block_type="heading" if is_heading else "paragraph",
            explicit_heading=text if is_heading else None,
            section=current_heading, section_number=current_section_number,
        )
        if item:
            units.append(item)

    for table_no, table in enumerate(doc.tables, start=1):
        rows = []
        for row_no, row in enumerate(table.rows, start=1):
            cells = [_docx_cell_text_with_links(cell) for cell in row.cells]
            if any(cells):
                rows.append(f"Row {row_no}: " + " | ".join(cells))
        if rows:
            item = _unit(
                len(paragraphs) + table_no, f"Table {table_no}\n" + "\n".join(rows),
                f"Table {table_no}", "docx", display_name,
                block_type="table", section=current_heading, section_number=current_section_number,
            )
            if item:
                units.append(item)
    return units


def _split_text_into_units(text: str, filename: str, source_type: str, words_per_unit: int = 600) -> List[Dict]:
    paragraphs = [clean_text(p, preserve_newlines=True) for p in re.split(r"\n\s*\n+", text) if clean_text(p)]
    if not paragraphs:
        paragraphs = [clean_text(text, preserve_newlines=True)]
    units, buffer, words = [], [], 0
    for paragraph in paragraphs:
        count = len(paragraph.split())
        if buffer and words + count > words_per_unit:
            no = len(units) + 1
            item = _unit(no, "\n\n".join(buffer), f"Part {no}", source_type, filename, block_type="text")
            if item:
                units.append(item)
            buffer, words = [], 0
        buffer.append(paragraph)
        words += count
    if buffer:
        no = len(units) + 1
        item = _unit(no, "\n\n".join(buffer), f"Part {no}", source_type, filename, block_type="text")
        if item:
            units.append(item)
    return units


def _rows_to_units(rows: List[str], filename: str, source_type: str, rows_per_unit: int = 30) -> List[Dict]:
    units = []
    for i in range(0, len(rows), rows_per_unit):
        no = len(units) + 1
        item = _unit(no, "\n".join(rows[i:i + rows_per_unit]), f"Rows {i + 1}-{min(i + rows_per_unit, len(rows))}", source_type, filename, block_type="rows")
        if item:
            units.append(item)
    return units


def load_csv_text(path: Path, filename: str | None = None) -> List[Dict]:
    display_name = filename or path.name
    raw = _read_text_file(path).splitlines()
    try:
        dialect = csv.Sniffer().sniff("\n".join(raw[:5]))
    except Exception:
        dialect = csv.excel
    rows = []
    reader = csv.DictReader(raw, dialect=dialect)
    if reader.fieldnames:
        for row_no, row in enumerate(reader, start=2):
            pairs = [f"{k}: {clean_text(str(v))}" for k, v in row.items() if clean_text(str(v))]
            if pairs:
                rows.append(f"Row {row_no} | " + " | ".join(pairs))
    return _rows_to_units(rows, display_name, "csv")


def load_xlsx_text(path: Path, filename: str | None = None) -> List[Dict]:
    import pandas as pd
    display_name = filename or path.name
    rows = []
    for sheet_name, df in pd.read_excel(path, sheet_name=None, dtype=str).items():
        df = df.fillna("")
        for idx, row in df.iterrows():
            pairs = [f"{col}: {clean_text(str(row[col]))}" for col in df.columns if clean_text(str(row[col]))]
            if pairs:
                rows.append(f"Sheet {sheet_name} Row {idx + 2} | " + " | ".join(pairs))
    return _rows_to_units(rows, display_name, "xlsx")


def load_html_text(path: Path, filename: str | None = None) -> List[Dict]:
    from bs4 import BeautifulSoup
    display_name = filename or path.name
    soup = BeautifulSoup(_read_text_file(path), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for anchor in soup.find_all("a"):
        href = (anchor.get("href") or "").strip()
        label = clean_text(anchor.get_text(" "))
        anchor.replace_with(f"[{label or href}]({href})" if href else label)
    return _split_text_into_units(soup.get_text("\n"), display_name, "html")


def load_pptx_text(path: Path, filename: str | None = None) -> List[Dict]:
    from pptx import Presentation
    display_name = filename or path.name
    prs = Presentation(str(path))
    units = []
    for slide_no, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and clean_text(shape.text):
                texts.append(clean_text(shape.text, preserve_newlines=True))
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [clean_text(cell.text) for cell in row.cells]
                    if any(cells):
                        texts.append(" | ".join(cells))
        item = _unit(slide_no, "\n".join(texts), f"Slide {slide_no}", "pptx", display_name, block_type="slide")
        if item:
            units.append(item)
    return units


def load_document_text(file_path: str, file_name: str | None = None) -> List[Dict]:
    path = Path(file_path)
    ext = path.suffix.lower()
    filename = file_name or path.name
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    if ext == ".pdf":
        return load_pdf_text(str(path), filename=filename)
    if ext == ".docx":
        return load_docx_text(path, filename=filename)
    if ext in {".txt", ".md", ".markdown"}:
        return _split_text_into_units(_read_text_file(path), filename, ext.lstrip("."))
    if ext == ".csv":
        return load_csv_text(path, filename)
    if ext in {".xlsx", ".xls"}:
        return load_xlsx_text(path, filename)
    if ext in {".html", ".htm"}:
        return load_html_text(path, filename)
    if ext == ".pptx":
        return load_pptx_text(path, filename)
    raise ValueError(f"Unsupported file type: {ext}")

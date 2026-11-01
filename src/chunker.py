from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple

CHUNKER_VERSION = "chunker-v7-clean-links-title-page"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“\[])")
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)*)[\s.\):-]+(.+)$")
_EXPLICIT_RE = re.compile(r"^(?:chapter|section|part|policy|clause|article)\s+(\d+(?:\.\d+)*)\b", re.I)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def split_sentences(text: str) -> List[str]:
    text = _norm(text)
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return parts or [text]


def _uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0


def detect_heading(line: str, explicit: str | None = None) -> tuple[str | None, str | None]:
    line = _norm(explicit or line)
    if not line or len(line) > 180:
        return None, None
    low = line.casefold()
    if (
        re.search(r"\.{4,}\s*\d*$", line)
        or low.startswith(("title:", "name:", "name :", "page ", "www.", "http", "prepared by", "date ", "issue "))
        or low.startswith("(c)")
        or re.match(r"^hrd\d+\b", low)
        or "@" in line
    ):
        return None, None

    match = _EXPLICIT_RE.match(line)
    if match:
        return match.group(1), line

    numbered = _NUMBERED_RE.match(line)
    if numbered:
        number, title = numbered.group(1), numbered.group(2).strip()
        word_count = len(title.split())
        colon_tail = title.split(":", 1)[1].split() if ":" in title else []
        title_words = re.findall(r"[A-Za-z][A-Za-z'&/-]*", title)
        minor = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
        significant = [w for w in title_words if w.casefold() not in minor]
        title_case_ratio = (sum(1 for w in significant if w[0].isupper()) / len(significant)) if significant else 0.0
        looks_titled = _uppercase_ratio(title) >= 0.55 or (word_count <= 10 and title_case_ratio >= 0.75)
        if "." in number and _uppercase_ratio(title) < 0.65:
            looks_titled = False
        if word_count <= 12 and len(colon_tail) <= 3 and looks_titled:
            return number, line

    if len(line.split()) <= 14 and _uppercase_ratio(line) >= 0.82 and not line.endswith((".", ",", ";")):
        return None, line

    words = re.findall(r"[A-Za-z][A-Za-z'&/-]*", line)
    if 2 <= len(words) <= 8 and ":" not in line and not line.endswith((".", "?", "!", ":", ",", ";")):
        major = sum(1 for w in words if w[0].isupper())
        if major / len(words) >= 0.8:
            return None, line
    return None, None


def choose_chunk_params(total_words: int) -> Tuple[int, int]:
    if total_words < 700:
        return 140, 1
    if total_words < 5_000:
        return 220, 1
    return 260, 1


def _is_toc(text: str) -> bool:
    low = text.casefold()
    if "table of contents" in low:
        return True
    return len(re.findall(r"\.{4,}\s*\d+", text)) >= 3


def _content_role(
    text: str,
    page: int,
    section: str | None,
    explicit_role: str | None = None,
    block_type: str | None = None,
) -> str:
    low = text.casefold().strip()
    if block_type == "links" or low.startswith("document links:"):
        return "links"
    if explicit_role in {"toc", "front_matter"}:
        return explicit_role
    if _is_toc(text):
        return "toc"
    if page <= 2 and any(
        term in low
        for term in (
            "document control", "version history", "distribution list",
            "authorisation", "authorization",
        )
    ):
        return "front_matter"

    # A short cover-page title plus a URL is useful for document identification
    # but should not answer substantive questions such as "what is the purpose".
    if page == 1 and len(text.split()) <= 55 and (
        "http://" in low or "https://" in low or "www." in low
    ):
        return "title_page"
    if section:
        return "policy_body"
    return "body"


def _stable_parent_id(filename: str, section_number: str | None, section_title: str | None, page: int) -> str:
    raw = f"{filename}|{section_number or ''}|{section_title or ''}|{page if not section_title else ''}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:18]


def _logical_segments(unit: Dict) -> List[Dict]:
    text = str(unit.get("text") or "").strip()
    if not text:
        return []
    if unit.get("record_type") == "faq":
        return [dict(unit)]

    lines = [line.strip() for line in text.splitlines() if line.strip()] or [text]
    segments: List[Dict] = []
    buffer: List[str] = []
    for line in lines:
        sec_no, heading = detect_heading(line, explicit=unit.get("explicit_heading") if len(lines) == 1 else None)
        if heading:
            if buffer:
                segment = dict(unit)
                segment["text"] = "\n".join(buffer)
                segments.append(segment)
                buffer = []
            heading_segment = dict(unit)
            heading_segment["text"] = heading
            heading_segment["detected_heading"] = heading
            heading_segment["detected_section_number"] = sec_no
            heading_segment["block_type"] = "heading"
            segments.append(heading_segment)
        else:
            buffer.append(line)
    if buffer:
        segment = dict(unit)
        segment["text"] = "\n".join(buffer)
        segments.append(segment)
    return segments


def _split_large_block(block: Dict, max_words: int) -> List[Dict]:
    text = _norm(block.get("text", ""))
    if len(text.split()) <= max_words:
        b = dict(block)
        b["text"] = text
        return [b]
    sentences = split_sentences(text)
    parts, current, words = [], [], 0
    for sentence in sentences:
        count = len(sentence.split())
        if current and words + count > max_words:
            b = dict(block)
            b["text"] = " ".join(current)
            parts.append(b)
            current = current[-1:]
            words = sum(len(s.split()) for s in current)
        current.append(sentence)
        words += count
    if current:
        b = dict(block)
        b["text"] = " ".join(current)
        parts.append(b)
    return parts


def _merge_overlapping_texts(parts: List[str], max_overlap_words: int = 140) -> str:
    merged_words: List[str] = []
    for part in parts:
        words = _norm(part).split()
        if not words:
            continue
        if not merged_words:
            merged_words.extend(words)
            continue
        max_check = min(max_overlap_words, len(merged_words), len(words))
        overlap = 0
        for size in range(max_check, 0, -1):
            if [w.casefold() for w in merged_words[-size:]] == [w.casefold() for w in words[:size]]:
                overlap = size
                break
        merged_words.extend(words[overlap:])
    return " ".join(merged_words).strip()


def chunk_text(units: List[Dict], max_words: int, overlap_sentences: int) -> List[Dict]:
    chunks: List[Dict] = []

    # Atomic FAQ records are never merged with neighboring questions.
    for unit in units:
        if unit.get("record_type") != "faq":
            continue
        question = _norm(unit.get("question", ""))
        answer = str(unit.get("answer") or "").strip()
        text = f"Question: {question}\nAnswer: {answer}".strip()
        chunks.append({
            "text": text,
            "search_text": f"{question}\n{answer}",
            "answer": answer,
            "question": question,
            "record_type": "faq",
            "pages": [int(unit.get("page", 1))],
            "labels": [unit.get("label", f"FAQ {len(chunks) + 1}")],
            "source_type": unit.get("source_type", "docx"),
            "filename": unit.get("filename", ""),
            "section": "FAQs",
            "section_title": "FAQs",
            "section_number": None,
            "parent_id": unit.get("parent_id") or _stable_parent_id(unit.get("filename", ""), None, "FAQs", int(unit.get("page", 1))),
            "content_role": "faq",
        })

    logical: List[Dict] = []
    for unit in (u for u in units if u.get("record_type") != "faq"):
        logical.extend(_logical_segments(unit))

    prepared: List[Dict] = []
    current_section: str | None = None
    current_section_number: str | None = None
    current_parent_id: str | None = None

    for segment in logical:
        text = _norm(segment.get("text", ""))
        if not text:
            continue

        # Keep link-only blocks isolated from the surrounding policy section.
        # This prevents a cover-page website from becoming part of the parent
        # text for a substantive section.
        if segment.get("block_type") == "links" or text.casefold().startswith("document links:"):
            segment = dict(segment)
            segment["section"] = None
            segment["section_number"] = None
            segment["parent_id"] = _stable_parent_id(
                segment.get("filename", ""), None, "__links__", int(segment.get("page", 1))
            )
            prepared.append(segment)
            continue

        sec_no, heading = detect_heading(text, explicit=segment.get("detected_heading") or segment.get("explicit_heading"))
        if heading:
            proposed_number = sec_no or segment.get("detected_section_number") or segment.get("section_number")
            repeated_number_subheading = bool(
                current_section and proposed_number and current_section_number == proposed_number
                and _norm(current_section).casefold() != _norm(heading).casefold()
            )
            segment = dict(segment)
            segment["is_heading"] = True
            if repeated_number_subheading:
                segment["subsection"] = heading
                segment["section"] = current_section
                segment["section_number"] = current_section_number
                segment["parent_id"] = current_parent_id
            else:
                current_section = heading
                current_section_number = proposed_number
                current_parent_id = _stable_parent_id(
                    segment.get("filename", ""), current_section_number, current_section, int(segment.get("page", 1))
                )
                segment["section"] = current_section
                segment["section_number"] = current_section_number
                segment["parent_id"] = current_parent_id
            prepared.append(segment)
            continue

        segment = dict(segment)
        if segment.get("section") and not current_section:
            current_section = segment.get("section")
            current_section_number = segment.get("section_number")
        if current_parent_id is None:
            current_parent_id = _stable_parent_id(
                segment.get("filename", ""), current_section_number, current_section, int(segment.get("page", 1))
            )
        segment["section"] = current_section
        segment["section_number"] = current_section_number
        segment["parent_id"] = current_parent_id
        prepared.extend(_split_large_block(segment, max_words))

    current_blocks: List[Dict] = []
    current_words = 0

    def flush() -> None:
        nonlocal current_blocks, current_words
        if not current_blocks:
            return
        text_parts, seen = [], set()
        for block in current_blocks:
            text = _norm(block.get("text", ""))
            sig = text.casefold()
            if text and sig not in seen:
                seen.add(sig)
                text_parts.append(text)
        if not text_parts:
            current_blocks, current_words = [], 0
            return
        first = current_blocks[0]
        pages = sorted({int(b.get("page", 1)) for b in current_blocks})
        labels = list(dict.fromkeys(str(b.get("label", f"Page {pages[0]}")) for b in current_blocks))
        section = first.get("section")
        filename = first.get("filename", "")
        text = "\n\n".join(text_parts)
        chunks.append({
            "text": text,
            "search_text": "\n".join(filter(None, [filename, section or "", first.get("subsection", ""), text])),
            "record_type": "policy",
            "pages": pages,
            "labels": labels,
            "source_type": first.get("source_type", "document"),
            "filename": filename,
            "section": section,
            "section_title": section,
            "section_number": first.get("section_number"),
            "subsection": first.get("subsection"),
            "parent_id": first.get("parent_id"),
            "content_role": _content_role(text, pages[0], section, first.get("page_role"), first.get("block_type")),
        })
        current_blocks, current_words = [], 0

    for block in prepared:
        if block.get("is_heading"):
            flush()
            current_blocks = [block]
            current_words = len(_norm(block.get("text", "")).split())
            continue
        if current_blocks:
            first = current_blocks[0]
            if block.get("filename") != first.get("filename") or block.get("parent_id") != first.get("parent_id"):
                flush()
        count = len(_norm(block.get("text", "")).split())
        if current_blocks and current_words + count > max_words:
            previous = current_blocks[-overlap_sentences:] if overlap_sentences else []
            flush()
            current_blocks = [dict(b) for b in previous]
            current_words = sum(len(_norm(b.get("text", "")).split()) for b in current_blocks)
        current_blocks.append(block)
        current_words += count
    flush()

    # Build one clean, overlap-free parent section for exact section retrieval.
    parent_parts: dict[tuple[str, str], list[str]] = defaultdict(list)
    for chunk in chunks:
        key = (chunk.get("filename", ""), chunk.get("parent_id", ""))
        parent_parts[key].append(chunk["text"])
    parent_texts = {key: _merge_overlapping_texts(parts)[:14000] for key, parts in parent_parts.items()}

    final: List[Dict] = []
    for chunk in chunks:
        if len(chunk.get("text", "").split()) < 3:
            continue
        chunk["chunk_id"] = len(final)
        key = (chunk.get("filename", ""), chunk.get("parent_id", ""))
        chunk["parent_text"] = parent_texts.get(key, chunk["text"])
        final.append(chunk)
    return final

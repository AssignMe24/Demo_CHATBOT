from __future__ import annotations

import html as html_lib
import os
import re
import requests

from confidence import (
    ANSWERED_PARTIAL,
    ANSWERED_SUPPORTED,
    GENERATION_FAILURE,
    REFUSED_INSUFFICIENT_EVIDENCE,
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_NUM_THREAD = int(os.getenv("OLLAMA_NUM_THREAD", "4"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "7600"))
MAX_NUM_PREDICT = int(os.getenv("RAG_MAX_NUM_PREDICT", "640"))
ENABLE_LLM_VERIFY = os.getenv("RAG_LLM_VERIFY", "0") == "1"
ENABLE_QUERY_REWRITE = os.getenv("RAG_QUERY_REWRITE", "1") == "1"
ENABLE_FAQ_DISAMBIGUATION = os.getenv("RAG_FAQ_LLM_DISAMBIGUATION", "1") == "1"
NOT_FOUND = "Answer not found in the document."

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "could", "did", "do", "does",
    "for", "from", "has", "have", "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or",
    "our", "please", "should", "that", "the", "their", "this", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "you", "your", "tell", "show", "give",
    "explain", "about", "document", "policy",
}

_ANSWER_PROMPT = """You are a precise document question-answering assistant.
Use only the supplied evidence. Never use outside knowledge.

Rules:
1. Answer the user's question directly and naturally.
2. Do not mention evidence, context, retrieval, chunks, sources, filenames, page labels, or document names unless the user explicitly asks for them.
3. Never repeat metadata such as "SOURCE:", "SECTION:", "Question:", "Answer:", or "Document links:".
4. Do not add commentary such as "the evidence supports only...", "the evidence does not state...", or "based on the provided evidence".
5. If only part of the question can be answered, provide the supported part directly without discussing evidence limitations.
6. For questions asking for documents, requirements, conditions, or steps, include every directly relevant item present in the supplied evidence, including trailing notes.
7. Keep alternatives and conditional cases separate. Do not present requirements for different cases as if every requirement applies simultaneously.
8. Preserve names, phone numbers, email addresses, deadlines, durations, amounts, codes, and ordered steps exactly.
9. For procedures, keep the steps in source order.
10. When the relevant evidence contains a URL, preserve that exact URL as a clickable link. Never create, guess, alter, complete, or substitute a URL. If the relevant evidence contains no URL, do not provide one.
11. Do not return a website merely because it appears on a cover page or elsewhere in the document; include only links connected to the answer.
12. If the question asks about more than one topic, answer every requested topic under separate short headings. Do not answer only the best-matching topic.
13. For questions asking what to do first, immediately, or initially, identify the earliest action or phase and include all parallel actions that share that same initial deadline or phase.
14. If the evidence does not answer the question, output exactly: Answer not found in the document.
15. Be concise unless complete details or a summary are requested.

--- EVIDENCE START ---
{context}
--- EVIDENCE END ---
Question: {question}
Answer:"""

_VERIFY_PROMPT = """Determine whether every material claim in the answer is supported by the evidence.
Return only one label: ANSWERED_SUPPORTED, ANSWERED_PARTIAL, or UNSUPPORTED.
Evidence:
{context}
Answer:
{answer}"""

_REWRITE_PROMPT = """Rewrite the user's question as two short document-search queries.
Preserve intent, names, acronyms, dates, and numbers. Prefer terminology likely to appear in the indexed source documents.
Do not answer. Return only two lines.
Question: {question}
Search queries:"""


_REQUIREMENTS_PROMPT = """You are a precise document question-answering assistant.
Use only the supplied evidence. Never use outside knowledge.

The user is asking for requirements, documents, conditions, eligibility, or steps.
Return every directly relevant item. Preserve each condition exactly.
When one source requirement contains several items joined by "and" or "or", retain every joined item; do not shorten the sentence after the first item.
Exclude later changes, company response steps, return-to-work rules, and other subsequent actions unless the user explicitly asks for them.
If the question asks what to do first or immediately, include all actions in the earliest stated phase or deadline, not just one item.
If the question asks about multiple topics, cover each topic separately.
When requirements differ by case, group them separately using this form:
- For <case or condition>: <requirement>
Never merge two different cases into one universal requirement.
Preserve names, numbers, dates, deadlines, codes, links, and step order exactly.
Do not mention evidence, context, sources, filenames, pages, or retrieval.
If the evidence does not answer the question, output exactly: Answer not found in the document.

--- EVIDENCE START ---
{context}
--- EVIDENCE END ---
Question: {question}
Answer:"""


_ELIGIBILITY_PROMPT = """You are a precise document question-answering assistant.
Use only the supplied evidence. Never use outside knowledge.

The user is asking when someone qualifies, is eligible, or is entitled.
- State the qualifying condition or conditions first.
- Then list the rights, benefits, or entitlements that follow.
- Do not treat a benefit as though it were a qualification condition.
- Keep ordinary, additional, total, paid, and unpaid entitlements separate.
- Preserve service periods, deadlines, durations, amounts, and exceptions exactly.
- Do not include return-to-work rules or later procedures unless requested.
- Do not mention evidence, context, sources, filenames, pages, or retrieval.
If the evidence does not answer the question, output exactly: Answer not found in the document.

--- EVIDENCE START ---
{context}
--- EVIDENCE END ---
Question: {question}
Answer:"""

_MULTI_TOPIC_PROMPT = """You are a precise document question-answering assistant.
Use only the supplied evidence. Never use outside knowledge.

The user asks about multiple topics.
- Identify every requested topic.
- Answer each topic under a separate short Markdown heading.
- Use only details relevant to that topic.
- Do not omit a topic merely because another topic has stronger or longer evidence.
- Preserve names, numbers, dates, deadlines, amounts, conditions, and links exactly.
- Keep every number attached to the same subject it modifies in the source. Never move a duration, percentage, or amount from one topic to another.
- When an entitlement is divided into ordinary, additional, total, first, remaining, higher, standard, or other named phases, include each source-defined phase and its total when present.
- When a payment is divided into multiple rate periods, include each rate period and the total payment period when present.
- Do not mention evidence, context, sources, filenames, pages, or retrieval.
If none of the requested topics is answered by the evidence, output exactly: Answer not found in the document.

--- EVIDENCE START ---
{context}
--- EVIDENCE END ---
Question: {question}
Answer:"""



_INITIAL_ACTION_PROMPT = """You are a precise document question-answering assistant.
Use only the supplied evidence. Never use outside knowledge.

The user asks what to do first, immediately, or initially.
- Identify the earliest relevant phase, deadline, or opening instruction block.
- Include every parallel action attached to that earliest phase or deadline.
- Do not stop after numbered item 1 when items 2 or 3 are part of the same opening phase.
- If the evidence states an immediate report plus an evidence-preservation action, include both.
- Exclude later-stage actions, return-to-work rules, and unrelated procedures.
- Preserve names, contacts, deadlines, systems, and ordered steps exactly.
- Do not mention evidence, context, sources, filenames, pages, or retrieval.
If the evidence does not answer the question, output exactly: Answer not found in the document.

--- EVIDENCE START ---
{context}
--- EVIDENCE END ---
Question: {question}
Answer:"""

_CONDITION_REPAIR_PROMPT = """Rewrite the draft answer using only the supplied evidence.
The draft has lost or merged conditional cases. Restore every relevant case or condition explicitly.
Keep different cases separate. Do not add facts, commentary, sources, or filenames.
Use concise Markdown bullets.

Evidence:
{context}

Question: {question}

Draft answer:
{answer}

Corrected answer:"""

_COMPLETION_REPAIR_PROMPT = """Rewrite the draft as a complete, concise answer using only the supplied evidence.
The previous draft ended mid-sentence or was cut off. Preserve all supported facts already present, finish every sentence, and do not add commentary, sources, filenames, or unsupported details.

Evidence:
{context}

Question: {question}

Truncated draft:
{answer}

Complete answer:"""

_NUMERIC_REPAIR_PROMPT = """Rewrite the draft using only the supplied evidence.
One or more numbers may be attached to the wrong subject. Keep every duration, percentage, amount, deadline, and count with the exact concept it modifies in the evidence. Do not transfer a number from pay to leave, from one eligibility condition to another, or from one topic to another. Remove any numeric claim that cannot be stated exactly from the evidence.
Do not mention evidence, sources, filenames, pages, or this correction.

Evidence:
{context}

Question: {question}

Draft answer:
{answer}

Corrected answer:"""

_MULTI_TOPIC_COMPLETENESS_PROMPT = """Rewrite the draft as a complete, concise answer using only the supplied evidence.
The draft omitted one or more important source-defined phases, totals, durations, or rate periods.
Cover every topic requested by the user under separate short headings. Include every mandatory fact listed below, but do not add unsupported information. Keep each number attached to its exact source concept.
Do not mention evidence, sources, filenames, pages, retrieval, or this correction.

Mandatory source facts that were missing:
{facts}

Evidence:
{context}

Question: {question}

Draft answer:
{answer}

Complete corrected answer:"""

_FAQ_SELECT_PROMPT = """Choose the candidate FAQ that answers the same information need as the user's question.
Focus on the requested output type: identifier, access link, deadline, consequence, contact, procedure, eligibility, amount, status, or explanation.
Do not choose a candidate merely because it shares a product, portal, policy, or department name.
Use the answer preview only to distinguish the kind of answer; do not copy or answer the user.
Return only the candidate number. Return 0 if none matches.

User question: {question}

Candidates:
{candidates}

Candidate number:"""

_MD_LINK_RE = re.compile(r"\[([^\]]{1,200})\]\((https?://[^\s\)]+)\)")
_URL_RE = re.compile(r"https?://[^\s\]\)<>\"']+")
_META_SENTENCE_PATTERNS = (
    r"\bevidence supports only\b",
    r"\bthe evidence does not\b",
    r"\bevidence does not\b",
    r"\bbased on (?:the )?(?:provided )?evidence\b",
    r"\baccording to (?:the )?(?:provided )?evidence\b",
    r"\baccording to (?:the )?context\b",
    r"\bthe context does not\b",
    r"\bprovided context\b",
    r"\bprovided evidence\b",
    r"\bretrieved (?:context|evidence|chunks?)\b",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(text or "").casefold())).strip()


def _tokens(text: str) -> list[str]:
    return [t for t in _normalize(text).split() if len(t) > 1 and t not in STOPWORDS]


def explicit_link_request(question: str) -> bool:
    """Return True when the user explicitly asks for a web address.

    This is used only to prioritize link-only retrieval blocks. It does not
    control whether a source-backed link is displayed: relevant links already
    present in the retrieved document are always allowed.
    """
    q = _normalize(question)
    return bool(re.search(r"\b(link|url|website|web address|hyperlink)\b", q))


def _strip_source_metadata(text: str) -> str:
    """Remove source headers accidentally copied by a small local model."""
    value = str(text or "")

    # Remove machine-readable metadata headers while preserving answer content.
    value = re.sub(
        r"(?is)(?:^|\n|\s)SOURCE:\s*.*?(?=(?:\bQuestion:|\bAnswer:|\bSOURCE:|\n|$))",
        " ",
        value,
    )
    value = re.sub(r"(?im)^\s*(?:SECTION|PAGE|LOCATION|EVIDENCE|CONTEXT)\s*:\s*.*$", " ", value)
    value = re.sub(r"(?im)^\s*Document links\s*:\s*.*$", " ", value)

    # Remove repeating PDF headers even when extraction flattened them into the
    # middle of a substantive paragraph. The patterns require page/version or
    # confidentiality markers so normal document titles in an answer survive.
    value = re.sub(
        r"(?is)\bTitle\s*:\s*.{1,180}?\s+Page(?:\s*\d+\s*:\s*\d+|\s*:\s*\d+|\s+\d+)\b",
        " ",
        value,
    )
    value = re.sub(r"(?i)\b[A-Z]{2,8}\d{2,8}\s+Issue\s+[0-9.]+\b", " ", value)
    value = re.sub(
        r"(?i)\(C\)\s+[^\n.]{1,160}?Company Confidential\b",
        " ",
        value,
    )

    # Remove copied FAQ labels. If an Answer: label exists, retain its content.
    value = re.sub(
        r"(?is)(?:^|\s)Question:\s*.*?(?=(?:\bAnswer:|\bSOURCE:|\n|$))",
        " ",
        value,
    )
    value = re.sub(r"(?i)\bAnswer\s*:\s*", "", value)
    return value


def _remove_meta_commentary(text: str) -> str:
    """Remove model commentary about evidence sufficiency.

    The assistant should either answer or refuse. Internal verification language
    is retained in backend fields, not shown to the employee. Existing line
    breaks are preserved so ordered procedures remain readable.
    """
    value = str(text or "").strip()

    value = re.sub(
        r"(?is)^\s*(?:here(?:'s| is)\s+)?(?:the\s+)?(?:answer|information|details)\s+"
        r"(?:about|on|regarding)?\s*.*?\bbased on (?:the )?(?:provided )?(?:evidence|context)\s*:\s*",
        "",
        value,
        count=1,
    )
    value = re.sub(
        r"(?is)^\s*(?:according to|based on)\s+(?:the\s+)?(?:provided\s+)?(?:evidence|context|document)[,:]?\s*",
        "",
        value,
        count=1,
    )
    value = re.sub(
        r"(?is)^\s*according to\s+[^,\n]{1,180}\.(?:pdf|docx|txt|md|xlsx|pptx)\s*,?\s*",
        "",
        value,
    )
    value = re.sub(
        r"(?is)\([^()]{0,500}(?:evidence supports only|evidence does not|provided evidence|provided context)[^()]{0,500}\)",
        " ",
        value,
    )

    cleaned_lines: list[str] = []
    for line in value.splitlines() or [value]:
        line = line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        kept_sentences: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = sentence.strip()
            if not sentence:
                continue
            low = sentence.casefold()
            if any(re.search(pattern, low) for pattern in _META_SENTENCE_PATTERNS):
                continue
            kept_sentences.append(sentence)
        if kept_sentences:
            cleaned_lines.append(" ".join(kept_sentences))

    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()
    return "\n".join(cleaned_lines).strip()


def _apply_link_policy(text: str, question: str) -> str:
    """Preserve candidate links until they are checked against retrieved text.

    The final source-validation step lives in ``app.py`` because it has access
    to the exact retrieved chunks. This function must not invent or append a
    link; it only avoids deleting a link that may have come from the document.
    """
    value = str(text or "")
    value = re.sub(r"(?im)^\s*Document links\s*:\s*$", "", value)
    return value


def _canonical_url(url: str) -> str:
    return html_lib.unescape(str(url or "")).strip().rstrip(".,;:!?)\"]}")


def _extract_source_links(text: str) -> list[tuple[str, str]]:
    value = str(text or "")
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, raw_url in _MD_LINK_RE.findall(value):
        url = _canonical_url(raw_url)
        if url and url not in seen:
            seen.add(url)
            links.append((re.sub(r"\s+", " ", label).strip() or "Link", url))
    for raw_url in _URL_RE.findall(value):
        url = _canonical_url(raw_url)
        if url and url not in seen:
            seen.add(url)
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            links.append((domain or "Link", url))
    return links


def _remove_unverified_links(answer: str, allowed_urls: set[str]) -> str:
    value = str(answer or "")
    placeholders: dict[str, str] = {}

    def protect_markdown(match: re.Match) -> str:
        label = match.group(1).strip() or "Link"
        url = _canonical_url(match.group(2))
        if url not in allowed_urls:
            return label
        token = f"__SOURCE_LINK_{len(placeholders)}__"
        placeholders[token] = f"[{label}]({url})"
        return token

    value = _MD_LINK_RE.sub(protect_markdown, value)

    def protect_raw(match: re.Match) -> str:
        url = _canonical_url(match.group(0))
        if url not in allowed_urls:
            return ""
        token = f"__SOURCE_LINK_{len(placeholders)}__"
        placeholders[token] = url
        return token

    value = _URL_RE.sub(protect_raw, value)
    for token, original in placeholders.items():
        value = value.replace(token, original)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"\n\s*\n\s*\n+", "\n\n", value)
    return value.strip()


def apply_source_link_policy(answer: str, question: str, source_text: str) -> str:
    """Display only relevant URLs that occur verbatim in retrieved source text.

    A source-backed link is retained automatically. A missing relevant link may
    be appended from the retrieved source. Any URL not present in source text is
    removed, so the chatbot cannot invent, guess, or modify links.
    """
    value = sanitize_answer(answer, question)
    source_links = _extract_source_links(source_text)
    allowed_urls = {url for _label, url in source_links}
    value = _remove_unverified_links(value, allowed_urls)

    if not source_links or _MD_LINK_RE.search(value) or _URL_RE.search(value):
        return value

    q_tokens = set(_tokens(question))
    a_tokens = set(_tokens(value))
    q_norm = _normalize(question)
    a_norm = _normalize(value)
    ranked: list[tuple[float, str, str]] = []

    for label, url in source_links:
        label_norm = _normalize(label)
        target_tokens = set(_tokens(f"{label} {url}"))
        answer_overlap = len(target_tokens & a_tokens)
        question_overlap = len(target_tokens & q_tokens)
        score = 3.0 * answer_overlap + 1.25 * question_overlap
        if label_norm and label_norm in a_norm:
            score += 5.0
        if label_norm and label_norm in q_norm:
            score += 2.5
        if re.search(r"\b(portal|guide|page|form|app|calendar|list|recording)\b", label_norm):
            score += 0.75
        ranked.append((score, label, url))

    if not ranked:
        return value
    ranked.sort(reverse=True, key=lambda item: item[0])
    best_score, best_label, best_url = ranked[0]
    if best_score < 2.0:
        return value

    safe_label = re.sub(r"[\[\]\(\)]", "", best_label).strip() or "Open link"
    return f"{value.rstrip()}\n\n[{safe_label}]({best_url})"



def _format_procedure_layout(text: str) -> str:
    """Restore readable Markdown structure lost during document extraction."""
    value = str(text or "").replace("\u00a0", " ").replace("\u202f", " ")

    # Keep Markdown link labels on one line. A hyphen inside a link title is
    # punctuation, not a new list item.
    def _normalize_markdown_link(match: re.Match) -> str:
        label = re.sub(r"\s+", " ", match.group(1)).strip()
        return f"[{label}]({match.group(2)})"

    value = _MD_LINK_RE.sub(_normalize_markdown_link, value)

    value = re.sub(
        r"(?i)(?<![A-Za-z0-9])Step\s*(\d+)\s*[:.)-]?\s*",
        lambda match: ("" if match.start() == 0 else "\n") + f"{match.group(1)}. ",
        value,
    )
    value = re.sub(r"\s*[•▪◦‣]\s*", "\n- ", value)

    # Convert trailing labelled details into explicit Markdown bullets. This
    # prevents Streamlit from visually merging them into the final numbered step.
    label_pattern = re.compile(
        r"(?<!\*)\b(Note|(?:[A-Z][a-z]+(?:[ -]+[A-Z][a-z]+){0,3}) Details)\s*:\s*"
    )
    matches = list(label_pattern.finditer(value))
    if matches:
        rebuilt: list[str] = []
        cursor = 0
        for idx, match in enumerate(matches):
            rebuilt.append(value[cursor:match.start()].rstrip())
            label = re.sub(r"\s+", " ", match.group(1)).strip()
            next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(value)
            body = value[match.end():next_start].strip()
            rebuilt.append(f"\n\n- **{label}:** {body}".rstrip())
            cursor = next_start
        rebuilt.append(value[cursor:])
        value = "".join(rebuilt)

    value = re.sub(r"[ \t]+(?=\d+\.\s+)", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _requirements_question(question: str) -> bool:
    q = _normalize(question)
    if re.search(
        r"\b(apply|application|document|documents|submit|submission|required|requirement|requirements|condition|conditions|eligibility|eligible|entitlement|entitled|steps|procedure|process|notify|notification|provide|what do i need|what is needed)\b",
        q,
    ):
        return True
    # Generic action/compliance questions should retain every directly relevant
    # required action from a matched section, rather than only the first one.
    return bool(re.search(
        r"\bwhat should (?:the )?(?:company|organisation|organization|employee|staff|individual|person|team) do\b",
        q,
    ))


def _eligibility_question(question: str) -> bool:
    q = _normalize(question)
    return bool(re.search(
        r"\b(eligible|eligibility|entitled|entitlement|entitlements|rights?|benefits?|qualify|qualified|qualification|who can|when can|under what conditions|criteria)\b",
        q,
    ))


def _initial_action_question(question: str) -> bool:
    q = _normalize(question)
    return bool(re.search(
        r"\b(first|first step|initially|initial action|immediately|right away|as soon as possible|what am i expected to do first)\b",
        q,
    ))



def _extract_initial_phase_actions(context: str) -> tuple[str, list[str]]:
    """Extract all parallel actions under the earliest explicit phase/deadline.

    The logic is structural and corpus-neutral. It recognises phrases such as
    "immediately" and "within 5 business days", then reads the bullet items
    attached to that phase until the next timeline marker.
    """
    value = str(context or "").replace("\u00a0", " ").replace("\u202f", " ")
    # PDF extraction often flattens bullets into one long line. Restore generic
    # bullet boundaries before identifying parallel actions in the first phase.
    value = re.sub(r"\s*[•▪◦‣]\s*", "\n- ", value)
    value = re.sub(r"(?<!\w)\s+[–—]\s+(?=[A-Z])", "\n- ", value)
    if not value.strip():
        return "", []

    marker_re = re.compile(
        r"\b(?:immediately|without delay|as soon as possible|"
        r"within\s+(?:the\s+first\s+)?\d+\s+(?:business\s+)?(?:hours?|days?|weeks?|months?))\b",
        re.I,
    )
    matches = list(marker_re.finditer(value))
    if not matches:
        return "", []

    def marker_rank(match: re.Match) -> tuple[int, int]:
        phrase = match.group(0).casefold()
        if any(term in phrase for term in ("immediately", "without delay", "as soon as possible")):
            return 0, match.start()
        number_match = re.search(r"\d+", phrase)
        number = int(number_match.group(0)) if number_match else 9999
        if "hour" in phrase:
            units = number
        elif "day" in phrase:
            units = number * 24
        elif "week" in phrase:
            units = number * 24 * 7
        else:
            units = number * 24 * 30
        return units, match.start()

    chosen = min(matches, key=marker_rank)
    next_markers = [match for match in matches if match.start() > chosen.end()]
    end = min((match.start() for match in next_markers), default=len(value))
    segment = value[chosen.start():end]

    # Stop before explanatory or later-stage prose when it follows the bullet block.
    segment = re.split(
        r"\n\s*(?:o\s+)?(?:This ensures|After this|Thereafter|Subsequently|During the next|Later)\b",
        segment,
        maxsplit=1,
        flags=re.I,
    )[0]

    lines = [re.sub(r"\s+", " ", line).strip() for line in segment.splitlines()]
    actions: list[str] = []
    current = ""
    for line in lines:
        if not line:
            continue
        bullet = re.match(r"^[•▪◦‣\-–—]+\s*(.+)$", line)
        numbered = re.match(r"^\d+[.)]\s+(.+)$", line)
        if bullet or numbered:
            if current:
                actions.append(current.strip())
            current = (bullet or numbered).group(1).strip()
        elif current:
            current += " " + line
    if current:
        actions.append(current.strip())

    # Keep actionable statements only and deduplicate overlap artefacts.
    action_re = re.compile(
        r"\b(update|notify|inform|contact|call|email|submit|provide|preserve|"
        r"complete|review|schedule|set up|setup|meet|report|record|share|select|login|log in)\b",
        re.I,
    )
    output: list[str] = []
    seen: set[str] = set()
    for action in actions:
        action = action.strip(" .;:")
        sig = _normalize(action)[:220]
        if len(action.split()) >= 3 and action_re.search(action) and sig not in seen:
            seen.add(sig)
            output.append(action + ("." if not re.search(r"[.!?]$", action) else ""))
    label = re.sub(r"\s+", " ", chosen.group(0)).strip()
    return label, output[:8]


def _initial_action_coverage(answer: str, actions: list[str]) -> float:
    if not actions:
        return 1.0
    answer_tokens = set(_tokens(answer))
    covered = 0
    for action in actions:
        important = {token for token in _tokens(action) if len(token) > 3}
        if not important:
            continue
        if len(important & answer_tokens) / len(important) >= 0.45:
            covered += 1
    return covered / len(actions)


def _render_initial_actions(label: str, actions: list[str]) -> str:
    heading = label[:1].upper() + label[1:] if label else "Initial actions"
    return heading.rstrip(":") + ":\n" + "\n".join(f"- {action}" for action in actions)


def _initial_marker_rank(label: str) -> int:
    """Convert a generic time marker into comparable hours."""
    value = _normalize(label)
    if any(term in value for term in ("immediately", "without delay", "as soon as possible")):
        return 0
    match = re.search(r"\b(\d+)\s+(business\s+)?(hour|day|week|month)s?\b", value)
    if not match:
        return 10**9
    number = int(match.group(1))
    unit = match.group(3)
    return number * {"hour": 1, "day": 24, "week": 168, "month": 720}[unit]


def _best_initial_phase_actions(context_chunks: list[str], question: str = "") -> tuple[str, list[str]]:
    """Choose the best complete opening-phase action block.

    Each retrieved chunk is analysed independently before the context is joined.
    This prevents an early deadline in one section from absorbing unrelated text
    from a later section when PDF parent text has been flattened.
    """
    q_tokens = set(_tokens(question))
    candidates: list[tuple[tuple, str, list[str]]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    for raw in context_chunks:
        label, actions = _extract_initial_phase_actions(str(raw or ""))
        if not label or not actions:
            continue
        clean_actions: list[str] = []
        for action in actions:
            action = re.split(
                r"\s+(?:o\s+)?(?:This ensures|After this|Thereafter|Subsequently|During the next|Later)\b",
                action, maxsplit=1, flags=re.I,
            )[0].strip(" .;:")
            if action:
                clean_actions.append(action + ("" if re.search(r"[.!?]$", action) else "."))
        actions = list(dict.fromkeys(clean_actions))
        if not actions:
            continue

        signature = (_normalize(label), tuple(_normalize(a) for a in actions))
        if signature in seen:
            continue
        seen.add(signature)

        action_tokens = set(_tokens(" ".join(actions)))
        topic_overlap = len(q_tokens & action_tokens) / max(1, len(q_tokens))
        action_lengths = [len(action.split()) for action in actions]
        truncated_penalty = sum(1 for length in action_lengths if length < 4)
        long_action_penalty = sum(max(0, length - 40) for length in action_lengths)
        # Earlier phases win first; among equal phases prefer the block with the
        # most complete parallel actions. Very long prose captured as one action
        # is penalised because it usually indicates a flattened section boundary.
        rank = (
            _initial_marker_rank(label),
            -min(len(actions), 8),
            truncated_penalty,
            long_action_penalty,
            -topic_overlap,
            sum(action_lengths),
        )
        candidates.append((rank, label, actions))

    if not candidates:
        return "", []
    candidates.sort(key=lambda item: item[0])
    _rank, label, actions = candidates[0]
    return label, actions[:8]


def _context_numeric_fact_sentences(context: str, question: str) -> list[str]:
    """Return important numeric phase/rate sentences relevant to a broad question."""
    value = re.sub(r"\s+", " ", str(context or "")).strip()
    if not value:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", value) if part.strip()]
    q_terms = set(_tokens(question))
    q_norm = _normalize(question)
    qualifier_re = re.compile(
        r"\b(ordinary|additional|total|first|remaining|maximum|minimum|higher|standard|lower|"
        r"up to|rate|period|weeks?|days?|months?|years?|percent|percentage|earnings)\b",
        re.I,
    )
    primary_fact_re = re.compile(
        r"\b(ordinary|additional|total|first|remaining|maximum|higher|standard|lower|"
        r"payable|payment|pay|rate|earnings|percentage|percent)\b",
        re.I,
    )
    asks_timing = bool(re.search(r"\b(when|start|begin|commence|before|after|deadline|notice)\b", q_norm))
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        # Remove clause identifiers such as 4.1 or 6.1.3 before deciding which
        # numbers are substantive. Very long flattened list blocks are skipped
        # because appending them verbatim creates duplicate, unreadable answers.
        cleaned_sentence = re.sub(r"(?<!\d)\b\d+(?:\.\d+){1,4}[.)]?\s*", "", sentence)
        cleaned_sentence = re.sub(r"\s+", " ", cleaned_sentence).strip()
        if len(cleaned_sentence) > 430:
            continue
        if not cleaned_sentence[:1].isalnum() or (cleaned_sentence[:1].islower() and not cleaned_sentence.startswith(("if ", "for "))):
            continue
        if not re.search(r"\b\d+(?:\.\d+)?%?\b", cleaned_sentence) or not qualifier_re.search(cleaned_sentence):
            continue
        # Broad multi-topic completeness should preserve core durations and rate
        # phases, not append every procedural start-date or notice deadline.
        if not primary_fact_re.search(cleaned_sentence) and not asks_timing:
            continue
        procedural_timing = bool(re.search(
            r"\b(begin|begins|start|starts|commence|commences|before|after|notice|notification)\b",
            cleaned_sentence,
            re.I,
        ))
        core_phase = bool(re.search(
            r"\b(ordinary|additional|total|first|remaining|maximum|higher|standard|lower|rate|earnings|percentage|percent)\b",
            cleaned_sentence,
            re.I,
        ))
        if procedural_timing and not asks_timing and not core_phase:
            continue
        sentence = cleaned_sentence
        terms = set(_tokens(sentence))
        overlap = len(q_terms & terms) / max(1, len(q_terms))
        qualifier_count = len(set(re.findall(qualifier_re, sentence)))
        score = 2.0 * overlap + 0.18 * qualifier_count
        if len(sentence) <= 340:
            score += 0.12
        scored.append((score, index, sentence))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(scored[:8], key=lambda item: item[1])
    output: list[str] = []
    seen: set[str] = set()
    for _score, _index, sentence in selected:
        sig = _normalize(sentence)[:240]
        if sig not in seen:
            seen.add(sig)
            output.append(sentence)
    return output


def _numeric_fact_covered(answer: str, fact: str) -> bool:
    answer_norm = _normalize(answer)
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", fact)
    qualifiers = [
        term for term in (
            "ordinary", "additional", "total", "first", "remaining", "maximum",
            "higher", "standard", "lower", "weeks", "days", "months", "years", "earnings",
        ) if term in _normalize(fact)
    ]
    if not numbers:
        return True
    for number in numbers:
        if number not in answer_norm:
            return False
    if qualifiers and not any(term in answer_norm for term in qualifiers):
        return False
    return True


def _missing_multi_topic_numeric_facts(answer: str, context: str, question: str) -> list[str]:
    return [
        fact for fact in _context_numeric_fact_sentences(context, question)
        if not _numeric_fact_covered(answer, fact)
    ][:5]


def _multi_topic_question(question: str) -> bool:
    q = _normalize(question)
    content_terms = [term for term in q.split() if term not in STOPWORDS]
    return len(content_terms) >= 5 and bool(
        re.search(r"\b(and|plus|as well as|along with|together with)\b", q)
    )


def _extract_condition_phrases(context: str) -> list[str]:
    value = re.sub(r"\s+", " ", str(context or ""))
    phrases: list[str] = []
    patterns = (
        # Capture only the condition, not the requirement that follows it.
        r"\bif\s+[^,.;:]{3,80}?(?=\bthen\b|[,.;:])",
        r"\bunless\s+[^,.;:]{3,80}",
        r"\bfor\s+(?:an?|the)\s+[A-Za-z0-9'/-]+(?:\s+[A-Za-z0-9'/-]+){0,5}",
        r"\bin the case of\s+[^,.;:]{3,80}",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, value, flags=re.I):
            phrase = re.sub(r"\s+", " ", match.group(0)).strip(" ,;:")
            norm = _normalize(phrase)
            if len(norm.split()) >= 2 and norm not in {_normalize(p) for p in phrases}:
                phrases.append(phrase)
    return phrases[:8]


def _condition_key(phrase: str) -> str:
    value = _normalize(phrase)
    value = re.sub(r"^(?:if|unless|for a|for an|for the|in the case of)\s+", "", value)
    terms = [t for t in value.split() if t not in STOPWORDS and t not in {"then", "must", "shall", "provide", "submit", "required", "requirement"}]
    return " ".join(terms[:6])


def _condition_coverage(answer: str, conditions: list[str]) -> float:
    if not conditions:
        return 1.0
    answer_norm = _normalize(answer)
    covered = 0
    for phrase in conditions:
        key = _condition_key(phrase)
        if key and key in answer_norm:
            covered += 1
    return covered / len(conditions)



_REQUIREMENT_ACTION_WORDS_RE = re.compile(
    r"\b(must|shall|required|requirement|provide|submit|notify|inform|advise|need to|needs to|have to)\b",
    re.I,
)
_REQUIREMENT_COMPONENT_GENERIC = {
    "also", "company", "employee", "employees", "must", "shall", "should",
    "required", "requirement", "provide", "submit", "notify", "inform", "advise",
    "need", "needs", "have", "has", "having", "then", "therefore",
}


def _light_stem(token: str) -> str:
    token = str(token or "").casefold()
    for suffix in ("ingly", "edly", "ation", "ments", "ment", "ing", "ied", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[:-3] + "y" if suffix == "ies" else token[:-len(suffix)]
    return token


def _requirement_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", str(text or "").casefold()):
        if len(raw) <= 1 or raw in STOPWORDS or raw in _REQUIREMENT_COMPONENT_GENERIC:
            continue
        terms.add(_light_stem(raw))
    return terms


def _term_set_coverage(source_terms: set[str], target_terms: set[str]) -> float:
    if not source_terms:
        return 1.0
    covered = 0
    for source in source_terms:
        if source in target_terms:
            covered += 1
            continue
        if any(
            min(len(source), len(target)) >= 5
            and (source.startswith(target[:5]) or target.startswith(source[:5]))
            for target in target_terms
        ):
            covered += 1
    return covered / len(source_terms)


def _answer_requirement_segments(answer: str) -> list[str]:
    value = re.sub(r"\s*[•▪◦‣]\s*", "\n", str(answer or ""))
    parts = re.split(r"\n+|(?<=[.!?])\s+", value)
    return [part.strip(" \t-•") for part in parts if len(part.strip(" \t-•")) >= 8]


def _requirement_clause_units(context: str) -> list[str]:
    value = re.sub(r"\s+", " ", str(context or "")).strip()
    if not value:
        return []
    numbered = [
        re.sub(r"^\d+(?:\.\d+)+\s*", "", part).strip()
        for part in re.split(r"(?=\b\d+(?:\.\d+)+\s+)", value)
        if part.strip()
    ]
    clauses = [part for part in numbered if _REQUIREMENT_ACTION_WORDS_RE.search(part)]
    if clauses:
        return clauses
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", value)
        if _REQUIREMENT_ACTION_WORDS_RE.search(part)
    ]


def _local_answer_for_requirement_clause(clause: str, answer: str) -> str:
    segments = _answer_requirement_segments(answer)
    if not segments:
        return str(answer or "")

    action_match = _REQUIREMENT_ACTION_WORDS_RE.search(clause)
    prefix = clause[: action_match.start()] if action_match else ""
    condition_terms = _requirement_terms(prefix) if re.search(
        r"\b(if|when|unless|in the case of|for an?|for the|under|overseas|domestic|local)\b",
        prefix,
        flags=re.I,
    ) else set()

    if condition_terms:
        ranked = sorted(
            (
                _term_set_coverage(condition_terms, _requirement_terms(segment)),
                _term_set_coverage(_requirement_terms(clause), _requirement_terms(segment)),
                segment,
            )
            for segment in segments
        )
        best_condition = ranked[-1][0] if ranked else 0.0
        selected = [
            segment for condition_score, clause_score, segment in ranked
            if condition_score >= max(0.50, best_condition * 0.80) and clause_score >= 0.12
        ]
        if selected:
            return " ".join(selected[-2:])

    ranked = sorted(
        (
            _term_set_coverage(_requirement_terms(clause), _requirement_terms(segment)),
            segment,
        )
        for segment in segments
    )
    return ranked[-1][1] if ranked else str(answer or "")


def _missing_compound_requirement_parts(answer: str, context: str) -> list[str]:
    """Find omitted items from a source requirement joined by ``and``/``or``.

    The comparison is local to the matching answer sentence. This avoids a word
    appearing in another case from falsely covering a missing item in the
    current case. The logic is corpus-neutral and operates only on the selected
    source section and the generated answer.
    """
    missing: list[str] = []
    seen: set[str] = set()
    for clause in _requirement_clause_units(context):
        # Bullet lists are already handled by the general requirement/list
        # extractors. This repair targets a single prose requirement that a
        # small model may truncate after the first conjunction.
        if any(marker in clause for marker in ("•", "▪", "◦", "‣")):
            continue
        components = [
            part.strip(" ,;:.-")
            for part in re.split(r"\s+\b(?:and|or)\b\s+", clause, flags=re.I)
            if len(part.strip().split()) >= 3
        ]
        if len(components) < 2:
            continue

        local_answer = _local_answer_for_requirement_clause(clause, answer)
        local_terms = _requirement_terms(local_answer)
        component_terms = [_requirement_terms(component) for component in components]
        distinct_coverages: list[float] = []
        component_coverages: list[float] = []
        for index, terms in enumerate(component_terms):
            other_terms: set[str] = set()
            for other_index, other in enumerate(component_terms):
                if other_index != index:
                    other_terms.update(other)
            distinctive = terms - other_terms
            if not distinctive:
                distinctive = terms
            distinct_coverages.append(_term_set_coverage(distinctive, local_terms))
            component_coverages.append(_term_set_coverage(terms, local_terms))

        # The clause must already be represented in the answer. We repair only
        # later joined items, preventing unrelated requirements from being added.
        represented = max(component_coverages or [0.0]) >= 0.50 or max(distinct_coverages or [0.0]) >= 0.60
        if not represented:
            continue
        for index in range(1, len(components)):
            if distinct_coverages[index] >= 0.50:
                continue
            component = re.sub(r"\s+", " ", components[index]).strip(" -;:")
            if not (8 <= len(component) <= 260):
                continue
            signature = _normalize(component)[:220]
            if signature and signature not in seen:
                seen.add(signature)
                missing.append(component)
    return missing[:6]


def _append_missing_compound_requirements(answer: str, context: str) -> str:
    missing = _missing_compound_requirement_parts(answer, context)
    if not missing:
        return answer

    repaired = str(answer or "")
    handled: set[str] = set()
    # Prefer replacing the partial answer sentence with the complete source
    # clause. This is cleaner than repeating the already-present first item and
    # remains fully extractive.
    for clause in _requirement_clause_units(context):
        clause_missing = [part for part in missing if part in clause]
        if not clause_missing or any(marker in clause for marker in ("•", "▪", "◦", "‣")):
            continue
        local = _local_answer_for_requirement_clause(clause, repaired)
        clean_clause = re.sub(r"\s+", " ", clause).strip(" -;:")
        if local and len(local) >= 20 and local in repaired and len(clean_clause) <= 620:
            repaired = repaired.replace(local, clean_clause, 1)
            handled.update(clause_missing)

    remaining = [part for part in missing if part not in handled]
    if not remaining:
        return repaired

    addition = "\n".join(
        f"- {part[:1].upper() + part[1:]}" if re.search(r"[.!?)]$", part)
        else f"- {part[:1].upper() + part[1:]}."
        for part in remaining
    )
    return repaired.rstrip() + "\n\nAdditional required details:\n" + addition


def _extractive_conditional_requirements(context: str, question: str) -> str:
    """Extract only requirement statements that directly match the question."""
    value = re.sub(r"\s+", " ", str(context or "")).strip()
    if not value:
        return NOT_FOUND

    sentences = [
        re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", part.strip(" -•\n\t"))
        for part in re.split(r"(?<=[.!?])\s+|(?=\s+\d+(?:\.\d+)+\s+[A-Z])", value)
    ]
    sentences = [sentence for sentence in sentences if len(sentence) >= 12]
    if not sentences:
        return NOT_FOUND

    requirement_re = re.compile(
        r"\b(must|shall|required|requirement|provide|submit|need to|needs to|have to|eligible|entitled)\b",
        re.I,
    )
    q_terms = set(_tokens(question))
    document_intent = bool(re.search(
        r"\b(document|documents|paperwork|proof|certificate|form|notification|submit|submission)\b",
        _normalize(question),
    ))
    document_re = re.compile(
        r"\b(document|documents|paperwork|proof|certificate|form|notification|provide|submit|attachment|record)\b",
        re.I,
    )

    scored: list[tuple[float, int, str]] = []
    for idx, sentence in enumerate(sentences):
        if not requirement_re.search(sentence):
            continue
        s_terms = set(_tokens(sentence))
        overlap = len(q_terms & s_terms) / max(1, len(q_terms))
        score = 2.2 * overlap
        if document_intent and document_re.search(sentence):
            score += 1.5
        if re.search(r"\b(if|when|unless|for an?|for the|in the case of)\b", sentence, re.I):
            score += 0.45
        if len(sentence) <= 320:
            score += 0.15
        scored.append((score, idx, sentence))

    if not scored:
        return NOT_FOUND

    best_score = max(item[0] for item in scored)
    threshold = max(0.45, best_score * 0.58)
    selected = [(idx, sentence) for score, idx, sentence in scored if score >= threshold]
    selected.sort(key=lambda item: item[0])

    seen: set[str] = set()
    output: list[str] = []
    for _idx, sentence in selected[:6]:
        sig = _normalize(sentence)[:220]
        if sig and sig not in seen:
            seen.add(sig)
            output.append(sentence)

    if not output:
        return NOT_FOUND
    return sanitize_answer("\n".join(f"- {item}" for item in output), question)


def sanitize_answer(answer: str, question: str = "") -> str:
    """Return clean employee-facing text while preserving factual content."""
    value = str(answer or "").strip()
    if not value:
        return NOT_FOUND
    if value.startswith("[ERROR]"):
        return value

    value = _strip_source_metadata(value)
    value = _remove_meta_commentary(value)
    value = _apply_link_policy(value, question)
    value = _format_procedure_layout(value)

    # Remove duplicated whitespace and empty punctuation left by filtering.
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"\n\s*\n\s*\n+", "\n\n", value)
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = value.strip(" \n\t:")

    return value or NOT_FOUND


def _safe_context_cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    floor = max(0, int(limit * 0.55))
    candidates: list[tuple[int, int]] = []
    for marker in ("\n\n", "\n", ". ", "; ", ": "):
        pos = text.rfind(marker, floor, limit)
        if pos >= floor:
            candidates.append((pos, len(marker)))
    if candidates:
        pos, marker_len = max(candidates)
        return text[: pos + marker_len].strip()
    return text[:limit].rsplit(" ", 1)[0].strip()


def _trim_context(chunks: list[str], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    seen: set[str] = set()
    selected: list[str] = []
    total = 0
    for raw in chunks:
        text = str(raw or "").strip()
        sig = _normalize(text)[:500]
        if not text or not sig or sig in seen:
            continue
        seen.add(sig)
        remaining = max_chars - total
        if remaining <= 0:
            break
        text = _safe_context_cut(text, remaining)
        if text:
            selected.append(text)
            total += len(text) + 2
    return "\n\n".join(selected)


def _call_ollama(prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max(16, min(int(max_tokens), MAX_NUM_PREDICT)),
                    "temperature": float(temperature),
                    "top_p": 0.9,
                    "repeat_penalty": 1.05,
                    "num_ctx": OLLAMA_NUM_CTX,
                    "num_thread": OLLAMA_NUM_THREAD,
                },
                "keep_alive": "20m",
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        return str(response.json().get("response", "")).strip()
    except requests.exceptions.ConnectionError:
        return "[ERROR] Cannot connect to Ollama. Is it running? (ollama serve)"
    except requests.exceptions.Timeout:
        return "[ERROR] Ollama timed out while generating the answer."
    except Exception as exc:
        return f"[ERROR] {exc}"


def rewrite_query(question: str) -> list[str]:
    if not ENABLE_QUERY_REWRITE or not question.strip():
        return []
    result = _call_ollama(_REWRITE_PROMPT.format(question=question.strip()), 72, 0.0)
    if not result or result.startswith("[ERROR]"):
        return []
    values: list[str] = []
    for line in result.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if len(line) >= 4 and line.casefold() != question.strip().casefold():
            values.append(line)
    return list(dict.fromkeys(values))[:2]


def choose_faq_candidate(
    question: str,
    candidate_questions: list[str],
    candidate_answers: list[str] | None = None,
) -> int | None:
    """Use the local LLM only to disambiguate close FAQ intents.

    The candidate answer is shown only as a short preview so the model can
    distinguish answer types such as an identifier versus an access link.
    This remains corpus-neutral and contains no document-specific mappings.
    """
    if not ENABLE_FAQ_DISAMBIGUATION or not question.strip() or not candidate_questions:
        return None

    limited_questions = [str(item or "").strip() for item in candidate_questions[:6]]
    limited_answers = list(candidate_answers or [])[: len(limited_questions)]
    rows: list[str] = []
    for idx, candidate_question in enumerate(limited_questions):
        preview = ""
        if idx < len(limited_answers):
            preview = re.sub(r"\s+", " ", str(limited_answers[idx] or "")).strip()[:180]
        row = f"{idx + 1}. Question: {candidate_question}"
        if preview:
            row += f"\n   Answer type preview: {preview}"
        rows.append(row)

    result = _call_ollama(
        _FAQ_SELECT_PROMPT.format(
            question=question.strip(),
            candidates="\n".join(rows),
        ),
        8,
        0.0,
    )
    if not result or result.startswith("[ERROR]"):
        return None
    match = re.search(r"\b([0-6])\b", result)
    if not match:
        return None
    selected = int(match.group(1))
    return selected - 1 if 1 <= selected <= len(limited_questions) else None


def _clean_fallback_candidate(candidate: str, question: str) -> str:
    value = str(candidate or "").strip()
    if not value:
        return ""
    if re.match(r"(?i)^\s*(SOURCE|SECTION|PAGE|LOCATION|EVIDENCE|CONTEXT)\s*:", value):
        return ""
    if re.match(r"(?i)^\s*Document links\s*:", value):
        return ""
    if "Answer:" in value:
        value = value.rsplit("Answer:", 1)[-1].strip()
    value = sanitize_answer(value, question)
    return "" if value == NOT_FOUND else value


def _extractive_fallback_answer(context_chunks: list[str], question: str, max_sentences: int = 4) -> str:
    q = set(_tokens(question))
    if not q:
        return NOT_FOUND
    context = _trim_context(context_chunks)
    candidates = [line.strip() for line in context.splitlines() if len(line.strip()) >= 6]
    candidates += re.split(r"(?<=[.!?])\s+", context)
    scored: list[tuple[float, str]] = []
    for raw in candidates:
        sent = _clean_fallback_candidate(raw, question)
        if not sent:
            continue
        sent_tokens = set(_tokens(sent))
        overlap = len(q & sent_tokens)
        if not overlap:
            continue
        score = overlap / max(1, len(q))
        if "http" in sent or "](" in sent:
            score += 0.15
        if re.match(r"^(?:step\s*\d+|\d+[.)])", sent, re.I):
            score += 0.12
        if len(sent) < 260:
            score += 0.08
        scored.append((score, sent))

    if not scored:
        return NOT_FOUND

    selected: list[str] = []
    seen: set[str] = set()
    for _, sent in sorted(scored, key=lambda item: item[0], reverse=True):
        sig = _normalize(sent)[:180]
        if sig and sig not in seen:
            seen.add(sig)
            selected.append(sent)
        if len(selected) >= max_sentences:
            break
    return sanitize_answer("\n".join(selected), question)


def _numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", str(text or "")))


def _sentence_units(text: str) -> list[str]:
    value = re.sub(r"[ \t]+", " ", str(text or "")).strip()
    units = []
    for line in value.splitlines():
        line = line.strip(" -•\t")
        if not line:
            continue
        units.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", line) if part.strip())
    return units


def _number_context_terms(sentence: str, number: str, radius: int = 9) -> set[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9%]+", str(sentence or ""))
    positions = [index for index, token in enumerate(raw_tokens) if token == number or token.rstrip("%") == number.rstrip("%") ]
    if not positions:
        return set(_tokens(sentence))
    position = positions[0]
    window = " ".join(raw_tokens[max(0, position - radius): position + radius + 1])
    return {token for token in _tokens(window) if not token.isdigit()}


_NUMERIC_SUBJECT_TERMS = {
    "leave", "pay", "payment", "salary", "allowance", "rate", "earnings",
    "notice", "service", "employment", "entitlement", "benefit", "duration",
    "period", "work", "training", "contact", "deadline", "limit", "time",
}


def _numeric_subject_terms(sentence: str, number: str) -> set[str]:
    """Extract likely subject nouns attached to a numeric claim.

    Preceding terms are preferred because English policy sentences usually name
    the subject before phrases such as "for up to 39 weeks". If no subject is
    found before the number, a short following window is used.
    """
    raw_tokens = re.findall(r"[A-Za-z0-9%]+", str(sentence or ""))
    positions = [
        index for index, token in enumerate(raw_tokens)
        if token == number or token.rstrip("%") == number.rstrip("%")
    ]
    if not positions:
        return set()
    position = positions[0]
    before = " ".join(raw_tokens[max(0, position - 12):position])
    after = " ".join(raw_tokens[position + 1:position + 7])
    before_terms = set(_tokens(before)) & _NUMERIC_SUBJECT_TERMS
    if before_terms:
        return before_terms
    return set(_tokens(after)) & _NUMERIC_SUBJECT_TERMS


def _unsupported_numeric_sentences(answer: str, context: str) -> list[str]:
    context_sentences = _sentence_units(context)
    unsupported = []
    for sentence in _sentence_units(answer):
        numbers = _numeric_tokens(sentence)
        if not numbers:
            continue
        answer_terms = set(_tokens(sentence))
        sentence_supported = True
        for number in numbers:
            candidates = [source for source in context_sentences if number in _numeric_tokens(source)]
            if not candidates:
                sentence_supported = False
                break
            answer_local = _number_context_terms(sentence, number)
            answer_subject = _numeric_subject_terms(sentence, number)
            restrictive_terms = {
                "first", "remaining", "additional", "ordinary", "standard",
                "higher", "lower", "maximum", "minimum", "total",
                "paid", "unpaid", "consecutive", "business", "calendar",
            }
            answer_qualifiers = answer_local & restrictive_terms
            best = 0.0
            for source in candidates:
                source_local = _number_context_terms(source, number)
                source_subject = _numeric_subject_terms(source, number)
                if answer_subject and source_subject and not (answer_subject & source_subject):
                    continue
                # A restrictive modifier must stay attached to the same number.
                # For example, a duration labelled "additional" cannot be
                # supported by a source sentence that labels the same number as
                # a pay period.
                if answer_qualifiers and not answer_qualifiers.issubset(source_local):
                    continue
                shared = answer_local & source_local
                denominator = max(1, min(len(answer_local), len(source_local)))
                local_score = len(shared) / denominator
                global_score = len(answer_terms & set(_tokens(source))) / max(1, len(answer_terms))
                best = max(best, .72 * local_score + .28 * global_score)
            minimum_score = .42 if answer_qualifiers else .30
            if best < minimum_score:
                sentence_supported = False
                break
        if not sentence_supported:
            unsupported.append(sentence)
    return unsupported


def _remove_sentences(text: str, rejected: list[str]) -> str:
    value = str(text or "")
    for sentence in rejected:
        value = value.replace(sentence, " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n\s*\n\s*\n+", "\n\n", value)
    return value.strip()


def _looks_truncated_answer(answer: str) -> bool:
    value = str(answer or "").strip()
    if len(value) < 80:
        return False
    plain = re.sub(r"\[[^\]]+\]\([^\)]+\)", "link", value).strip()
    if re.search(r"\b(?:if|when|unless|and|or|but|because|however|which|that|who|whose|where|to|of|for|with|a|an|the)\s*$", plain, re.I):
        return True
    if plain.count("(") > plain.count(")") or plain.count("[") > plain.count("]"):
        return True
    if len(plain) > 420 and not re.search(r"[.!?)]$", plain):
        return True
    last_line = plain.splitlines()[-1].strip() if plain.splitlines() else plain
    if len(plain) > 260 and 0 < len(last_line.split()) <= 4 and not re.search(r"[.!?)]$", last_line):
        return True
    return False


def _trim_incomplete_tail(answer: str) -> str:
    value = str(answer or "").rstrip()
    if not _looks_truncated_answer(value):
        return value
    last_terminal = max(value.rfind("."), value.rfind("!"), value.rfind("?"), value.rfind(")"))
    if last_terminal >= max(40, int(len(value) * 0.45)):
        return value[:last_terminal + 1].rstrip()
    lines = value.splitlines()
    if len(lines) > 1 and len(lines[-1].split()) <= 8:
        return "\n".join(lines[:-1]).rstrip()
    return value


def generate_answer_full(context_chunks: list[str], question: str, max_tokens: int = 320) -> str:
    context = _trim_context(context_chunks)
    if not context:
        return NOT_FOUND

    if _multi_topic_question(question):
        prompt_template = _MULTI_TOPIC_PROMPT
    elif _initial_action_question(question):
        prompt_template = _INITIAL_ACTION_PROMPT
    elif _eligibility_question(question):
        prompt_template = _ELIGIBILITY_PROMPT
    elif _requirements_question(question):
        prompt_template = _REQUIREMENTS_PROMPT
    else:
        prompt_template = _ANSWER_PROMPT
    raw_answer = _call_ollama(prompt_template.format(context=context, question=question), max_tokens, 0.0)
    if not raw_answer or raw_answer.startswith("[ERROR]"):
        return _extractive_fallback_answer(context_chunks, question)

    answer = sanitize_answer(raw_answer, question)
    if NOT_FOUND.casefold() in answer.casefold():
        fallback = _extractive_fallback_answer(context_chunks, question)
        return fallback if fallback != NOT_FOUND else NOT_FOUND

    # Requirement answers often contain mutually exclusive cases. If a small
    # model drops those conditions, run one focused correction pass.
    if _requirements_question(question):
        conditions = _extract_condition_phrases(context)
        if len(conditions) >= 2 and _condition_coverage(answer, conditions) < 0.5:
            repaired_raw = _call_ollama(
                _CONDITION_REPAIR_PROMPT.format(
                    context=context, question=question, answer=answer
                ),
                max_tokens,
                0.0,
            )
            if repaired_raw and not repaired_raw.startswith("[ERROR]"):
                repaired = sanitize_answer(repaired_raw, question)
                if _condition_coverage(repaired, conditions) >= _condition_coverage(answer, conditions):
                    answer = repaired
            if _condition_coverage(answer, conditions) < 0.5:
                extractive = _extractive_conditional_requirements(context, question)
                if extractive != NOT_FOUND:
                    answer = extractive

        # A source requirement may contain several mandatory items joined by
        # "and" or "or". Small local models sometimes preserve the first item
        # but drop a later date, proof, or condition. Restore only the missing
        # joined components from the same matched source clause.
        answer = _append_missing_compound_requirements(answer, context)

    unsupported_numeric = _unsupported_numeric_sentences(answer, context)
    if unsupported_numeric:
        repaired_raw = _call_ollama(
            _NUMERIC_REPAIR_PROMPT.format(
                context=context, question=question, answer=answer
            ),
            max(max_tokens, 320),
            0.0,
        )
        if repaired_raw and not repaired_raw.startswith("[ERROR]"):
            repaired = sanitize_answer(repaired_raw, question)
            if repaired != NOT_FOUND and len(_unsupported_numeric_sentences(repaired, context)) < len(unsupported_numeric):
                answer = repaired
        remaining_unsupported = _unsupported_numeric_sentences(answer, context)
        if remaining_unsupported:
            answer = _remove_sentences(answer, remaining_unsupported) or NOT_FOUND

    if _looks_truncated_answer(answer):
        completed_raw = _call_ollama(
            _COMPLETION_REPAIR_PROMPT.format(
                context=context, question=question, answer=answer
            ),
            MAX_NUM_PREDICT,
            0.0,
        )
        if completed_raw and not completed_raw.startswith("[ERROR]"):
            completed = sanitize_answer(completed_raw, question)
            if completed != NOT_FOUND and not _looks_truncated_answer(completed):
                answer = completed
        if _looks_truncated_answer(answer):
            answer = _trim_incomplete_tail(answer)

    # Small local models sometimes return only the first bullet from an opening
    # phase even when the same deadline contains several parallel actions.
    if _initial_action_question(question):
        phase_label, phase_actions = _best_initial_phase_actions(context_chunks, question)
        if len(phase_actions) >= 2 and _initial_action_coverage(answer, phase_actions) < 0.75:
            answer = _render_initial_actions(phase_label, phase_actions)

    # Broad multi-topic answers must retain source-defined component durations
    # and rate phases. Repair once, then append exact source sentences if needed.
    if _multi_topic_question(question):
        missing_facts = _missing_multi_topic_numeric_facts(answer, context, question)
        if missing_facts:
            repaired_raw = _call_ollama(
                _MULTI_TOPIC_COMPLETENESS_PROMPT.format(
                    facts="\n".join(f"- {fact}" for fact in missing_facts),
                    context=context,
                    question=question,
                    answer=answer,
                ),
                max(MAX_NUM_PREDICT, max_tokens),
                0.0,
            )
            if repaired_raw and not repaired_raw.startswith("[ERROR]"):
                repaired = sanitize_answer(repaired_raw, question)
                if repaired != NOT_FOUND and len(_missing_multi_topic_numeric_facts(repaired, context, question)) < len(missing_facts):
                    answer = repaired
            remaining = _missing_multi_topic_numeric_facts(answer, context, question)
            if remaining:
                clean_remaining = []
                answer_norm = _normalize(answer)
                answer_terms = set(_tokens(answer))
                seen = set()
                seen_terms: list[set[str]] = []
                for fact in remaining:
                    fact = re.sub(r"(?<!\d)\b\d+(?:\.\d+){1,4}[.)]?\s*", "", fact)
                    fact = re.sub(r"\s+", " ", fact).strip(" -;:")
                    sig = _normalize(fact)[:240]
                    fact_terms = set(_tokens(fact))
                    overlap_with_answer = len(fact_terms & answer_terms) / max(1, len(fact_terms))
                    duplicate_fact = any(
                        len(fact_terms & previous) / max(1, len(fact_terms | previous)) >= 0.72
                        for previous in seen_terms
                    )
                    if (
                        not fact or len(fact) > 430 or sig in seen or sig in answer_norm
                        or overlap_with_answer >= 0.82 or duplicate_fact
                    ):
                        continue
                    seen.add(sig)
                    seen_terms.append(fact_terms)
                    clean_remaining.append(fact)
                if clean_remaining:
                    answer = answer.rstrip() + "\n\n### Additional details\n" + "\n".join(
                        f"- {fact}" for fact in clean_remaining[:3]
                    )

    # Completeness/numeric repair can itself produce a new unfinished tail.
    # Apply one final deterministic trim after every repair stage.
    answer = sanitize_answer(answer, question)
    if _looks_truncated_answer(answer):
        answer = _trim_incomplete_tail(answer)
    return sanitize_answer(answer, question)


def answer_support_score(context_chunks: list[str], answer: str) -> float:
    if not answer or answer.startswith("[ERROR]") or NOT_FOUND.casefold() in answer.casefold():
        return 0.0
    context = _normalize(_trim_context(context_chunks, 9000))
    answer_tokens = [token for token in _tokens(answer) if len(token) > 2]
    if not answer_tokens or not context:
        return 0.0
    token_support = sum(1 for token in answer_tokens if token in context) / len(answer_tokens)
    exact = re.findall(r"https?://\S+|[\w.+-]+@[\w.-]+|\b\d+(?:\.\d+)?%?\b", answer)
    exact_support = 1.0
    if exact:
        raw = _trim_context(context_chunks, 10000)
        exact_support = sum(1 for value in exact if value.rstrip(".,;)") in raw) / len(exact)
    return max(0.0, min(1.0, 0.75 * token_support + 0.25 * exact_support))


def verify_answer(context_chunks: list[str], answer: str) -> str:
    if not answer or answer.startswith("[ERROR]"):
        return GENERATION_FAILURE
    if NOT_FOUND.casefold() in answer.casefold():
        return REFUSED_INSUFFICIENT_EVIDENCE
    deterministic = answer_support_score(context_chunks, answer)
    if ENABLE_LLM_VERIFY:
        verdict = _call_ollama(
            _VERIFY_PROMPT.format(context=_trim_context(context_chunks[:6], 4200), answer=answer),
            12,
            0.0,
        ).upper()
        if "ANSWERED_SUPPORTED" in verdict:
            return ANSWERED_SUPPORTED
        if "ANSWERED_PARTIAL" in verdict or "UNSUPPORTED" in verdict:
            return ANSWERED_PARTIAL
    return ANSWERED_SUPPORTED if deterministic >= 0.72 else ANSWERED_PARTIAL


def generate_document_summary(context_chunks: list[str]) -> str:
    if os.getenv("RAG_GENERATE_DOC_SUMMARY", "0") != "1":
        return ""
    return _call_ollama(
        f"Give a 3-bullet summary using only this evidence:\n\n{_trim_context(context_chunks[:4], 3000)}\n\nSummary:",
        120,
        0.0,
    )


def generate_freeform(prompt: str, max_tokens: int = 120, temperature: float = 0.2) -> str:
    if os.getenv("RAG_ALLOW_FREEFORM", "0") != "1":
        return ""
    return _call_ollama(prompt, max_tokens, temperature)

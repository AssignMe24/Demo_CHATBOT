from __future__ import annotations

import base64
import html as html_lib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import fitz
import requests
import streamlit as st
import streamlit.components.v1 as components
import numpy as np  # np is not imported at module level

from document_loader import SUPPORTED_EXTENSIONS
from backend_watcher import BackendFileWatcher
from rag_pipeline import RAGPipeline, RAGResult, ingest_document, load_document_cache
from answer_generation import _extractive_conditional_requirements, apply_source_link_policy, generate_answer_full, sanitize_answer
from confidence import ANSWERED_SUPPORTED, REFUSED_INSUFFICIENT_EVIDENCE, confidence_color, confidence_label
from embedder import embed_chunks
from vector_store import VectorStore

try:
    from rag_pipeline import RetrievedChunk
except Exception:
    RetrievedChunk = None

try:
    from chat_history import ChatHistory
except Exception:
    ChatHistory = None

st.set_page_config(page_title="Elite Multi-Doc RAG", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
:root { --bg: #131314; --surface: #1e1f20; --surface2: #2a2b2e; --border: #33363d; --accent: #a8c7fa; --text: #e3e3e3; --muted: #8e918f; }
html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--text); font-family: 'Inter', sans-serif; }
[data-testid="stHeader"] { background: transparent !important; } #MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
.preview-container { background: var(--surface); padding: 1rem; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.2); height: 70vh; overflow-y: auto; }
.badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; align-items: center; }
.pill { background: var(--surface); border: 1px solid var(--border); padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; display: inline-flex; align-items: center; gap: 4px; }
div[data-testid="stExpander"] { background: transparent !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
.dot-typing { display: flex; gap: 4px; padding: 8px 0; }
.dot-typing .dot { width: 6px; height: 6px; background-color: var(--accent); border-radius: 50%; animation: typing 1.4s infinite ease-in-out both; }
.dot-typing .dot:nth-child(1) { animation-delay: -0.32s; } .dot-typing .dot:nth-child(2) { animation-delay: -0.16s; }
@keyframes typing { 0%, 80%, 100% { transform: scale(0); opacity: 0.3; } 40% { transform: scale(1); opacity: 1; } }
[data-testid="stTextInput"] input { background: var(--surface) !important; border-radius: 20px !important; border: 1px solid var(--border) !important; padding: 14px 24px !important; font-size: 1.05rem !important; color: white !important; box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important; }
[data-testid="stTextInput"] input:focus { border-color: var(--accent) !important; }
button[kind="secondary"] { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; color: var(--accent) !important; padding: 2px 10px !important; min-height: 0 !important; height: 32px !important; width: 100% !important; }
.small-note { color: var(--muted); font-size: 0.78rem; }

/* Production UI cleanup: hide Streamlit deploy/menu/sidebar controls */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeaderActionElements"],

#MainMenu,
footer {
    display: none !important;
    visibility: hidden !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
    height: 0rem !important;
}



/* Let the chatbot use the full width after sidebar removal */
.block-container {
    padding-top: 2.2rem !important;
    max-width: 1180px !important;
}

/* HR Assistant branding */
.hr-brand-wrap {
    max-width: 900px;
    margin: 2.5rem auto 1.25rem auto;
    display: flex;
    align-items: center;
    gap: 14px;
}

.hr-brand-icon {
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: #263e9a;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.22);
    flex-shrink: 0;
}

.hr-brand-icon svg {
    width: 30px;
    height: 30px;
    display: block;
}

.hr-brand-title {
    color: var(--text);
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    line-height: 1.05;
}

.hr-brand-subtitle {
    color: var(--muted);
    font-size: 0.88rem;
    font-weight: 500;
    margin-top: 4px;
    letter-spacing: 0.01em;
}

@media (max-width: 700px) {
    .hr-brand-wrap {
        margin: 1.25rem auto 1rem auto;
        padding: 0 0.5rem;
    }
    .hr-brand-title {
        font-size: 1.25rem;
    }
    .hr-brand-icon {
        width: 46px;
        height: 46px;
    }
}

/* Final production hard-hide for Streamlit toolbar/deploy controls */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeaderActionElements"],
[data-testid="stDeployButton"],
[data-testid="stActionButton"],

button[title="Deploy"],
button[aria-label="Deploy"],
a[title="Deploy"],
a[aria-label="Deploy"],

footer,
#MainMenu {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

[data-testid="stAppViewBlockContainer"] {
    padding-top: 1rem !important;
}


/* Integrated chat-history panel, not Streamlit sidebar */
.hr-app-shell {
    max-width: 1320px;
    margin: 0 auto;
    padding: 0 1rem;
    display: grid;
    grid-template-columns: 280px minmax(0, 1fr);
    gap: 24px;
    align-items: start;
}

.hr-history-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px;
    min-height: 460px;
    max-height: calc(100vh - 130px);
    overflow-y: auto;
    box-shadow: 0 14px 35px rgba(0,0,0,0.24);
    position: sticky;
    top: 24px;
}

.hr-history-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: var(--text);
}

.hr-history-session {
    font-size: 0.74rem;
    color: var(--muted);
    margin-bottom: 14px;
    word-break: break-all;
}

.hr-history-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 12px;
    color: var(--text);
    font-size: 0.82rem;
    line-height: 1.35;
    margin-bottom: 8px;
}

.hr-history-number {
    color: var(--muted);
    margin-right: 6px;
    font-weight: 600;
}

.hr-history-note {
    color: var(--muted);
    font-size: 0.78rem;
    line-height: 1.45;
    margin-top: 12px;
}

.hr-main-panel {
    min-width: 0;
}

.hr-main-panel .hr-brand-wrap {
    margin-left: 0 !important;
    margin-right: 0 !important;
}

/* Keep chat area comfortable inside main panel */
.hr-main-panel [data-testid="stChatMessage"] {
    max-width: 100%;
}



@media (max-width: 900px) {
    .hr-app-shell {
        display: block;
        padding: 0 0.75rem;
    }
    .hr-history-panel {
        position: relative;
        top: 0;
        min-height: unset;
        max-height: 220px;
        margin-bottom: 18px;
    }
}


.hr-export-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 12px;
}

.hr-export-title {
    color: var(--text);
    font-size: 0.88rem;
    font-weight: 600;
    margin-top: 16px;
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)

CACHE_DIR = Path(os.getenv("RAG_CACHE_DIR", SRC_DIR.parent / "cache" / "elite_rag_multidoc_cache")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DB = SRC_DIR / "feedback.db"

SUPPORTED_TYPES = sorted({ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS})

# Production knowledge base folder.
# Admins add/remove files here; users never upload or choose documents in the UI.
DOCUMENTS_DIR = Path(os.getenv("RAG_DOCUMENTS_DIR", SRC_DIR.parent / "documents")).resolve()
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
WATCH_INTERVAL_SECONDS = int(os.getenv("RAG_WATCH_INTERVAL_SECONDS", "90"))
BACKEND_WATCHER = BackendFileWatcher(DOCUMENTS_DIR, SUPPORTED_EXTENSIONS)

DEFAULT_CFG = {
    # Accuracy-first defaults that remain practical on an i5 / 12 GB RAM laptop.
    "candidate_k": 42,
    "final_k": 6,
    "rerank_k": 34,
    "max_tokens": 320,
    "context_window": 0,
    "evidence_threshold": 0.38,
    "faq_evidence_threshold": 0.36,
    "use_hyde": False,
    "use_policy_router": True,
    "use_faq_layer": True,
    "use_offtopic_guard": True,
    "use_verification": True,
}

if "pipelines" not in st.session_state:
    st.session_state.update({
        "pipelines": {},
        "active_doc": None,
        "doc_bytes": {},
        "doc_types": {},
        "global_pipeline": None,
        "global_signature": None,
        "search_scope": "All documents",
        "backend_file_snapshot": {},
        "backend_last_scan": 0.0,
        "backend_index_errors": [],
        "chat_history": [],
        "view_page": 1,
        "search_mode": "hybrid",
        "cfg": DEFAULT_CFG.copy(),
        "pending_query": None,
        "user_input": "",
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8],
        "awaiting_continue": False,
        "chat_closed": False,
        "session_feedback_saved": False,
        "session_feedback_comment": "",
        "auto_scroll_bottom": False,
        "processing_question": None,
        "processing_timestamp": None,
    })

if "history_store" not in st.session_state:
    if ChatHistory:
        try:
            st.session_state.history_store = ChatHistory(db_path=str(SRC_DIR / "chat_history.db"))
        except Exception:
            st.session_state.history_store = None
    else:
        st.session_state.history_store = None

ss = st.session_state

# Backward-compatible keys for existing Streamlit sessions after upgrades.
if "global_pipeline" not in ss:
    ss.global_pipeline = None
if "global_signature" not in ss:
    ss.global_signature = None
if "search_scope" not in ss:
    ss.search_scope = "All documents"
if "backend_file_snapshot" not in ss:
    ss.backend_file_snapshot = {}
if "backend_last_scan" not in ss:
    ss.backend_last_scan = 0.0
if "backend_index_errors" not in ss:
    ss.backend_index_errors = []
if "awaiting_continue" not in ss:
    ss.awaiting_continue = False
if "chat_closed" not in ss:
    ss.chat_closed = False
if "session_feedback_saved" not in ss:
    ss.session_feedback_saved = False
if "session_feedback_comment" not in ss:
    ss.session_feedback_comment = ""
if "auto_scroll_bottom" not in ss:
    ss.auto_scroll_bottom = False
if "processing_question" not in ss:
    ss.processing_question = None
if "processing_timestamp" not in ss:
    ss.processing_timestamp = None

# ─────────────────────────────────────────────
# General utilities
# ─────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how", "i", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "you", "your", "me", "my", "please", "tell", "show", "give", "explain", "about", "document", "policy", "file",
    "been", "am", "im", "ive", "ve", "first", "step", "steps", "per", "company", "kindly", "easily", "simply",
    "can", "could", "should", "would", "do", "does", "did", "need",
}


def normalize_text(text: str, keep_dots: bool = False) -> str:
    text = str(text or "").lower()
    allowed = r"[^a-z0-9\.\s]" if keep_dots else r"[^a-z0-9\s]"
    text = re.sub(allowed, " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return [t for t in normalize_text(text).split() if t and t not in STOPWORDS and len(t) > 1]


def keyword_score(query: str, text: str) -> float:
    q = set(tokens(query))
    if not q:
        return 0.0
    t = set(tokens(text))
    if not t:
        return 0.0
    return len(q & t) / max(1, len(q))


def phrase_score(query: str, text: str) -> float:
    q = normalize_text(query)
    t = normalize_text(text)
    if not q or not t:
        return 0.0
    if q == t:
        return 1.0
    if q in t or t in q:
        return 0.75
    return 0.0


def clean_for_speech(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


LINK_MD_RE = re.compile(r"\[([^\]]{1,120})\]\((https?://[^\s\)]+)\)")
URL_RE = re.compile(r"https?://[^\s\]\)<>\"']+")


def extract_links_from_text(text: str) -> list[tuple[str, str]]:
    """Extract document links from Markdown links, label: URL, and plain URLs."""
    text = str(text or "")
    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    for label, url in LINK_MD_RE.findall(text):
        url = url.strip().rstrip(".,;")
        label = re.sub(r"\s+", " ", label).strip() or "Link"
        if url not in seen:
            seen.add(url)
            links.append((label, url))

    # Handles extracted text such as "Employee Portal: https://..." or "User Guide https://..."
    label_url_pattern = re.compile(r"([A-Za-z][A-Za-z0-9 \-_/]{1,80})[:\s]+(https?://[^\s\]\)<>\"']+)")
    for label, url in label_url_pattern.findall(text):
        url = url.strip().rstrip(".,;")
        label = re.sub(r"\s+", " ", label).strip(" :-") or "Link"
        if url not in seen:
            seen.add(url)
            links.append((label[-80:], url))

    for url in URL_RE.findall(text):
        url = url.strip().rstrip(".,;")
        if url not in seen:
            seen.add(url)
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            links.append((domain or "Link", url))

    return links


def answer_has_link(answer: str) -> bool:
    return bool(LINK_MD_RE.search(answer or "") or URL_RE.search(answer or ""))


def link_relevance_score(question: str, answer: str, label: str, url: str) -> float:
    q = normalize_text(question)
    a = normalize_text(answer)
    target = normalize_text(f"{label} {url}")
    score = 0.0
    for token in tokens(q):
        if token in target:
            score += 1.0
        if token in a:
            score += 0.25
    if any(word in q for word in ("link", "portal", "url", "website", "open", "click")):
        score += 0.75
    return score


def enrich_answer_with_document_links(answer: str, question: str, result: RAGResult) -> str:
    """Apply the source-validated link policy to the final answer."""
    source_text = ""
    if result and result.retrieved:
        source_text = "\n".join(str(item.text or "") for item in result.retrieved)
    return apply_source_link_policy(answer, question, source_text)


def doc_key(file_name: str, existing: dict) -> str:
    stem = Path(file_name).stem
    key = stem
    counter = 2
    while key in existing:
        key = f"{stem}_{counter}"
        counter += 1
    return key


def invalidate_global_pipeline() -> None:
    """Force rebuilding the all-document search index after files change."""
    ss.global_pipeline = None
    ss.global_signature = None


def documents_signature() -> tuple:
    """A small signature that changes whenever the loaded document set changes."""
    return tuple(
        (key, pipe.pdf_name, ss.doc_types.get(key, ""), len(pipe.chunks))
        for key, pipe in sorted(ss.pipelines.items())
    )


def build_global_pipeline() -> Optional[RAGPipeline]:
    """Build one RAGPipeline containing chunks from every uploaded document.

    This is the production mode: users ask one question, and the system decides
    which document(s) contain the answer.
    """
    if not ss.pipelines:
        return None

    signature = documents_signature()
    if ss.global_pipeline is not None and ss.global_signature == signature:
        ss.global_pipeline.cfg = ss.cfg
        return ss.global_pipeline

   

    combined_chunks = []
    vec_blocks = []
    for doc_key_value, pipe in ss.pipelines.items():
        n = pipe.vector_store.index.ntotal
        if n == 0 or len(pipe.chunks) != n:
            # Fallback: only re-embed a doc whose counts don't line up.
            vecs = embed_chunks([c.get("search_text") or c["text"] for c in pipe.chunks]) if pipe.chunks else None
        else:
            vecs = pipe.vector_store.index.reconstruct_n(0, n)
        if vecs is None:
            continue
        vec_blocks.append(vecs)
        for original in pipe.chunks:
            c = dict(original)
            c["document_key"] = doc_key_value
            c["filename"] = c.get("filename") or pipe.pdf_name
            c["source_type"] = c.get("source_type") or ss.doc_types.get(doc_key_value, "document")
            c["chunk_id"] = len(combined_chunks)
            combined_chunks.append(c)

    if not combined_chunks:
        return None

    embeddings = np.vstack(vec_blocks).astype(np.float32)
    vector_store = VectorStore(embeddings)

    global_pipe = RAGPipeline(
        combined_chunks,
        vector_store,
        "All Documents",
        ss.cfg,
        summary=f"Ready to search across {len(ss.pipelines)} document(s) and {len(combined_chunks)} chunks.",
        source_type="multi",
    )
    ss.global_pipeline = global_pipe
    ss.global_signature = signature
    return global_pipe


def get_answer_pipeline() -> Optional[RAGPipeline]:
    """Return the pipeline used for answering the user question."""
    if ss.search_scope == "All documents":
        return build_global_pipeline()
    if ss.active_doc and ss.active_doc in ss.pipelines:
        return ss.pipelines[ss.active_doc]
    return build_global_pipeline()


def document_key_from_filename(filename: str) -> Optional[str]:
    """Map a retrieved source filename back to the document key used by the UI."""
    if not filename:
        return None
    for key, pipe in ss.pipelines.items():
        if pipe.pdf_name == filename:
            return key
    # Fallback: compare stems, because filenames may be normalized by loaders.
    target_stem = Path(filename).stem.lower()
    for key, pipe in ss.pipelines.items():
        if Path(pipe.pdf_name).stem.lower() == target_stem:
            return key
    return None


def focus_preview_from_result(result: RAGResult) -> None:
    """After all-document search, switch the left preview to the most relevant source."""
    if not result or not result.retrieved:
        return
    first = result.retrieved[0]
    doc_key_value = document_key_from_filename(first.filename)
    if doc_key_value and doc_key_value in ss.pipelines:
        ss.active_doc = doc_key_value
        ss.view_page = first.page


def add_uploaded_doc(uploaded) -> None:
    file_name = uploaded.name
    ext = Path(file_name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        st.error(f"Unsupported file type: {ext}")
        return

    file_bytes = uploaded.getvalue()
    key = doc_key(file_name, ss.pipelines)

    cached = load_document_cache(file_bytes, file_name, CACHE_DIR)
    if cached:
        chunks, vs = cached
        summary = "Document loaded from cache."
    else:
        chunks, vs, summary = ingest_document(file_bytes, file_name, CACHE_DIR, ss.cfg)

    source_type = ext.lstrip(".")
    ss.doc_bytes[key] = file_bytes
    ss.doc_types[key] = source_type
    ss.pipelines[key] = RAGPipeline(chunks, vs, file_name, ss.cfg, summary, source_type=source_type)
    ss.active_doc = key
    ss.view_page = 1
    invalidate_global_pipeline()




def backend_doc_key(relative_path: str) -> str:
    """Stable internal key for backend-managed documents."""
    safe = relative_path.replace("\\", "/").strip("/")
    return "backend::" + safe


def index_backend_file(file_path: Path, relative_path: str, signature: tuple) -> None:
    """Index or re-index one backend document file."""
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return

    display_name = relative_path.replace("\\", "/")
    key = backend_doc_key(display_name)
    file_bytes = file_path.read_bytes()

    cached = load_document_cache(file_bytes, display_name, CACHE_DIR)
    if cached:
        chunks, vs = cached
        summary = "Document loaded from backend cache."
    else:
        chunks, vs, summary = ingest_document(file_bytes, display_name, CACHE_DIR, ss.cfg)

    source_type = ext.lstrip(".")
    ss.doc_bytes[key] = file_bytes
    ss.doc_types[key] = source_type
    ss.pipelines[key] = RAGPipeline(chunks, vs, display_name, ss.cfg, summary, source_type=source_type)
    ss.backend_file_snapshot[display_name] = signature

    if ss.active_doc is None:
        ss.active_doc = key
        ss.view_page = 1


def remove_backend_file(relative_path: str) -> None:
    """Remove a backend-managed file from the in-session index when deleted from disk."""
    display_name = relative_path.replace("\\", "/")
    key = backend_doc_key(display_name)

    ss.pipelines.pop(key, None)
    ss.doc_bytes.pop(key, None)
    ss.doc_types.pop(key, None)
    ss.backend_file_snapshot.pop(display_name, None)

    if ss.active_doc == key:
        ss.active_doc = next(iter(ss.pipelines.keys()), None)
        ss.view_page = 1


def sync_backend_documents(force: bool = False) -> None:
    """Scan backend documents folder and keep the app index synchronized.

    This is the Streamlit equivalent of the old backend watcher:
    - newly added files are indexed automatically
    - modified files are re-indexed automatically
    - deleted files are removed from the searchable index
    """
    now = time.time()
    if not force and (now - float(ss.backend_last_scan or 0)) < WATCH_INTERVAL_SECONDS:
        return

    ss.backend_last_scan = now
    current, added_or_modified, removed = BACKEND_WATCHER.scan(ss.backend_file_snapshot)
    changed = False
    errors: list[str] = []

    for relative_path in removed:
        remove_backend_file(relative_path)
        changed = True

    for relative_path, meta in added_or_modified.items():
        try:
            index_backend_file(Path(meta["path"]), relative_path, meta["signature"])
            changed = True
        except Exception as exc:
            errors.append(f"{relative_path}: {exc}")

    # Store only existing-current files in the snapshot.
    for relative_path, meta in current.items():
        if relative_path not in ss.backend_file_snapshot:
            ss.backend_file_snapshot[relative_path] = meta["signature"]

    ss.backend_index_errors = errors[-10:]
    if changed:
        invalidate_global_pipeline()


def render_pdf_page(pdf_bytes: bytes, page_num: int, highlight_text: str = "") -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_num = max(1, min(page_num, len(doc)))
    page = doc[page_num - 1]

    if highlight_text:
        for phrase in highlight_text.split("."):
            phrase = phrase.strip()
            if len(phrase) > 20:
                for rect in page.search_for(phrase[:80]):
                    page.add_highlight_annot(rect)

    img_bytes = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False).tobytes("png")
    doc.close()
    return base64.b64encode(img_bytes).decode()


def render_text_preview(pipeline: RAGPipeline, location_no: int, highlight_text: str = "") -> str:
    selected = [c for c in pipeline.chunks if c["pages"][0] == location_no]
    if not selected:
        selected = pipeline.chunks[:5]

    html_blocks = []
    for c in selected[:8]:
        label = (c.get("labels") or [f"Location {c['pages'][0]}"])[0]
        text = html_lib.escape(c["text"])
        if highlight_text:
            short = html_lib.escape(highlight_text[:80].strip())
            if short:
                text = text.replace(short, f"<mark>{short}</mark>")
        html_blocks.append(
            f"<div style='margin-bottom:16px;'>"
            f"<div style='color:var(--accent); font-size:0.8rem;'>📍 {html_lib.escape(label)}</div>"
            f"<div style='white-space:pre-wrap; line-height:1.55;'>{text}</div>"
            f"</div>"
        )
    return "".join(html_blocks)


def build_badges(result: RAGResult) -> str:
    """User-facing source badge. Technical confidence/mode values are kept in backend/export only."""
    ref_count = len(result.retrieved or [])
    if ref_count:
        source_text = f"{ref_count} source reference" + ("s" if ref_count != 1 else "")
        return (
            '<div class="badge-row">'
            '<span class="pill">✅ Source-grounded answer</span>'
            f'<span class="pill">📄 {source_text}</span>'
            '</div>'
        )
    return '<div class="badge-row"><span class="pill">⚠️ No direct source reference found</span></div>'


def submit_query():
    if ss.user_input.strip():
        ss.pending_query = ss.user_input.strip()
        ss.user_input = ""

# ─────────────────────────────────────────────
# Deterministic policy/HR features from old Flask app
# ─────────────────────────────────────────────


def make_retrieved_from_chunk(c: dict, vector_score: float = 1.0, rerank_score: float = 1.0):
    label = (c.get("labels") or [f"Location {c.get('pages', [1])[0]}"])[0]
    if RetrievedChunk is None:
        return None
    return RetrievedChunk(
        chunk_id=c.get("chunk_id", 0),
        text=c.get("text", ""),
        page=(c.get("pages") or [1])[0],
        section=c.get("section"),
        vector_score=float(vector_score),
        rerank_score=float(rerank_score),
        label=label,
        source_type=c.get("source_type", "document"),
        filename=c.get("filename", ""),
    )


def manual_result(answer: str, pipeline: RAGPipeline, mode: str, chunks: list[dict] | None = None, score: float = 1.0) -> RAGResult:
    retrieved = []
    for c in (chunks or [])[:8]:
        rc = make_retrieved_from_chunk(c, vector_score=score, rerank_score=score)
        if rc is not None:
            retrieved.append(rc)
    return RAGResult(
        answer=answer,
        verification=ANSWERED_SUPPORTED,
        retrieved=retrieved,
        confidence_score=score,
        confidence_label=confidence_label(score),
        confidence_color=confidence_color(score),
        search_mode=mode,
        elapsed_s=0.0,
        pdf_name=pipeline.pdf_name,
    )


def is_offtopic_or_smalltalk(question: str) -> bool:
    q = normalize_text(question)
    if not q:
        return True
    smalltalk_exact = {
        "hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening", "good night",
        "how are you", "who are you", "what can you do", "help", "test",
    }
    if q in smalltalk_exact:
        return True

    # Do not whitelist any policy, product, person, department, or document
    # topic here. A non-empty content-bearing question proceeds to retrieval;
    # the evidence gate decides whether the indexed corpus can answer it.
    return len(tokens(question)) == 0


def wants_full_document(question: str) -> bool:
    q = normalize_text(question)
    patterns = [
        "full document", "entire document", "whole document", "complete document", "all content",
        "show complete", "show all", "full policy", "entire policy", "complete policy",
        "read the document", "give me the document",
    ]
    return any(p in q for p in patterns)


def wants_full_section(question: str) -> bool:
    q = normalize_text(question)
    return bool(re.search(r"\b(full|entire|complete|whole|all)\s+(section|chapter|part|clause|topic)\b", q))


def wants_exhaustive_topic(question: str) -> bool:
    q = normalize_text(question)
    return bool(re.search(
        r"\b(all|every|complete|full|entire|whole)\b", q
    )) and not wants_full_document(question)


def question_answer_type(question: str) -> str:
    q = normalize_text(question)
    patterns = (
        ("identifier", r"\b(username|user name|user id|login id|account id|employee id|identifier|code)\b"),
        ("consequence", r"\b(what happens|consequence|missed|miss|late|too late|forgot|forget|failed|fail)\b"),
        ("deadline", r"\b(deadline|due date|by when|how often|frequency|weekly|monthly|every week|month end)\b"),
        ("eligibility", r"\b(can i|may i|eligible|eligibility|entitled|entitlement|entitlements|rights?|benefits?|qualify|qualification|circumstances|under what conditions|allowed|permitted|withdraw|withdrawal|before retirement)\b"),
        ("purpose", r"\b(purpose|scope|objective|overview|meaning)\b"),
        ("contact", r"\b(who|whom|contact|call|email|reach out|connect)\b"),
        ("procedure", r"\b(how|steps?|procedure|process|update|correct|edit|change|submit|apply)\b"),
    )
    for label, pattern in patterns:
        if re.search(pattern, q):
            return label
    return "information"


def heading_answer_type(section: str) -> str:
    value = normalize_text(section)
    patterns = (
        # Procedural headings must be classified before the generic word
        # "requirements"; otherwise "Notification Requirements" is mistaken
        # for an eligibility section.
        ("procedure", r"\b(notification|application|procedure|process|steps|how to)\b"),
        ("eligibility", r"\b(eligibility|eligible|entitlement|entitlements|qualification|qualifying|criteria|admission requirements|entry requirements)\b"),
        ("purpose", r"\b(purpose|scope|objective|overview)\b"),
        ("contact", r"\b(contact|contacts|communication|who to contact)\b"),
    )
    for label, pattern in patterns:
        if re.search(pattern, value):
            return label
    return "information"



_HEADING_GENERIC_TERMS = {
    "eligible", "eligibility", "entitled", "entitlement", "entitlements", "right", "rights", "benefit", "benefits",
    "qualify", "qualification", "qualifying", "criteria", "requirements",
    "allowed", "permitted", "circumstances", "purpose", "scope", "objective",
    "overview", "meaning", "contact", "procedure", "process", "steps",
    "when", "employee", "employees", "staff", "company", "policy",
    "use", "using", "before", "after", "first", "immediately",
}


def heading_topic_overlap(question: str, chunks: list[dict]) -> float:
    """Measure whether a deterministic heading belongs to the user's topic.

    Answer-shape words such as eligibility, contact, and procedure are removed.
    The remaining topic anchors must occur in the selected filename, heading, or
    opening source text. This prevents a generic eligibility heading in one
    document from intercepting a question about an unrelated subject.
    """
    q_terms = {term for term in tokens(question) if term not in _HEADING_GENERIC_TERMS}
    if not q_terms:
        return 0.0
    target = " ".join(
        " ".join(filter(None, [
            str(chunk.get("filename") or ""),
            str(chunk.get("section_title") or chunk.get("section") or ""),
            str(chunk.get("text") or "")[:900],
        ]))
        for chunk in chunks[:4]
    )
    target_terms = set(tokens(target))
    return len(q_terms & target_terms) / len(q_terms)


def _mentioned_loaded_filenames(question: str, pipeline: RAGPipeline | None) -> set[str]:
    """Return corpus-derived filenames explicitly named by the user."""
    if pipeline is None:
        return set()
    q_terms = set(tokens(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(question or ""))))
    if not q_terms:
        return set()
    generic = {
        "policy", "policies", "document", "documents", "guideline", "guidelines",
        "final", "draft", "version", "issue", "copy", "pdf", "docx", "txt", "file",
    }
    filenames = {
        str(chunk.get("filename") or "")
        for chunk in getattr(pipeline, "chunks", [])
        if str(chunk.get("filename") or "")
    }
    matched: set[str] = set()
    for filename in filenames:
        stem = Path(filename).stem
        stem = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
        name_terms = {term for term in tokens(stem) if term not in generic and not term.isdigit()}
        if not name_terms:
            continue
        overlap = q_terms & name_terms
        required = 1 if len(name_terms) == 1 else min(2, len(name_terms))
        if len(overlap) >= required:
            matched.add(filename)
    return matched


def _question_mentions_loaded_document(question: str, pipeline: RAGPipeline | None) -> bool:
    """Detect an explicit document reference from currently indexed filenames."""
    return bool(_mentioned_loaded_filenames(question, pipeline))


def _short_query_matches_policy_heading(question: str, pipeline: RAGPipeline | None) -> bool:
    """Return True when a short title-like query clearly names a policy section.

    Short resource names can also be FAQ topics, so they should not be blocked
    from the FAQ layer unless they actually match a substantive non-FAQ heading
    in the indexed corpus.
    """
    if pipeline is None:
        return False
    q_norm = normalize_text(question)
    if not q_norm:
        return False
    for chunk in getattr(pipeline, "chunks", []):
        if chunk.get("record_type") == "faq" or chunk.get("content_role") in {
            "faq", "toc", "front_matter", "title_page", "links"
        }:
            continue
        heading = str(chunk.get("section_title") or chunk.get("section") or "").strip()
        if not heading:
            continue
        h_norm = normalize_text(heading)
        if not h_norm:
            continue
        if q_norm == h_norm or q_norm in h_norm or h_norm in q_norm:
            return True
        if keyword_score(question, heading) >= .60 and keyword_score(heading, question) >= .80:
            return True
    return False


def should_try_faq_first(question: str, pipeline: RAGPipeline | None = None) -> bool:
    """Use the atomic FAQ layer before broad heading routing when appropriate."""
    q = normalize_text(question)
    if not q or detect_clause(question) or detect_section_number(question):
        return False
    if wants_full_document(question) or wants_full_section(question) or wants_exhaustive_topic(question):
        return False
    if _question_mentions_loaded_document(question, pipeline):
        return False
    # Purpose/scope/overview questions describe a document section. Trying the
    # FAQ layer first can match an unrelated FAQ merely because both contain
    # generic words such as "policy". Route these directly to section retrieval.
    if question_answer_type(question) == "purpose":
        return False
    if re.search(r"\b(section|chapter|clause|article|full policy|entire policy)\b", q):
        return False
    # Short title-like inputs may be either a policy section or a named FAQ
    # resource. Skip FAQ routing only when the text actually matches a
    # substantive policy heading in the loaded corpus.
    starts_as_question = bool(re.match(
        r"^(what|how|why|where|when|who|whom|which|can|could|should|would|is|are|do|does|did|i|we|my|our|under)\b",
        q,
    ))
    if not starts_as_question and len(tokens(question)) <= 6:
        return not _short_query_matches_policy_heading(question, pipeline)
    return True


def detect_section_number(question: str) -> Optional[str]:
    q = normalize_text(question, keep_dots=True)
    patterns = [
        r"\bsection\s+([0-9]+(?:\.[0-9]+)*)\b",
        r"\bchapter\s+([0-9]+(?:\.[0-9]+)*)\b",
        r"\bpart\s+([0-9]+(?:\.[0-9]+)*)\b",
        r"\btopic\s+([0-9]+(?:\.[0-9]+)*)\b",
        r"\bsec\.?\s+([0-9]+(?:\.[0-9]+)*)\b",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            return m.group(1)
    return None


def detect_clause(question: str) -> Optional[str]:
    q = normalize_text(question, keep_dots=True)
    m = re.search(r"\b([0-9]+(?:\.[0-9]+)+)\b", q)
    if not m:
        return None
    explicit = bool(re.search(r"\b(clause|subclause|subsection|sub section|paragraph|para|point|condition|item)\b", q))
    if explicit or len(tokens(question)) <= 6:
        return m.group(1)
    return None


def section_heading_matches(section: str, sec_no: str) -> bool:
    s = normalize_text(section or "", keep_dots=True)
    if not s:
        return False
    return bool(
        re.match(rf"^{re.escape(sec_no)}(?:\s|\.|\))", s)
        or re.search(rf"\bsection\s+{re.escape(sec_no)}\b", s)
        or re.search(rf"\bclause\s+{re.escape(sec_no)}\b", s)
    )


def find_section_chunks_by_number(pipeline: RAGPipeline, sec_no: str) -> list[dict]:
    matches = []
    for c in pipeline.chunks:
        if c.get("record_type") == "faq" or c.get("content_role") == "faq":
            continue
        sec = c.get("section") or ""
        text = c.get("text") or ""
        if section_heading_matches(sec, sec_no):
            matches.append(c)
            continue
        first_220 = normalize_text(text[:220], keep_dots=True)
        if re.match(rf"^{re.escape(sec_no)}(?:\s|\.|\))", first_220) or re.search(rf"\bsection\s+{re.escape(sec_no)}\b", first_220):
            matches.append(c)
    if not matches and "." in sec_no:
        top = sec_no.split(".", 1)[0]
        return find_section_chunks_by_number(pipeline, top)
    return sorted(matches, key=lambda c: c.get("chunk_id", 0))


def find_section_chunks_by_topic(pipeline: RAGPipeline, question: str) -> list[dict]:
    """Find a substantive section without letting the document title win.

    Terms that identify the document are removed from the section-intent query
    on a per-file basis. For example, document-title terms route the file while
    words such as "purpose" or "scope" select the substantive section.
    This is corpus-neutral and works for future documents as well.
    """
    question_terms = tokens(question)
    ranked = []

    for chunk in pipeline.chunks:
        role = chunk.get("content_role")
        if chunk.get("record_type") == "faq" or role == "faq":
            continue
        if role in {"toc", "front_matter", "title_page", "links"}:
            continue

        filename = str(chunk.get("filename") or "")
        filename_stem = Path(filename).stem
        filename_stem = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", filename_stem)
        filename_terms = set(tokens(filename_stem))
        residual_terms = [term for term in question_terms if term not in filename_terms]
        residual_query = " ".join(residual_terms or question_terms)

        section = chunk.get("section_title") or chunk.get("section") or ""
        section_number = str(chunk.get("section_number") or "")
        text = chunk.get("text") or ""

        section_score = (
            2.8 * keyword_score(residual_query, section)
            + 1.1 * phrase_score(residual_query, section)
            + 0.65 * keyword_score(residual_query, text[:1800])
        )
        requested_type = question_answer_type(question)
        section_type = heading_answer_type(section)
        if requested_type != "information" and requested_type == section_type:
            section_score += 3.2
        elif requested_type == "eligibility" and re.search(
            r"\b(entitlement|entitlements|eligibility|qualification|qualifying|criteria)\b",
            normalize_text(section),
        ):
            section_score += 3.8
        document_score = 0.55 * keyword_score(question, filename)

        question_dotted = normalize_text(question, keep_dots=True)
        if section_number and re.search(rf"\b{re.escape(section_number)}\b", question_dotted):
            section_score += 2.0
        if normalize_text(section) and normalize_text(section) in normalize_text(question):
            section_score += 1.4

        score = section_score + document_score
        if score > 0:
            ranked.append((score, section_score, chunk))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if not ranked or ranked[0][1] < 0.34:
        return []

    best = ranked[0][2]
    filename = best.get("filename")
    parent_id = best.get("parent_id")
    if parent_id:
        same = [
            chunk for chunk in pipeline.chunks
            if chunk.get("filename") == filename
            and chunk.get("parent_id") == parent_id
            and chunk.get("content_role") not in {"links", "title_page", "toc", "front_matter"}
        ]
        return sorted(same, key=lambda chunk: chunk.get("chunk_id", 0)) or [best]

    section = best.get("section_title") or best.get("section")
    same = [
        chunk for chunk in pipeline.chunks
        if chunk.get("filename") == filename
        and (chunk.get("section_title") or chunk.get("section")) == section
        and chunk.get("content_role") not in {"links", "title_page", "toc", "front_matter"}
    ]
    return sorted(same, key=lambda chunk: chunk.get("chunk_id", 0)) or [best]

def extract_clause_text_from_chunk(text: str, clause_no: str) -> str:
    if not text:
        return ""
    pat = re.compile(
        rf"(?is)(\b{re.escape(clause_no)}\b[\s\).:-]*.*?)(?=\b\d+(?:\.\d+)+\b[\s\).:-]|\n[A-Z][A-Z\s]{{4,}}|$)"
    )
    m = pat.search(text)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def clause_answer(question: str, pipeline: RAGPipeline) -> Optional[RAGResult]:
    clause_no = detect_clause(question)
    if not clause_no:
        return None
    candidates = []
    for c in pipeline.chunks:
        if c.get("record_type") == "faq" or c.get("content_role") == "faq":
            continue
        text = c.get("text", "")
        sec = c.get("section", "")
        extracted = extract_clause_text_from_chunk(text, clause_no)
        if extracted:
            score = 1.0 + keyword_score(question, extracted) + 0.25 * keyword_score(question, sec)
            c2 = dict(c)
            c2["text"] = extracted
            candidates.append((score, c2))
        elif re.search(rf"\b{re.escape(clause_no)}\b", text):
            score = 0.75 + keyword_score(question, text)
            candidates.append((score, c))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for _s, c in candidates[:4]]
    answer = "\n\n".join(c["text"] for c in top_chunks)
    return manual_result(answer, pipeline, mode="clause", chunks=top_chunks, score=0.97)


def section_answer(question: str, pipeline: RAGPipeline) -> Optional[RAGResult]:
    sec_no = detect_section_number(question)
    chunks = []
    mode = "section"
    topic_mode = False
    if sec_no:
        chunks = find_section_chunks_by_number(pipeline, sec_no)
    elif re.search(r"\b(section|chapter|topic)\b", normalize_text(question)):
        chunks = find_section_chunks_by_topic(pipeline, question)
        mode = "section-topic"
        topic_mode = True

    if not chunks:
        return None

    max_chunks = 18 if wants_full_section(question) else 6
    selected = chunks[:max_chunks]
    title = selected[0].get("section") or (f"Section {sec_no}" if sec_no else "Requested section")

    # Topic-style section questions get a generated answer, not a raw dump.
    if topic_mode and not wants_full_section(question):
        seen: set[str] = set()
        texts = []
        for c in selected:
            t = (c.get("text") or "").strip()
            if t and t not in seen:
                seen.add(t)
                texts.append(t)
        if texts:
            answer = generate_answer_full(texts, question, ss.cfg.get("max_tokens", 512))
            if not (answer.startswith("[ERROR]") or "Answer not found" in answer):
                res = manual_result(answer, pipeline, mode=mode, chunks=selected, score=0.85)
                res.answer = enrich_answer_with_document_links(answer, question, res)
                return res

    body = "\n\n".join(c.get("text", "") for c in selected)
    answer = f"{title}\n\n{body}"
    if len(chunks) > max_chunks:
        answer += f"\n\n[Note: Showing the first {max_chunks} matching chunks from this section. Use a more specific question for a shorter answer.]"
    return manual_result(answer, pipeline, mode=mode, chunks=selected, score=0.94)


def full_document_answer(question: str, pipeline: RAGPipeline) -> Optional[RAGResult]:
    if not wants_full_document(question):
        return None

    if getattr(pipeline, "source_type", "") == "multi" and len(ss.pipelines) > 1:
        q_norm = normalize_text(question)
        matched_key = None
        for key, pipe in ss.pipelines.items():
            name = normalize_text(Path(pipe.pdf_name).stem)
            if name and (name in q_norm or any(part for part in name.split() if len(part) > 3 and part in q_norm)):
                matched_key = key
                break
        if matched_key:
            pipeline = ss.pipelines[matched_key]
        else:
            names = ", ".join(pipe.pdf_name for pipe in ss.pipelines.values())
            return manual_result(
                "Please specify which document you want the full content from. Available documents: " + names,
                pipeline,
                mode="full-document-clarify",
                chunks=[],
                score=0.9,
            )

    chunks = sorted(pipeline.chunks, key=lambda c: c.get("chunk_id", 0))
    parts = []
    seen_parents = set()
    seen_text = set()
    for c in chunks:
        parent_key = (c.get("filename", ""), c.get("parent_id", ""))
        if c.get("parent_id") and parent_key in seen_parents:
            continue
        if c.get("parent_id"):
            seen_parents.add(parent_key)
        txt = re.sub(r"\s+", " ", c.get("parent_text") or c.get("text", "")).strip()
        sig = normalize_text(txt)[:300]
        if txt and sig not in seen_text:
            seen_text.add(sig)
            parts.append(txt)
    answer = f"Full content from {pipeline.pdf_name}:\n\n" + "\n\n".join(parts)
    if len(answer) > 60000:
        answer = answer[:60000] + "\n\n[Truncated for display. Export or ask for a specific section for the remaining content.]"
    return manual_result(answer, pipeline, mode="full-document", chunks=chunks[:8], score=0.92)


def _clean_faq_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"^[A\.\:\-\)\s]+", "", text).strip()
    return text


def _clean_faq_answer(text: str) -> str:
    """Remove leaked next FAQ/question content from an extracted FAQ answer."""
    text = _clean_faq_text(text)

    # If the regex captured the next numbered FAQ, cut it off.
    text = re.split(r"(?is)\s+\d{1,3}[\.\)]\s+[^?]{5,260}\?\s+", text, maxsplit=1)[0].strip()

    # If paragraph/list numbering was lost, cut at the next unnumbered FAQ-style
    # question. This handles normal questions and scenario-prefixed questions
    # such as "I worked on a weekend. How should I log it?" without hardcoding
    # any policy topic.
    faq_starters = r"(?:How|What|Where|When|Whom|Who|Why|Which|Can|Could|Should|Would|Will|Is|Are|Do|Does|Did)"
    scenario_prefix = r"(?:(?:I|We|My|Our|An employee|The employee|Employees|Staff|A staff member)\b[^?]{0,120}?\.\s+)?"
    text = re.split(rf"(?s)\s+{scenario_prefix}{faq_starters}\b.{{3,200}}\?\s+", text, maxsplit=1)[0].strip()

    # Also cut off common FAQ heading leakage.
    text = re.split(r"(?is)\s+FAQs?\s*$", text, maxsplit=1)[0].strip()

    # Remove exact repeated sentences caused by overlapped chunks.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    deduped = []
    seen = set()
    for part in parts:
        sig = normalize_text(part)[:180]
        if sig and sig in seen:
            continue
        seen.add(sig)
        deduped.append(part)
    if deduped:
        text = " ".join(deduped).strip()

    return text


def _make_faq_chunk(base_chunk: dict, question: str, answer: str) -> dict:
    """Build a precise source chunk for the FAQ answer.

    This prevents the answer router from attaching links or text from the next
    FAQ item in the same document chunk.
    """
    c = dict(base_chunk or {})
    c["text"] = f"{question}\n\n{answer}".strip()
    if not c.get("pages"):
        c["pages"] = [1]
    if not c.get("labels"):
        c["labels"] = [f"Location {c.get('pages', [1])[0]}"]
    return c


def _faq_question_similarity(user_question: str, faq_question: str, faq_answer: str = "") -> float:
    """Stable FAQ scoring that prefers the FAQ question but can handle scenario wording."""
    try:
        from difflib import SequenceMatcher
        fuzzy = SequenceMatcher(None, normalize_text(user_question), normalize_text(faq_question)).ratio()
        combined_fuzzy = SequenceMatcher(None, normalize_text(user_question), normalize_text(f"{faq_question} {faq_answer}")).ratio()
    except Exception:
        fuzzy = 0.0
        combined_fuzzy = 0.0

    question_overlap = keyword_score(user_question, faq_question)
    answer_overlap = keyword_score(user_question, faq_answer)

    score = 0.0
    score += 2.2 * question_overlap
    score += 1.2 * phrase_score(user_question, faq_question)
    score += 1.8 * fuzzy

    # Exact or near-exact FAQ question should always win over answer-text matches.
    if normalize_text(user_question) == normalize_text(faq_question):
        score += 5.0

    # Answer text is secondary evidence, but it must be strong enough to handle
    # real-life scenario wording where the question says "I am admitted to the
    # hospital" and the FAQ answer says "within 24 hours of hospital admission".
    score += 1.0 * answer_overlap
    score += 0.25 * phrase_score(user_question, faq_answer)
    score += 0.5 * combined_fuzzy
    if answer_overlap >= 0.55:
        score += 1.0

    q_norm = normalize_text(user_question)
    if any(word in q_norm for word in ("link", "portal", "url", "website", "open", "click", "guide")):
        if extract_links_from_text(faq_answer) or re.search(r"\b(portal|website|link|guide|app|page|form|calendar|recording|list)\b", normalize_text(faq_answer)):
            score += 1.25

    return score


def _extract_qa_label_answer_pairs_from_chunk(c: dict) -> list[dict]:
    """Extract FAQ pairs from Q:/A: style text."""
    pairs = []
    text = c.get("text", "")
    pattern = re.compile(
        r"(?is)(?:^|\n|\s)(?:Q|Question)\s*[:\-\)]\s*(.*?)(?:\s|\n)+(?:A|Answer)\s*[:\-\)]\s*(.*?)(?=(?:\s|\n)(?:Q|Question)\s*[:\-\)]|$)"
    )
    for m in pattern.finditer(text):
        q = _clean_faq_text(m.group(1))
        a = _clean_faq_answer(m.group(2))
        if len(q) >= 6 and len(a) >= 3:
            pairs.append({"question": q, "answer": a, "chunk": _make_faq_chunk(c, q, a)})
    return pairs


def _extract_numbered_faq_pairs_from_text(text: str, base_chunk: dict) -> list[dict]:
    """Extract numbered FAQ pairs from a larger text block.

    Supports:
      11. How to access ...?
      Please refer [Example Application Guide](...)

      12. Whom to contact ...?
      Contact Name – ...
    """
    pairs = []
    text = re.sub(r"\s+", " ", str(text or "")).strip()

    numbered_pattern = re.compile(
        r"(?is)(?:^|\s)(\d{1,3})[\.\)]\s+(.{6,300}?\?)\s*(.*?)(?=\s+\d{1,3}[\.\)]\s+.{6,300}?\?\s*|$)"
    )

    for m in numbered_pattern.finditer(text):
        question = _clean_faq_text(m.group(2))
        answer = _clean_faq_answer(m.group(3))

        if len(question) >= 6 and len(answer) >= 3:
            pairs.append({"question": question, "answer": answer, "chunk": _make_faq_chunk(base_chunk, question, answer)})

    return pairs


def _extract_numbered_faq_pairs_from_chunk(c: dict) -> list[dict]:
    """Extract numbered FAQ pairs inside one chunk."""
    return _extract_numbered_faq_pairs_from_text(c.get("text", ""), c)


def _extract_unnumbered_faq_pairs_from_text(text: str, base_chunk: dict) -> list[dict]:
    """Extract FAQ pairs when questions have NO leading number.

    Word "List Paragraph" numbering is visual only and is absent from the
    extracted text, so the numbered extractor misses these. A question here is
    any clause that begins with an interrogative word and ends in '?'. The
    answer is the text up to the next such question.
    """
    pairs = []
    text = re.sub(r"\s+", " ", str(text or "")).strip()

    starters = r"(?:How|What|Where|When|Whom|Who|Why|Which|Can|Could|Should|Would|Will|Is|Are|Do|Does|Did)"
    scenario_prefix = r"(?:(?:I|We|My|Our|An employee|The employee|Employees|Staff|A staff member)\b[^?]{0,120}?\.\s+)?"
    # Case-sensitive on purpose: skips lowercase "is/are/do" sitting inside answer text.
    # The optional scenario prefix keeps first-person FAQ questions intact.
    q_pattern = re.compile(rf"({scenario_prefix}{starters}\b.{{3,200}}?\?)")


    matches = list(q_pattern.finditer(text))
    for i, m in enumerate(matches):
        question = _clean_faq_text(m.group(1))
        ans_start = m.end()
        ans_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        answer = _clean_faq_answer(text[ans_start:ans_end])
        if len(question) >= 6 and len(answer) >= 3:
            pairs.append({"question": question, "answer": answer, "chunk": _make_faq_chunk(base_chunk, question, answer)})

    return pairs


def extract_faq_pairs(pipeline: RAGPipeline) -> list[dict]:
    """Extract FAQ pairs both chunk-wise and across neighbouring chunks.

    Some DOCX/PDF chunks split a question and its answer across boundaries.
    This function also builds a document-level joined text per file, so exact
    FAQ answers remain consistent and do not accidentally include the next FAQ.
    """
    pairs = []
    seen = set()

    def add_pair(pair: dict):
        q = _clean_faq_text(pair.get("question", ""))
        a = _clean_faq_answer(pair.get("answer", ""))
        if len(q) < 6 or len(a) < 3:
            return
        sig = (normalize_text(q), normalize_text(a)[:240])
        if sig in seen:
            return
        seen.add(sig)
        chunk = pair.get("chunk") or {}
        pairs.append({"question": q, "answer": a, "chunk": _make_faq_chunk(chunk, q, a)})

    # 1) Existing chunk-local extraction.
    for c in pipeline.chunks:
        for pair in _extract_qa_label_answer_pairs_from_chunk(c):
            add_pair(pair)
        for pair in _extract_numbered_faq_pairs_from_chunk(c):
            add_pair(pair)

    # 2) Cross-chunk extraction grouped by filename/document.
    grouped: dict[str, list[dict]] = {}
    for c in sorted(pipeline.chunks, key=lambda x: x.get("chunk_id", 0)):
        fname = c.get("filename") or "__document__"
        grouped.setdefault(fname, []).append(c)

    for _fname, chunks_for_file in grouped.items():
        joined_text = "\n".join(c.get("text", "") for c in chunks_for_file)
        base_chunk = dict(chunks_for_file[0]) if chunks_for_file else {}
        for pair in _extract_numbered_faq_pairs_from_text(joined_text, base_chunk):
            add_pair(pair)
        for pair in _extract_unnumbered_faq_pairs_from_text(joined_text, base_chunk):
            add_pair(pair)

    return pairs


def faq_answer(question: str, pipeline: RAGPipeline) -> Optional[RAGResult]:
    if not ss.cfg.get("use_faq_layer", True):
        return None

    pairs = extract_faq_pairs(pipeline)
    if not pairs:
        return None

    ranked_pairs = []
    for pair in pairs:
        score = _faq_question_similarity(question, pair["question"], pair["answer"])
        ranked_pairs.append((score, pair))

    ranked_pairs.sort(key=lambda x: x[0], reverse=True)
    best_score, best = ranked_pairs[0]

    # Require a real FAQ match. This prevents unrelated adjacent FAQs from being used.
    # Require a real FAQ match. Validated gate: wrong-routing cases score <= 1.83,
    # correct FAQ hits score >= 3.21, so 2.2 + a question-similarity check separates cleanly.
    if best_score < 2.2:
        return None
    try:
        from difflib import SequenceMatcher
        q_sim = SequenceMatcher(None, normalize_text(question), normalize_text(best["question"])).ratio()
    except Exception:
        q_sim = 0.0
    question_overlap = keyword_score(question, best["question"])
    answer_overlap = keyword_score(question, best["answer"])
    if q_sim < 0.55 and question_overlap < 0.5 and answer_overlap < 0.55:
        return None

    answer = _clean_faq_answer(best["answer"])
    chunk = _make_faq_chunk(best["chunk"], best["question"], answer)

    temp_result = manual_result(answer, pipeline, mode="faq", chunks=[chunk], score=min(1.0, 0.72 + best_score / 8))
    temp_result.answer = enrich_answer_with_document_links(answer, question, temp_result)
    return temp_result


def _unique_parent_chunks(pipeline: RAGPipeline) -> list[dict]:
    values: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for chunk in sorted(pipeline.chunks, key=lambda item: item.get("chunk_id", 0)):
        if chunk.get("record_type") == "faq" or chunk.get("content_role") in {"faq", "toc", "front_matter", "title_page", "links"}:
            continue
        key = (str(chunk.get("filename") or ""), str(chunk.get("parent_id") or chunk.get("chunk_id")))
        if key in seen:
            continue
        seen.add(key)
        values.append(chunk)
    return values


def _source_sentences(text: str) -> list[str]:
    value = re.sub(r"\s*[•▪◦‣]\s*", "\n• ", str(text or ""))
    parts = re.split(r"(?<=[.!?])\s+|\n+", value)
    return [re.sub(r"\s+", " ", part).strip(" \t\n•-") for part in parts if len(part.strip(" \t\n•-")) >= 8]


def anchored_list_answer(question: str, pipeline: RAGPipeline) -> Optional[RAGResult]:
    """Return a complete source list when the question names its lead phrase.

    This is generic for statements such as ``The main X are: • ...`` and avoids
    a small model returning only the first few bullets.
    """
    q_norm = normalize_text(question)
    match = re.match(r"^(?:what are|list|show me|give me)\s+(?:all\s+)?(?:the\s+)?(.+?)(?:\?|$)", q_norm)
    if not match:
        return None
    target = match.group(1).strip()
    target_terms = set(tokens(target))
    if len(target_terms) < 2:
        return None

    best: tuple[float, dict, str] | None = None
    for chunk in _unique_parent_chunks(pipeline):
        text = str(chunk.get("parent_text") or chunk.get("text") or "").strip()
        if text.count("•") < 2:
            continue
        prefix = text.split("•", 1)[0]
        prefix_terms = set(tokens(prefix))
        coverage = len(target_terms & prefix_terms) / max(1, len(target_terms))
        exact = 1.0 if normalize_text(target) in normalize_text(prefix) else 0.0
        score = coverage + exact
        if best is None or score > best[0]:
            best = (score, chunk, text)
    if best is None or best[0] < 0.82:
        return None

    _score, chunk, text = best
    section_title = str(chunk.get("section_title") or chunk.get("section") or "").strip()
    lead, *raw_items = text.split("•")
    if section_title and normalize_text(lead).startswith(normalize_text(section_title)):
        lead = lead[len(section_title):].strip()
    lead = re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", lead).strip()
    items: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        item = re.sub(r"\s+", " ", raw).strip(" -;:")
        sig = normalize_text(item)
        if len(item) >= 4 and sig not in seen:
            seen.add(sig)
            items.append(item)
    if len(items) < 2:
        return None
    heading = lead.rstrip(":") + ":" if lead else "Relevant items:"
    answer = heading + "\n" + "\n".join(f"- {item}" for item in items)
    return manual_result(sanitize_answer(answer, question), pipeline, mode="anchored-list", chunks=[chunk], score=0.98)


def application_requirements_answer(question: str, pipeline: RAGPipeline) -> Optional[RAGResult]:
    """Answer application and required-document questions from one coherent section.

    The trigger and ranking use only generic application language. This prevents
    a narrow sentence extractor from selecting an incidental later sentence and
    keeps the implementation reusable for future policies.
    """
    q_norm = normalize_text(question)
    asks_application = bool(re.search(r"\b(apply|application)\b", q_norm))
    asks_documents = bool(
        re.search(r"\b(documents?|proof|certificate)\b", q_norm)
        and re.search(r"\b(need|required|submit|provide|send|attach)\b", q_norm)
    )
    if not (asks_application or asks_documents):
        return None

    mentioned = _mentioned_loaded_filenames(question, pipeline)
    q_terms = set(tokens(question)) - {
        "apply", "application", "document", "documents", "need", "required",
        "submit", "submission", "provide", "send", "attach", "proof", "certificate",
        "policy", "guideline", "guidelines",
    }
    ranked: list[tuple[float, dict]] = []
    for chunk in _unique_parent_chunks(pipeline):
        filename = str(chunk.get("filename") or "")
        if mentioned and filename not in mentioned:
            continue
        section = str(chunk.get("section_title") or chunk.get("section") or "")
        body = str(chunk.get("parent_text") or chunk.get("text") or "")
        section_norm = normalize_text(section)
        body_norm = normalize_text(body)
        score = 0.0

        if re.search(r"\b(notification|application|procedure|process|requirements?)\b", section_norm):
            score += 4.0
        if re.search(r"\b(return|contact|pay|rights?|exception|monitoring)\b", section_norm):
            score -= 1.0

        body_terms = set(tokens(body[:5000]))
        score += 1.25 * (len(q_terms & body_terms) / max(1, len(q_terms)))
        if asks_documents:
            score += 0.55 * len(re.findall(
                r"\b(provide|submit|attach|certificate|proof|official notification)\b",
                body_norm,
            ))
        if asks_application:
            score += 0.35 * len(re.findall(
                r"\b(notify|notification|provide|submit|within\s+\d+|start date|placement date)\b",
                body_norm,
            ))
        ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked or ranked[0][0] < 2.0:
        return None

    best = ranked[0][1]
    context = str(best.get("parent_text") or best.get("text") or "").strip()
    if not context:
        return None

    answer = generate_answer_full([context], question, max(ss.cfg.get("max_tokens", 320), 420))
    if not answer or answer.startswith("[ERROR]") or "Answer not found" in answer:
        answer = _extractive_conditional_requirements(context, question)
    answer = sanitize_answer(answer, question)
    if not answer or "Answer not found" in answer:
        return None

    result = manual_result(answer, pipeline, mode="application-requirements", chunks=[best], score=0.95)
    result.answer = enrich_answer_with_document_links(answer, question, result)
    return result


def actor_action_answer(question: str, pipeline: RAGPipeline) -> Optional[RAGResult]:
    """Extract all parallel obligations for an explicitly named actor and topic."""
    q_norm = normalize_text(question)
    actor_match = re.search(
        r"\b(?:what|which)\s+(?:should|must|shall|does)\s+(?:the\s+)?"
        r"(company|organisation|organization|employer|employee|staff|manager|team)\s+do\b",
        q_norm,
    )
    if not actor_match:
        return None
    actor = actor_match.group(1)
    chunks = find_section_chunks_by_topic(pipeline, question)
    if not chunks:
        return None
    first = chunks[0]
    text = str(first.get("parent_text") or first.get("text") or "")
    section_title = str(first.get("section_title") or first.get("section") or "").strip()
    if section_title and normalize_text(text).startswith(normalize_text(section_title)):
        text = text[len(section_title):].strip()
    actor_forms = {actor}
    if actor in {"organisation", "organization", "employer"}:
        actor_forms.update({"company", "organisation", "organization", "employer"})
    if actor in {"employee", "staff"}:
        actor_forms.update({"employee", "employees", "staff"})
    topic_terms = set(tokens(question)) - {
        "company", "organisation", "organization", "employer", "employee", "employees",
        "staff", "manager", "team", "should", "must", "shall", "does", "what", "which", "do",
    }
    selected: list[tuple[int, str]] = []
    for index, sentence in enumerate(_source_sentences(text)):
        low = normalize_text(sentence)
        # The actor must occur before the obligation verb. This excludes
        # explanatory sentences such as "This will include ... the Company's ...".
        actor_before_verb = any(
            re.search(
                rf"\b{re.escape(form)}\b.{{0,120}}\b(must|shall|will|should|required to|is required to)\b",
                low,
            )
            for form in actor_forms
        )
        if not actor_before_verb:
            continue
        overlap = len(topic_terms & set(tokens(sentence)))
        if topic_terms and overlap == 0:
            continue
        selected.append((index, sentence))
    if len(selected) < 2:
        return None
    selected.sort(key=lambda item: item[0])
    answer = f"The {actor} should:\n" + "\n".join(f"- {sentence}" for _index, sentence in selected[:6])
    return manual_result(sanitize_answer(answer, question), pipeline, mode="actor-actions", chunks=chunks[:4], score=0.97)


def consequence_policy_answer(question: str, pipeline: RAGPipeline) -> Optional[RAGResult]:
    """Extract the distinct source consequences for a stated condition."""
    q_norm = normalize_text(question)
    if not re.search(r"\b(what happens if|what if|consequence|consequences)\b", q_norm):
        return None
    q_terms = set(tokens(question)) - {"what", "happens", "if", "consequence", "consequences", "employee", "employees"}
    scored: list[tuple[float, int, dict, str]] = []
    order = 0
    for chunk in _unique_parent_chunks(pipeline):
        for sentence in _source_sentences(str(chunk.get("parent_text") or chunk.get("text") or "")):
            order += 1
            s_terms = set(tokens(sentence))
            overlap = len(q_terms & s_terms) / max(1, len(q_terms))
            if overlap <= 0:
                continue
            low = normalize_text(sentence)
            consequence = bool(re.search(r"\b(must|shall|will|may|should|treated as|result in|required to)\b", low))
            if not consequence:
                continue
            negation_bonus = 0.0
            if "not" in q_norm:
                # Preserve the complete negative condition, not merely a stray
                # word such as "not" elsewhere in a return-to-work sentence.
                if "return" in q_norm:
                    negative_return = bool(re.search(
                        r"\b(?:(?:does?|do|will)\s+not(?:\s+wish\s+to)?|decides?\s+not\s+to|failure\s+to|fail(?:s|ed)?\s+to)\s+return(?:\s+to\s+work)?\b",
                        low,
                    ))
                    if not negative_return:
                        continue
                elif not re.search(r"\b(not|failure|fail(?:s|ed|ure)?|decides? not|does not|do not)\b", low):
                    continue
                negation_bonus = 0.45
            score = overlap + negation_bonus
            if score >= 0.35:
                scored.append((score, order, chunk, sentence))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_file = str(scored[0][2].get("filename") or "")
    chosen: list[tuple[int, dict, str]] = []
    seen: set[str] = set()
    for score, index, chunk, sentence in scored:
        if str(chunk.get("filename") or "") != best_file:
            continue
        sig = normalize_text(sentence)
        sentence_terms = set(tokens(sentence))
        duplicate = False
        for _old_index, _old_chunk, old_sentence in chosen:
            old_terms = set(tokens(old_sentence))
            if len(sentence_terms & old_terms) / max(1, len(sentence_terms | old_terms)) >= 0.68:
                duplicate = True
                break
        if sig in seen or duplicate:
            continue
        seen.add(sig)
        chosen.append((index, chunk, sentence))
        if len(chosen) >= 3:
            break
    if not chosen:
        return None
    chosen.sort(key=lambda item: item[0])
    answer = "The document distinguishes these outcomes:\n" + "\n".join(
        f"- {sentence}" for _index, _chunk, sentence in chosen
    )
    return manual_result(sanitize_answer(answer, question), pipeline, mode="consequence-extractive", chunks=[item[1] for item in chosen], score=0.95)


def high_overlap_sentence_answer(question: str, pipeline: RAGPipeline) -> Optional[RAGResult]:
    """Return one highly matching sentence only for a genuinely narrow query.

    Broad explanations, summaries, multi-topic requests, and explicit section-title
    questions must continue to section or full RAG handling. Otherwise this shortcut
    can reduce a complete policy answer to one incidental sentence.
    """
    if (
        is_multi_topic_request(question)
        or is_summary_request(question)
        or wants_full_section(question)
        or wants_exhaustive_topic(question)
    ):
        return None

    q_norm = normalize_text(question)
    # Application, submission, and document-requirement questions need a
    # complete procedure/requirements section. A one-sentence shortcut can
    # otherwise select an incidental sentence from a later section.
    if re.search(
        r"\b(apply|application|documents?|submit|submission|provide|proof|certificate|required documents?|requirements?)\b",
        q_norm,
    ):
        return None

    mentioned = _mentioned_loaded_filenames(question, pipeline)
    if not mentioned and "according to" not in normalize_text(question):
        return None
    q_terms = set(tokens(question)) - {"according", "policy", "document", "guideline", "guidelines"}
    if len(q_terms) < 3:
        return None
    best: tuple[float, dict, str] | None = None
    for chunk in _unique_parent_chunks(pipeline):
        if mentioned and str(chunk.get("filename") or "") not in mentioned:
            continue
        for sentence in _source_sentences(str(chunk.get("parent_text") or chunk.get("text") or "")):
            s_terms = set(tokens(sentence))
            overlap_count = len(q_terms & s_terms)
            coverage = overlap_count / max(1, len(q_terms))
            if overlap_count < 3:
                continue
            score = coverage + 0.15 * keyword_score(question, sentence)
            if best is None or score > best[0]:
                best = (score, chunk, sentence)
    if best is None or best[0] < 0.58:
        return None
    _score, chunk, sentence = best
    return manual_result(sanitize_answer(sentence, question), pipeline, mode="high-overlap-extractive", chunks=[chunk], score=0.94)

def deterministic_policy_answer(question: str, pipeline: RAGPipeline) -> Optional[RAGResult]:
    if ss.cfg.get("use_offtopic_guard", True) and is_offtopic_or_smalltalk(question):
        return manual_result(
            "I'm an HR Assistant. Please ask me an HR-related question from the available documents.",
            pipeline,
            mode="guard",
            chunks=[],
            score=1.0,
        )

    for route in (
        clause_answer, full_document_answer, section_answer, anchored_list_answer,
        actor_action_answer, consequence_policy_answer, application_requirements_answer,
        heading_answer, high_overlap_sentence_answer,
    ):
        hit = route(question, pipeline)
        if hit:
            return hit
    return None


def is_summary_request(question: str) -> bool:
    q = normalize_text(question)
    return bool(re.search(r"\b(summary|summarize|summarise|overview|main points|key points|brief|gist|short note|abstract)\b", q))


def is_heading_style_request(question: str) -> bool:
    q = normalize_text(question)
    return bool(re.search(
        r"\b(what is|what are|explain|describe|tell me about|details of|information about|meaning of)\b",
        q,
    ))


def is_multi_topic_request(question: str) -> bool:
    """Detect questions asking for more than one substantive topic.

    Such questions should use normal multi-chunk retrieval instead of being
    forced into a single heading/section. This is corpus-neutral.
    """
    q = normalize_text(question)
    return len(tokens(question)) >= 5 and bool(re.search(r"\b(and|or|plus|as well as|along with)\b", q))


def _root_section_term(term: str) -> str:
    """Return a light lexical root for generic section-title matching."""
    value = str(term or "").casefold()
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if value.endswith(suffix) and len(value) > len(suffix) + 3:
            return value[:-3] + "y" if suffix == "ies" else value[:-len(suffix)]
    return value


def _strong_named_section_chunks(pipeline: RAGPipeline, question: str) -> list[dict]:
    """Return one section when the user clearly names its title.

    Matching is derived entirely from indexed section metadata. Request words
    are ignored, light singular/verb normalisation is applied, and a minor
    spelling variation is tolerated. This prevents generic retrieval from
    selecting a neighbouring section when the actual section title is present
    in the question.
    """
    from difflib import SequenceMatcher

    q_norm = normalize_text(question)
    q_terms = {_root_section_term(term) for term in tokens(question)}
    if not q_terms:
        return []

    parent_groups: dict[tuple[str, str], list[dict]] = {}
    for chunk in getattr(pipeline, "chunks", []):
        role = chunk.get("content_role")
        if chunk.get("record_type") == "faq" or role in {
            "faq", "toc", "front_matter", "title_page", "links"
        }:
            continue
        filename = str(chunk.get("filename") or "")
        parent_id = str(chunk.get("parent_id") or f"chunk::{chunk.get('chunk_id', 0)}")
        parent_groups.setdefault((filename, parent_id), []).append(chunk)

    best: tuple[float, list[dict]] | None = None
    for group in parent_groups.values():
        first = group[0]
        raw_title = str(first.get("section_title") or first.get("section") or "").strip()
        if not raw_title:
            continue
        title = re.sub(r"^\s*\d+(?:\.\d+)*[\s.):-]+", "", raw_title).strip()
        title_terms = {_root_section_term(term) for term in tokens(title)}
        if len(title_terms) < 2:
            continue

        exact_hits = len(title_terms & q_terms)
        coverage = exact_hits / len(title_terms)

        # Approximate token coverage tolerates one small spelling error while
        # still requiring almost the whole section title to be present.
        approximate_hits = 0
        for title_term in title_terms:
            if title_term in q_terms:
                approximate_hits += 1
                continue
            if any(SequenceMatcher(None, title_term, q_term).ratio() >= 0.86 for q_term in q_terms):
                approximate_hits += 1
        approximate_coverage = approximate_hits / len(title_terms)

        clean_title = normalize_text(title)
        phrase_match = bool(clean_title and clean_title in q_norm)
        sequence = SequenceMatcher(
            None,
            " ".join(sorted(title_terms)),
            " ".join(sorted(q_terms)),
        ).ratio()
        score = (
            0.55 * approximate_coverage
            + 0.25 * coverage
            + 0.20 * sequence
            + (0.35 if phrase_match else 0.0)
        )

        required_hits = 2 if len(title_terms) <= 3 else max(3, len(title_terms) - 1)
        reliable = (
            phrase_match
            or (
                approximate_hits >= required_hits
                and approximate_coverage >= 0.75
                and sequence >= 0.52
            )
        )
        if reliable and (best is None or score > best[0]):
            best = (score, sorted(group, key=lambda item: item.get("chunk_id", 0)))

    return best[1] if best else []


def _heading_title_coverage(title: str, question: str) -> float:
    """Measure title coverage with light singular/derivational normalization."""
    def root(term: str) -> str:
        value = term.casefold()
        for suffix in ("ies", "ing", "ed", "es", "s"):
            if value.endswith(suffix) and len(value) > len(suffix) + 3:
                return value[:-3] + "y" if suffix == "ies" else value[:-len(suffix)]
        return value

    title_terms = {root(term) for term in tokens(title)}
    question_terms = {root(term) for term in tokens(question)}
    if not title_terms:
        return 0.0
    return len(title_terms & question_terms) / len(title_terms)


def heading_answer(question: str, pipeline: RAGPipeline) -> Optional[RAGResult]:
    """Answer from a strongly matched substantive section.

    Exact section-title requests may return the complete parent section without
    involving the LLM. Action questions with a strong section-title match use
    the whole parent section as grounded generation context.
    """
    if detect_clause(question) or detect_section_number(question) or wants_full_document(question):
        return None

    question_tokens = tokens(question)
    q_norm = normalize_text(question)
    starts_as_question = bool(re.match(
        r"^(what|how|why|where|when|who|whom|which|can|could|should|would|is|are|do|does|did)\b",
        q_norm,
    ))
    answer_type = question_answer_type(question)

    # Prefer a section title explicitly present in the question before
    # generic topic scoring. This is metadata-driven and works for any corpus.
    chunks = _strong_named_section_chunks(pipeline, question)
    if not chunks:
        chunks = find_section_chunks_by_topic(pipeline, question)
    if not chunks:
        return None

    first = chunks[0]
    title = first.get("section_title") or first.get("section") or "Matched section"
    # Coverage is measured from the section title into the question. A question
    # can contain additional scenario words while still explicitly naming the
    # section, e.g. "what should the company do after a data breach?".
    title_coverage = _heading_title_coverage(title, question)
    strong_heading_request = (
        title_coverage >= 0.66
        or (normalize_text(title) and normalize_text(title) in q_norm)
    )

    # The word "and" can be part of one section title, for example
    # "Rights on and after returning to work". Treat it as multi-topic only
    # when the query does not strongly name a single section.
    if is_multi_topic_request(question) and not wants_full_section(question) and not strong_heading_request:
        return None

    # Scenario, timing, and action questions normally continue to full RAG.
    # A strong section-title match is the exception because the relevant parent
    # section is already known deterministically.
    if starts_as_question and not is_heading_style_request(question) and answer_type != "eligibility" and not strong_heading_request:
        return None
    if not is_heading_style_request(question) and answer_type != "eligibility" and len(question_tokens) > 6 and not strong_heading_request:
        return None

    # A generic answer type is not enough to justify deterministic routing.
    # The selected heading must also match the substantive topic in the query.
    if starts_as_question and heading_topic_overlap(question, chunks) <= 0.0 and not strong_heading_request:
        return None

    selected = chunks[:18]
    short_heading_request = (
        not starts_as_question
        and len(question_tokens) <= 6
        and keyword_score(question, title) >= 0.65
    )
    explicit_title_content_request = strong_heading_request and bool(re.search(
        r"^(?:what\s+(?:is|are|should\s+be)|tell\s+me\s+(?:all|about)|"
        r"explain|describe|give\s+me\s+(?:all|the))\b",
        q_norm,
    ))

    if wants_full_section(question) or wants_exhaustive_topic(question) or short_heading_request or explicit_title_content_request:
        body = (first.get("parent_text") or "").strip()
        if not body:
            seen, parts = set(), []
            for chunk in selected:
                text = (chunk.get("text") or "").strip()
                sig = normalize_text(text)[:300]
                if text and sig not in seen:
                    seen.add(sig)
                    parts.append(text)
            body = "\n\n".join(parts)
        answer = body if normalize_text(body).startswith(normalize_text(title)) else f"{title}\n\n{body}"
        answer = sanitize_answer(answer, question)
        return manual_result(answer, pipeline, mode="heading-exact", chunks=selected[:8], score=0.96)

    texts, seen = [], set()
    # When the heading is strongly named, use the complete parent section so
    # all required actions are available to the generator.
    if strong_heading_request and first.get("parent_text"):
        texts = [str(first.get("parent_text") or "").strip()]
    else:
        for chunk in selected[:6]:
            text = (chunk.get("text") or "").strip()
            sig = normalize_text(text)[:300]
            if text and sig not in seen:
                seen.add(sig)
                texts.append(text)
    if not texts:
        return None

    answer = generate_answer_full(texts, question, max(ss.cfg.get("max_tokens", 320), 420 if strong_heading_request else 320))
    if answer.startswith("[ERROR]") or "Answer not found" in answer:
        return None
    answer = sanitize_answer(answer, question)
    result = manual_result(answer, pipeline, mode="heading", chunks=selected[:6], score=0.88)
    result.answer = enrich_answer_with_document_links(answer, question, result)
    return result


def estimate_available_content_words(question: str, pipeline: RAGPipeline) -> Optional[int]:
    """Estimate how much directly matched content exists for the question.

    This helps prevent long answers when the document only contains a small amount
    of relevant content under a heading/clause/section.
    """
    matched_chunks: list[dict] = []

    clause_no = detect_clause(question)
    if clause_no:
        for c in pipeline.chunks:
            extracted = extract_clause_text_from_chunk(c.get("text", ""), clause_no)
            if extracted:
                matched_chunks.append({"text": extracted})

    sec_no = detect_section_number(question)
    if sec_no and not matched_chunks:
        matched_chunks = find_section_chunks_by_number(pipeline, sec_no)

    if not matched_chunks and (is_heading_style_request(question) or wants_full_section(question)):
        matched_chunks = find_section_chunks_by_topic(pipeline, question)

    if matched_chunks:
        return sum(len((c.get("text") or "").split()) for c in matched_chunks)
    return None


def auto_backend_config(question: str, pipeline: RAGPipeline) -> tuple[dict, str]:
    """Choose hidden backend settings automatically for each question.

    Users should not see Candidate K, Final K, confidence thresholds, HyDE, or
    token limits. The app uses safe production defaults and adjusts internally.
    """
    cfg = DEFAULT_CFG.copy()
    search_mode = "hybrid"

    q = normalize_text(question)
    available_words = estimate_available_content_words(question, pipeline)
    total_chunks = max(1, len(pipeline.chunks))

    wants_detail = bool(re.search(
        r"\b(detail|details|detailed|elaborate|comprehensive|in[\s-]?depth|thorough|thoroughly|full explanation|step by step|step-by-step|explain|easily|simple explanation)\b",
        q,
    ))
    multi_topic = is_multi_topic_request(question)

    if wants_full_document(question):
        cfg.update({"candidate_k": 35, "final_k": 6, "rerank_k": 35, "max_tokens": 640, "context_window": 1, "evidence_threshold": 0.34})
    elif multi_topic:
        cfg.update({"candidate_k": 46, "final_k": 8, "rerank_k": 42, "max_tokens": 560, "context_window": 1, "evidence_threshold": 0.33})
    elif wants_detail:
        cfg.update({"candidate_k": 38, "final_k": 6, "rerank_k": 36, "max_tokens": 448, "context_window": 1, "evidence_threshold": 0.35})
    elif is_summary_request(question):
        cfg.update({"candidate_k": 32, "final_k": 5, "rerank_k": 32, "max_tokens": 384, "context_window": 1, "evidence_threshold": 0.34})
    elif detect_clause(question) or detect_section_number(question) or wants_full_section(question):
        cfg.update({"candidate_k": 30, "final_k": 4, "rerank_k": 30, "max_tokens": 384, "context_window": 0, "evidence_threshold": 0.37})
    elif re.search(r"\b(list|steps|procedure|process|requirements|conditions|eligibility|entitlement)\b", q):
        cfg.update({"candidate_k": 30, "final_k": 4, "rerank_k": 30, "max_tokens": 384, "context_window": 0, "evidence_threshold": 0.38})
    else:
        cfg.update({"candidate_k": 30, "final_k": 4, "rerank_k": 30, "max_tokens": 320, "context_window": 0, "evidence_threshold": 0.39})

    # Content-aware cap: keep answers short when little content exists, but never
    # cap explicit detail / full requests.
    if available_words is not None and not wants_detail and not multi_topic and not wants_full_document(question):
        if available_words <= 80:
            cfg["max_tokens"] = min(cfg["max_tokens"], 220)
            cfg["final_k"] = min(cfg["final_k"], 3)
        elif available_words <= 180:
            cfg["max_tokens"] = min(cfg["max_tokens"], 340)
            cfg["final_k"] = min(cfg["final_k"], 4)

            
    cfg["candidate_k"] = min(max(1, cfg["candidate_k"]), total_chunks)
    cfg["final_k"] = min(max(1, cfg["final_k"]), cfg["candidate_k"])
    cfg["rerank_k"] = min(max(cfg["final_k"], int(cfg.get("rerank_k", 12))), cfg["candidate_k"])
    cfg["use_hyde"] = False
    cfg["use_policy_router"] = True
    cfg["use_faq_layer"] = True
    cfg["use_offtopic_guard"] = True
    return cfg, search_mode


def apply_backend_config_for_question(question: str, pipeline: RAGPipeline) -> str:
    cfg, search_mode = auto_backend_config(question, pipeline)
    ss.cfg = cfg
    ss.search_mode = search_mode
    pipeline.cfg = cfg
    return search_mode

# ─────────────────────────────────────────────
# Chat export helpers
# ─────────────────────────────────────────────


def source_refs(result: RAGResult) -> list[str]:
    refs = []
    for r in result.retrieved:
        label = r.label or f"Location {r.page}"
        filename = r.filename or ""
        refs.append(f"{filename} · {label}".strip(" ·"))
    return sorted(set(refs))


def save_session_feedback(rating: int, comment: str = "") -> None:
    """Save one feedback entry for the whole chat session.

    Sources are intentionally hidden from the UI, but preserved in the DB/export
    for admin audit and accuracy improvement.
    """
    init_feedback_db()
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            rating INTEGER,
            comment TEXT,
            transcript_json TEXT,
            source_refs TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    transcript = []
    all_sources: list[str] = []
    for item in ss.chat_history:
        result = item.get("result")
        refs = source_refs(result) if result else []
        all_sources.extend(refs)
        transcript.append({
            "timestamp": item.get("timestamp", ""),
            "question": item.get("q", ""),
            "answer": result.answer if result else "",
            "sources": refs,
            "search_mode": result.search_mode if result else "",
            "confidence_score": result.confidence_score if result else None,
            "verification": result.verification if result else "",
        })

    conn.execute(
        "INSERT INTO session_feedback (session_id, rating, comment, transcript_json, source_refs) VALUES (?,?,?,?,?)",
        (
            ss.session_id,
            int(rating),
            str(comment or "").strip(),
            json.dumps(transcript, ensure_ascii=False),
            "; ".join(sorted(set(all_sources)))[:3000],
        ),
    )
    conn.commit()
    conn.close()


def generate_markdown_report(history: list, doc_name: str) -> str:
    md = f"# Chat History - {doc_name}\n*Generated by Elite Multi-Doc RAG*\n\n"
    md += f"- Session ID: `{ss.session_id}`\n"
    md += f"- Exported: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
    md += f"- Exchanges: `{len(history)}`\n\n---\n\n"
    for i, item in enumerate(history, start=1):
        result = item["result"]
        md += f"## {i}. Q: {item['q']}\n\n"
        md += f"**Answer:**\n\n{result.answer}\n\n"
        refs = source_refs(result)
        if refs:
            md += "**Sources:**\n"
            for ref in refs:
                md += f"- {ref}\n"
            md += "\n"
        md += f"**Confidence:** {result.confidence_score:.3f} | **Verification:** {result.verification} | **Mode:** {result.search_mode}\n\n---\n\n"
    return md


def generate_txt_export(history: list, title: str) -> str:
    lines = [
        "=" * 72,
        f"  {title}",
        "  Elite Multi-Doc RAG - Chat Export",
        "=" * 72,
        f"Session ID : {ss.session_id}",
        f"Exported   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Exchanges  : {len(history)}",
        "=" * 72,
        "",
    ]
    for i, item in enumerate(history, start=1):
        result = item["result"]
        lines += [f"[{i}] YOU:", f"  {item['q']}", "", "ASSISTANT:", f"  {result.answer}", ""]
        refs = source_refs(result)
        if refs:
            lines.append("SOURCES:")
            for ref in refs:
                lines.append(f"  - {ref}")
            lines.append("")
        lines += [f"Confidence: {result.confidence_score:.3f} | Verification: {result.verification} | Mode: {result.search_mode}", "-" * 72, ""]
    return "\n".join(lines)


def generate_json_export(history: list, title: str) -> str:
    data = {"title": title, "session_id": ss.session_id, "exported_at": datetime.now().isoformat(), "total_exchanges": len(history), "history": []}
    for item in history:
        result = item["result"]
        data["history"].append({
            "timestamp": item.get("timestamp", ""),
            "document_key": item.get("doc", ""),
            "question": item.get("q", ""),
            "answer": result.answer,
            "confidence_score": result.confidence_score,
            "verification": result.verification,
            "search_mode": result.search_mode,
            "references": source_refs(result),
            "source_chunks": [
                {"filename": r.filename, "label": r.label or f"Location {r.page}", "page_or_location": r.page, "section": r.section, "vector_score": r.vector_score, "rerank_score": r.rerank_score, "text": r.text}
                for r in result.retrieved
            ],
        })
    return json.dumps(data, indent=2, ensure_ascii=False)


def generate_html_export(history: list, title: str) -> str:
    safe_title = html_lib.escape(title)
    html = f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>{safe_title}</title>
<style>body{{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:20px;background:#f5f7fa;color:#111827}}.header{{background:#111827;color:white;padding:22px;border-radius:12px;margin-bottom:24px}}.user{{background:#1a56db;color:white;padding:12px 16px;border-radius:16px 16px 4px 16px;margin:12px 0 12px 20%;white-space:pre-wrap}}.bot{{background:white;border:1px solid #e2e8f0;padding:12px 16px;border-radius:16px 16px 16px 4px;margin:12px 12% 22px 0;white-space:pre-wrap}}.sources{{font-size:12px;color:#475569;margin-top:8px}}.meta{{font-size:11px;color:#64748b;margin-top:6px}}</style></head><body>
<div class="header"><h1>{safe_title}</h1><p>Session: {html_lib.escape(ss.session_id)} | Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Exchanges: {len(history)}</p></div>
'''
    for item in history:
        result = item["result"]
        q = html_lib.escape(item.get("q", ""))
        a = html_lib.escape(result.answer)
        refs = source_refs(result)
        refs_html = ""
        if refs:
            refs_html = "<div class='sources'><b>Sources:</b><ul>" + "".join(f"<li>{html_lib.escape(ref)}</li>" for ref in refs) + "</ul></div>"
        html += f"<div class='user'>{q}</div><div class='bot'>{a}{refs_html}<div class='meta'>Confidence: {result.confidence_score:.3f} | Verification: {html_lib.escape(str(result.verification))} | Mode: {html_lib.escape(str(result.search_mode))}</div></div>"
    html += "</body></html>"
    return html


def export_buttons(history: list, title: str, key_prefix: str):
    if not history:
        st.caption("No chat history yet.")
        return
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", title.lower()).strip("_") or "chat_history"
    st.download_button("⬇️ TXT", data=generate_txt_export(history, title), file_name=f"{safe_name}.txt", mime="text/plain", use_container_width=True, key=f"{key_prefix}_txt")
    st.download_button("⬇️ JSON", data=generate_json_export(history, title), file_name=f"{safe_name}.json", mime="application/json", use_container_width=True, key=f"{key_prefix}_json")
    st.download_button("⬇️ HTML", data=generate_html_export(history, title), file_name=f"{safe_name}.html", mime="text/html", use_container_width=True, key=f"{key_prefix}_html")
    st.download_button("⬇️ Markdown", data=generate_markdown_report(history, title), file_name=f"{safe_name}.md", mime="text/markdown", use_container_width=True, key=f"{key_prefix}_md")

# ─────────────────────────────────────────────
# Feedback and status helpers
# ─────────────────────────────────────────────


def init_feedback_db():
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            document_key TEXT,
            document_name TEXT,
            question TEXT,
            answer TEXT,
            rating INTEGER,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_feedback(item: dict, rating: int):
    init_feedback_db()
    result = item["result"]
    doc_key_value = item.get("doc", "")
    doc_name = ss.pipelines[doc_key_value].pdf_name if doc_key_value in ss.pipelines else doc_key_value
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute(
        "INSERT INTO feedback (session_id, document_key, document_name, question, answer, rating, sources) VALUES (?,?,?,?,?,?,?)",
        (ss.session_id, doc_key_value, doc_name, item.get("q", ""), result.answer, rating, "; ".join(source_refs(result)))
    )
    conn.commit()
    conn.close()


def feedback_stats() -> dict:
    if not FEEDBACK_DB.exists():
        return {"total": 0, "positive": 0, "negative": 0}
    conn = sqlite3.connect(FEEDBACK_DB)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM feedback")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM feedback WHERE rating > 0")
    pos = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM feedback WHERE rating < 0")
    neg = cur.fetchone()[0]
    conn.close()
    return {"total": total, "positive": pos, "negative": neg}


def check_ollama() -> bool:
    try:
        requests.get("http://localhost:11434", timeout=1.2)
        return True
    except Exception:
        return False


def remove_active_document():
    key = ss.active_doc
    if not key:
        return
    ss.pipelines.pop(key, None)
    ss.doc_bytes.pop(key, None)
    ss.doc_types.pop(key, None)
    ss.chat_history = [m for m in ss.chat_history if m.get("doc") != key]
    ss.active_doc = next(iter(ss.pipelines.keys()), None) if ss.pipelines else None
    ss.view_page = 1
    invalidate_global_pipeline()


def clear_all_documents():
    ss.pipelines = {}
    ss.doc_bytes = {}
    ss.doc_types = {}
    ss.chat_history = []
    ss.active_doc = None
    ss.view_page = 1
    invalidate_global_pipeline()

# ─────────────────────────────────────────────
# Browser voice reader
# ─────────────────────────────────────────────


def voice_reader_component(text: str, component_id: str):
    clean = clean_for_speech(text)[:6000]
    if not clean:
        return
    safe_text = json.dumps(clean, ensure_ascii=False)
    safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", component_id)
    components.html(f'''
<div style="display:flex;gap:8px;align-items:center;margin:4px 0 12px 0;">
<button onclick="speak_{safe_id}()" style="background:#2a2b2e;color:#a8c7fa;border:1px solid #33363d;border-radius:10px;padding:7px 12px;cursor:pointer;">🔊 Read answer</button>
<button onclick="stop_{safe_id}()" style="background:#2a2b2e;color:#e3e3e3;border:1px solid #33363d;border-radius:10px;padding:7px 12px;cursor:pointer;">⏹ Stop</button>
<select id="rate_{safe_id}" style="background:#1e1f20;color:#e3e3e3;border:1px solid #33363d;border-radius:8px;padding:6px;"><option value="0.85">Slow</option><option value="1" selected>Normal</option><option value="1.15">Fast</option></select>
</div>
<script>
const text_{safe_id} = {safe_text};
function speak_{safe_id}() {{
  if (!("speechSynthesis" in window)) {{ alert("Your browser does not support speech synthesis."); return; }}
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text_{safe_id});
  u.lang = "en-IN";
  u.rate = parseFloat(document.getElementById("rate_{safe_id}").value || "1");
  window.speechSynthesis.speak(u);
}}
function stop_{safe_id}() {{ if ("speechSynthesis" in window) window.speechSynthesis.cancel(); }}
</script>
''', height=52, scrolling=False)



def scroll_to_bottom_component():
    """Keep the latest answer/continue prompt visible after Streamlit reruns.

    Streamlit rebuilds the page after every chat submission. In long chats,
    the browser may return to the top, so this small browser-side script
    scrolls to the bottom anchor after the page finishes rendering.
    """
    components.html(
        """
        <script>
        function scrollChatToBottom() {
            const doc = window.parent.document;
            const anchors = doc.querySelectorAll('[data-scroll-anchor="chat-bottom"]');
            const anchor = anchors.length ? anchors[anchors.length - 1] : null;

            if (anchor) {
                anchor.scrollIntoView({behavior: "smooth", block: "end"});
            } else {
                window.parent.scrollTo({top: doc.body.scrollHeight, behavior: "smooth"});
            }

            const chatBox = doc.querySelector('[data-testid="stChatInput"] textarea, [data-testid="stChatInput"] input');
            if (chatBox) {
                setTimeout(function(){ chatBox.focus(); }, 250);
            }
        }

        setTimeout(scrollChatToBottom, 80);
        setTimeout(scrollChatToBottom, 350);
        setTimeout(scrollChatToBottom, 900);
        </script>
        """,
        height=0,
        scrolling=False,
    )

def voice_question_component():
    """Mic button: speech-to-text via the browser's Web Speech API.
    Submits the transcript as ?voice_q=... so the app can process it like a typed question."""
    components.html(
        """
        <style>
        .micwrap { display:flex; align-items:center; gap:10px; font-family:Arial,sans-serif; }
        #micbtn { background:#1a56db; color:#fff; border:none; border-radius:50%;
                  width:44px; height:44px; font-size:20px; cursor:pointer; }
        #micbtn.listening { background:#dc2626; animation:pulse 1s infinite; }
        @keyframes pulse { 0%{opacity:1} 50%{opacity:0.55} 100%{opacity:1} }
        #micstatus { color:#94a3b8; font-size:13px; }
        </style>
        <div class="micwrap">
          <button id="micbtn" title="Ask by voice">🎤</button>
          <span id="micstatus">Click the mic and speak your question</span>
        </div>
        <script>
        (function() {
            const btn = document.getElementById("micbtn");
            const status = document.getElementById("micstatus");
            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SR) {
                btn.disabled = true;
                status.textContent = "Voice input needs Chrome or Edge";
                return;
            }
            const rec = new SR();
            rec.lang = "en-IN";
            rec.interimResults = false;
            rec.maxAlternatives = 1;
            let listening = false;

            btn.onclick = function() {
                if (listening) { rec.stop(); return; }
                try { rec.start(); } catch(e) {}
            };
            rec.onstart = function() {
                listening = true;
                btn.classList.add("listening");
                status.textContent = "Listening... speak now";
            };
            rec.onend = function() {
                listening = false;
                btn.classList.remove("listening");
            };
            rec.onerror = function(e) {
                status.textContent = (e.error === "not-allowed")
                    ? "Microphone blocked - allow mic access in the browser"
                    : "Didn't catch that - click and try again";
            };
            rec.onresult = function(e) {
                const text = e.results[0][0].transcript.trim();
                if (!text) return;
                status.textContent = "Heard: " + text;
                const url = new URL(window.parent.location.href);
                url.searchParams.set("voice_q", text);
                window.parent.location.href = url.toString();
            };
        })();
        </script>
        """,
        height=60,
    )




def remove_streamlit_deploy_modal_component():
    """Remove Streamlit Deploy popup and toolbar repeatedly on every rerun."""
    components.html(
        """
        <script>
        (function() {
            function killDeployUI() {
                const doc = window.parent.document;

                // Hide toolbar/header/deploy buttons if Streamlit recreates them.
                const selectors = [
                    '[data-testid="stToolbar"]',
                    '[data-testid="stDecoration"]',
                    '[data-testid="stStatusWidget"]',
                    '[data-testid="stHeaderActionElements"]',
                    '[data-testid="stDeployButton"]',
                    '[data-testid="stActionButton"]',
                    
                    'button[title="Deploy"]',
                    'button[aria-label="Deploy"]',
                    'a[title="Deploy"]',
                    'a[aria-label="Deploy"]',
                    
                ];
                selectors.forEach(function(sel) {
                    doc.querySelectorAll(sel).forEach(function(el) {
                        el.style.display = "none";
                        el.style.visibility = "hidden";
                        el.style.opacity = "0";
                        el.style.pointerEvents = "none";
                        el.setAttribute("aria-hidden", "true");
                    });
                });

                // Remove the Deploy modal if it is already open.
                const all = Array.from(doc.querySelectorAll('div, section, aside, dialog'));
                all.forEach(function(el) {
                    const txt = (el.innerText || "").trim();
                    if (
                        txt.includes("Deploy this app using") ||
                        (txt.includes("Streamlit Community Cloud") && txt.includes("Snowflake") && txt.includes("Other platforms"))
                    ) {
                        const modal =
                            el.closest('[role="dialog"]') ||
                            el.closest('[data-testid="stModal"]') ||
                            el.closest('dialog') ||
                            el;
                        if (modal && modal.parentNode) {
                            modal.remove();
                        }
                    }
                });

                // Remove any overlay left by the modal.
                Array.from(doc.querySelectorAll('[data-baseweb="modal"], [role="dialog"]')).forEach(function(el) {
                    const txt = (el.innerText || "").trim();
                    if (txt.includes("Deploy this app using") || txt.includes("Streamlit Community Cloud")) {
                        el.remove();
                    }
                });

                // As fallback, click close button on deploy modal.
                Array.from(doc.querySelectorAll('button')).forEach(function(btn) {
                    const label = (btn.getAttribute("aria-label") || btn.title || btn.innerText || "").trim().toLowerCase();
                    const parentText = (btn.closest('[role="dialog"], [data-baseweb="modal"], div')?.innerText || "");
                    if ((label === "close" || label === "×" || label === "x") && parentText.includes("Deploy this app using")) {
                        btn.click();
                    }
                });
            }

            killDeployUI();
            setTimeout(killDeployUI, 100);
            setTimeout(killDeployUI, 500);
            setTimeout(killDeployUI, 1200);
            setInterval(killDeployUI, 1500);
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )




def truncate_history_text(text: str, max_len: int = 70) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def render_custom_chat_history_panel():
    """Render a clean user-visible chat history panel without using Streamlit's sidebar."""
    
    st.markdown('<div class="hr-history-title">💬 Chat History</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hr-history-session">Session: {html_lib.escape(str(ss.session_id[:22]))}...</div>',
        unsafe_allow_html=True,
    )

    if not ss.chat_history:
        st.markdown(
            '<div class="hr-history-note">Your asked questions will appear here during this session.</div>',
            unsafe_allow_html=True,
        )
    else:
        for i, item in enumerate(ss.chat_history, start=1):
            q = html_lib.escape(truncate_history_text(item.get("q", ""), 82))
            st.markdown(
                f'<div class="hr-history-item"><span class="hr-history-number">{i}.</span>{q}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="hr-history-note">Questions are shown here. Sources remain hidden but are saved internally.</div>',
        unsafe_allow_html=True,
    )

    if ss.chat_history:
        st.markdown('<div class="hr-export-title">Export chat</div>', unsafe_allow_html=True)
        export_col1, export_col2 = st.columns(2)
        export_title = "Full Session Chat History"
        safe_export_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", export_title.lower()).strip("_") or "chat_history"

        with export_col1:
            st.download_button(
                "TXT",
                data=generate_txt_export(ss.chat_history, export_title),
                file_name=f"{safe_export_name}.txt",
                mime="text/plain",
                use_container_width=True,
                key="custom_export_txt",
            )
            st.download_button(
                "HTML",
                data=generate_html_export(ss.chat_history, export_title),
                file_name=f"{safe_export_name}.html",
                mime="text/html",
                use_container_width=True,
                key="custom_export_html",
            )

        with export_col2:
            st.download_button(
                "JSON",
                data=generate_json_export(ss.chat_history, export_title),
                file_name=f"{safe_export_name}.json",
                mime="application/json",
                use_container_width=True,
                key="custom_export_json",
            )
            st.download_button(
                "MD",
                data=generate_markdown_report(ss.chat_history, export_title),
                file_name=f"{safe_export_name}.md",
                mime="text/markdown",
                use_container_width=True,
                key="custom_export_md",
            )
    else:
        st.caption("Export will appear after the first question.")

    if st.button("🧹 Start new chat", use_container_width=True, key="custom_start_new_chat"):
        ss.chat_history = []
        ss.awaiting_continue = False
        ss.chat_closed = False
        ss.session_feedback_saved = False
        ss.session_feedback_comment = ""
        ss.pending_query = None
        ss.processing_question = None
        ss.processing_timestamp = None
        if ss.history_store:
            try:
                ss.history_store.clear_session(ss.session_id)
            except Exception:
                pass
        st.rerun()
    


# Keep backend knowledge base synchronized before rendering UI.
sync_backend_documents(force=False)

# ─────────────────────────────────────────────
# Minimal production chatbot UI
# ─────────────────────────────────────────────

# Hidden production controls enforced by backend.
# No Streamlit sidebar is rendered for end users.
ss.search_scope = "All documents"
ss.search_mode = "hybrid"
if not ss.cfg:
    ss.cfg = DEFAULT_CFG.copy()


# ─────────────────────────────────────────────
# Backend loading screen
# ─────────────────────────────────────────────

if not ss.pipelines:
    st.markdown(
        "<h1 style='text-align: center; margin-top: 22vh; font-weight: 400;'>Knowledge base is not ready yet</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#8e918f;'>Please try again in a moment. The backend is preparing the knowledge base.</p>",
        unsafe_allow_html=True,
    )
    time.sleep(1.0)
    st.rerun()


# ─────────────────────────────────────────────
# Chat engine
# ─────────────────────────────────────────────

answer_pipeline: Optional[RAGPipeline] = get_answer_pipeline()
if answer_pipeline is None:
    st.error("Knowledge base is not ready yet. Please try again in a moment.")
    st.stop()
answer_pipeline.cfg = DEFAULT_CFG.copy()

current_history = ss.chat_history

with st.sidebar:
    render_custom_chat_history_panel()

st.markdown(
    """
    <div class="hr-brand-wrap">
      <div class="hr-brand-icon" aria-hidden="true">
        <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M32 8L50 15V29C50 42.5 41.7 53.2 32 57C22.3 53.2 14 42.5 14 29V15L32 8Z" fill="#f59e0b"/>
          <path d="M24.5 32.5L29.2 37.2L40.5 25.8" stroke="#1f2937" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div>
        <div class="hr-brand-title">HR Assistant</div>
        <div class="hr-brand-subtitle">Inventio AI-powered</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_area = st.container()
with chat_area:
    if not current_history:
        st.markdown(
            """
            <div style="max-width:900px;margin:18vh auto 0 auto;text-align:center;color:var(--muted);">
                Start by asking a question about the available knowledge base.
            </div>
            """,
            unsafe_allow_html=True,
        )

    for msg_idx, item in enumerate(current_history):
        with st.chat_message("user"):
            st.write(item["q"])

        with st.chat_message("assistant"):
            # Render as Markdown so validated links copied from source documents are clickable.
            st.markdown(item["result"].answer)

            if item.get("is_new", False):
                for glob_item in ss.chat_history:
                    if glob_item is item:
                        glob_item["is_new"] = False

            # Voice is a user-facing convenience feature. Source references remain hidden.
            if msg_idx == len(current_history) - 1:
                voice_reader_component(item["result"].answer, f"voice_{msg_idx}_{item.get('timestamp', '')}")


# ─────────────────────────────────────────────
# Continue / final feedback flow
# ─────────────────────────────────────────────

def process_pending_question(pending_question: str) -> None:
    """Process a user question after the question bubble has already been rendered.

    This prevents the UI from looking frozen: the user's question appears first,
    then the assistant shows the "Searching knowledge base..." state while the
    backend fetches/generates the answer.
    """
    # Apply hidden, content-aware backend configuration.
    search_mode = apply_backend_config_for_question(pending_question, answer_pipeline)

    # Atomic FAQs are checked before broad heading routing. This prevents a
    # generic policy heading from intercepting operational questions such as a
    # deadline, identifier, correction, access, or withdrawal question. The
    # FAQ layer returns None unless topic and answer-type gates are satisfied.
    result = None
    if should_try_faq_first(pending_question, answer_pipeline) and hasattr(answer_pipeline, "query_faq_only"):
        result = answer_pipeline.query_faq_only(pending_question)

    # Exact clauses/sections/headings remain deterministic. All other questions
    # continue through the full corrective RAG pipeline.
    if result is None:
        result = deterministic_policy_answer(pending_question, answer_pipeline)
    if result is None:
        result = answer_pipeline.query(pending_question, search_mode=search_mode)

    # Clean employee-facing text. Source metadata and evidence commentary stay
    # in backend fields only. A clickable link is shown only when it is present in the relevant retrieved source.
    result.answer = sanitize_answer(result.answer, pending_question)
    result.answer = enrich_answer_with_document_links(result.answer, pending_question, result)

    answer_doc_key = ss.active_doc
    if result.retrieved:
        answer_doc_key = document_key_from_filename(result.retrieved[0].filename) or ss.active_doc

    ss.chat_history.append({
        "doc": answer_doc_key,
        "scope": ss.search_scope,
        "q": pending_question,
        "result": result,
        "is_new": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    # References are not shown in the UI, but are saved in history/export.
    if ss.history_store:
        try:
            ss.history_store.save_exchange(
                session_id=ss.session_id,
                question=pending_question,
                answer=result.answer,
                source=result.search_mode,
                source_file="; ".join(source_refs(result))[:900],
            )
        except Exception:
            # Do not expose technical details in the main UI.
            pass

    if result.retrieved:
        focus_preview_from_result(result)

    ss.processing_question = None
    ss.processing_timestamp = None
    ss.awaiting_continue = True
    ss.chat_closed = False
    ss.session_feedback_saved = False
    ss.pending_query = None
    ss.auto_scroll_bottom = True


# Render the submitted question immediately while the backend is still working.
if ss.get("processing_question") and not ss.chat_closed:
    with st.chat_message("user"):
        st.write(ss.processing_question)

    with st.chat_message("assistant"):
        st.markdown("Searching knowledge base...")

    # Scroll down so the user sees their submitted question and the loading state.
    st.markdown('<div data-scroll-anchor="chat-bottom" style="height:1px;"></div>', unsafe_allow_html=True)
    scroll_to_bottom_component()

    with st.spinner("Searching knowledge base..."):
        process_pending_question(ss.processing_question)

    st.rerun()

elif ss.awaiting_continue and current_history and not ss.chat_closed:
    st.markdown("---")
    st.markdown("Would you like to ask another question?")
    c1, c2, c3 = st.columns([1.2, 1.2, 4])
    with c1:
        if st.button("Yes, continue", use_container_width=True):
            ss.awaiting_continue = False
            ss.auto_scroll_bottom = True
            st.rerun()
    with c2:
        if st.button("No, finish", use_container_width=True):
            ss.awaiting_continue = False
            ss.chat_closed = True
            ss.auto_scroll_bottom = True
            st.rerun()

elif ss.chat_closed:
    st.markdown("---")
    if ss.session_feedback_saved:
        st.success("Thank you. Your session feedback has been saved.")
    else:
        st.markdown("Please rate the overall chat session.")
        ss.session_feedback_comment = st.text_area(
            "Optional feedback",
            value=ss.session_feedback_comment,
            placeholder="Tell us what was correct or what should be improved...",
            label_visibility="collapsed",
            key="session_feedback_comment_box",
        )
        c1, c2, c3 = st.columns([1.2, 1.5, 4])
        with c1:
            if st.button("👍 Good session", use_container_width=True):
                save_session_feedback(1, ss.session_feedback_comment)
                ss.session_feedback_saved = True
                ss.auto_scroll_bottom = True
                st.rerun()
        with c2:
            if st.button("👎 Needs improvement", use_container_width=True):
                save_session_feedback(-1, ss.session_feedback_comment)
                ss.session_feedback_saved = True
                ss.auto_scroll_bottom = True
                st.rerun()

else:
    voice_question_component()

    # Voice question arrives via query param from the mic component.
    voice_q = st.query_params.get("voice_q", "")
    if voice_q:
        st.query_params.clear()

    user_question = st.chat_input("Ask your question...") or voice_q
    if user_question:
        pending_question = user_question.strip()
        if pending_question:
            # Store first, rerun immediately, then render the question bubble before heavy processing.
            ss.processing_question = pending_question
            ss.processing_timestamp = datetime.now().isoformat(timespec="seconds")
            ss.awaiting_continue = False
            ss.chat_closed = False
            ss.session_feedback_saved = False
            ss.auto_scroll_bottom = True
            st.rerun()




# Hidden bottom anchor used by the auto-scroll helper.
st.markdown('<div data-scroll-anchor="chat-bottom" style="height:1px;"></div>', unsafe_allow_html=True)

if ss.get("auto_scroll_bottom") or ss.get("processing_question") or (ss.awaiting_continue and current_history) or ss.chat_closed:
    scroll_to_bottom_component()
    ss.auto_scroll_bottom = False

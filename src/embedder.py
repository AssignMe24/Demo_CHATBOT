from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

_embedding_model: SentenceTransformer | None = None
_rerank_model: CrossEncoder | None = None
_embedding_model_name: str | None = None
_rerank_model_name: str | None = None

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FALLBACK_RERANK_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
SKIP_RERANK = os.getenv("RAG_SKIP_RERANK", "0") == "1"


def _best_device() -> str:
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    except Exception:
        pass
    return "cpu"


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model, _embedding_model_name
    if _embedding_model is None:
        preferred = os.getenv("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL
        fallback = os.getenv("RAG_EMBED_FALLBACK_MODEL", FALLBACK_EMBED_MODEL).strip() or FALLBACK_EMBED_MODEL
        try:
            _embedding_model = SentenceTransformer(preferred, device=_best_device())
            _embedding_model_name = preferred
        except Exception:
            _embedding_model = SentenceTransformer(fallback, device=_best_device())
            _embedding_model_name = fallback
    return _embedding_model


def _get_rerank_model() -> CrossEncoder:
    global _rerank_model, _rerank_model_name
    if _rerank_model is None:
        preferred = os.getenv("RAG_RERANK_MODEL", DEFAULT_RERANK_MODEL).strip() or DEFAULT_RERANK_MODEL
        fallback = os.getenv("RAG_RERANK_FALLBACK_MODEL", FALLBACK_RERANK_MODEL).strip() or FALLBACK_RERANK_MODEL
        try:
            _rerank_model = CrossEncoder(preferred, device=_best_device())
            _rerank_model_name = preferred
        except Exception:
            _rerank_model = CrossEncoder(fallback, device=_best_device())
            _rerank_model_name = fallback
    return _rerank_model


def embedding_model_name() -> str:
    return _embedding_model_name or os.getenv("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL)


def rerank_model_name() -> str:
    return _rerank_model_name or os.getenv("RAG_RERANK_MODEL", DEFAULT_RERANK_MODEL)


def _query_for_model(query: str) -> str:
    name = embedding_model_name().casefold()
    if "bge" in name and "e5" not in name:
        return "Represent this sentence for searching relevant passages: " + query
    if "e5" in name:
        return "query: " + query
    return query


def _documents_for_model(chunks: list[str]) -> list[str]:
    return ["passage: " + c for c in chunks] if "e5" in embedding_model_name().casefold() else chunks


def embed_chunks(chunks: list[str], batch_size: int | None = None) -> np.ndarray:
    model = _get_embedding_model()
    batch_size = batch_size or int(os.getenv("RAG_EMBED_BATCH_SIZE", "16"))
    return np.asarray(model.encode(
        _documents_for_model(chunks), batch_size=max(1, batch_size), convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ), dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    return _embed_query_cached(str(query or "").strip()).copy()


@lru_cache(maxsize=2048)
def _embed_query_cached(query: str) -> np.ndarray:
    model = _get_embedding_model()
    values = model.encode([_query_for_model(query)], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    values.flags.writeable = False
    return values


def rerank_with_scores(query: str, chunks: list[str], top_n: int | None = None) -> tuple[list[int], list[float]]:
    if not chunks:
        return [], []
    if SKIP_RERANK:
        indices = list(range(len(chunks)))
        if top_n is not None:
            indices = indices[:top_n]
        return indices, [1.0] * len(indices)
    raw = np.asarray(_get_rerank_model().predict([[query, chunk] for chunk in chunks], show_progress_bar=False), dtype=np.float32)
    order = np.argsort(raw)[::-1].tolist()
    if top_n is not None:
        order = order[:max(1, min(top_n, len(order)))]
    return order, [float(raw[i]) for i in order]


def rerank_chunks(query: str, chunks: list[str], top_n: int) -> list[int]:
    return rerank_with_scores(query, chunks, top_n=top_n)[0]


def rerank_scores(query: str, chunks: list[str]) -> list[float]:
    order, scores = rerank_with_scores(query, chunks, top_n=None)
    aligned = [0.0] * len(chunks)
    for idx, score in zip(order, scores):
        aligned[idx] = score
    return aligned

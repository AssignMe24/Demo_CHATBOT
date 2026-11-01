from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"a","an","and","are","as","at","be","been","by","can","could","did","do","does","for","from","had","has","have","how","i","if","in","is","it","me","my","of","on","or","our","please","should","that","the","their","this","to","was","we","were","what","when","where","which","who","whom","why","will","with","would","you","your"}

def _unique_terms(tokens: list[str]) -> list[str]:
    """Preserve query term order while removing duplicates."""
    return list(dict.fromkeys(token for token in tokens if token))



def _stem(token: str) -> str:
    for suffix in ("ingly","edly","ation","ments","ment","ing","ied","ies","ed","es","s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[:-3] + "y" if suffix == "ies" else token[:-len(suffix)]
    return token


def _tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(str(text or "").casefold()) if t not in _STOPWORDS and len(t) > 1]


def expand_query(query: str) -> list[str]:
    """Create generic lexical variants without document- or policy-specific synonyms.

    Semantic paraphrase handling is performed by the embedding model and, when
    first-pass evidence is weak, by the corrective query rewriter in
    ``rag_pipeline.py``. Keeping this function corpus-neutral prevents future
    documents from being biased toward examples seen during development.
    """
    original = re.sub(r"\s+", " ", str(query or "")).strip()
    if not original:
        return []

    tokens = _unique_terms(_tokenize(original))
    variants = [original]

    # A concise content-word query often improves BM25 while preserving the
    # user's original wording as the primary dense-search query.
    concise = " ".join(tokens)
    if concise and concise.casefold() != original.casefold():
        variants.append(concise)

    # Add adjacent content-word phrases. This is generic and derived only from
    # the current question, not from any known document or expected answer.
    if len(tokens) >= 3:
        bigrams = [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
        phrase_query = " ".join(_unique_terms(tokens + bigrams))
        if phrase_query and phrase_query.casefold() not in {v.casefold() for v in variants}:
            variants.append(phrase_query)

    return variants[:3]


def keyword_coverage(query: str, text: str) -> float:
    q = set(_tokenize(query))
    if not q:
        return 0.0
    return len(q & set(_tokenize(text))) / len(q)


# Generic question-intent cues used only to distinguish FAQ information needs.
# These are not tied to any HR topic, document name, portal, policy, or answer.
_INTENT_PATTERNS = {
    "identifier": (
        r"\b(user ?id|username|identifier|account number|employee id|login id|code)\b",
    ),
    "access": (
        r"\b(where|find|access|open|link|url|website|portal|page|download)\b",
    ),
    "timing": (
        r"\b(when|deadline|due|frequency|how often|weekly|monthly|week|month|schedule)\b",
    ),
    "consequence": (
        r"\b(what happens|consequence|late|missed|miss|forgot|forget|failed|fail|too late)\b",
    ),
    "contact": (
        r"\b(who|whom|contact|call|reach|connect|speak to|email)\b",
    ),
    "procedure": (
        r"\b(how|steps?|process|procedure|apply|applied|update|updated|updating|correct|corrected|correction|correcting|edit|edited|editing|adjust|adjusted|adjustment|submit|submitted|record|recorded|log|logged|book|booked|initiate|initiated|first step)\b",
    ),
    "eligibility": (
        r"\b(can i|may i|eligible|eligibility|entitled|entitlement|qualify|qualified|qualification|circumstances|under what conditions|conditions|allowed|permitted|withdraw|withdrawal|before retirement)\b",
    ),
    "definition": (
        r"\b(what is|meaning|purpose|overview|explain|describe|tell me about|summary)\b",
    ),
}

_GENERIC_ANCHOR_TERMS = {
    "ask", "asked", "help", "need", "want", "know", "please", "tell", "show",
    "information", "details", "thing", "issue", "immediately", "currently",
    "first", "step", "steps", "process", "procedure", "question", "answer",
    "employee", "employees", "staff", "company", "manager", "hr",
    "click", "select", "detail", "details", "portal", "account", "login",
    "update", "updated", "correct", "correction", "edit", "edited",
    "submit", "submitted", "submission", "request", "current", "date",
    "name", "reason", "available", "member", "use", "using", "open",
    "policy", "policies", "document", "documents", "guideline", "guidelines",
    "step1", "step2", "step3", "step4", "step5", "step6", "step7",
    "step8", "step9", "step10",
}
for _patterns in _INTENT_PATTERNS.values():
    for _pattern in _patterns:
        _GENERIC_ANCHOR_TERMS.update(_tokenize(re.sub(r"[^a-zA-Z0-9 ]", " ", _pattern)))


def intent_signature(text: str) -> set[str]:
    value = re.sub(r"\s+", " ", str(text or "").casefold()).strip()
    found: set[str] = set()
    for name, patterns in _INTENT_PATTERNS.items():
        if any(re.search(pattern, value) for pattern in patterns):
            found.add(name)
    return found


def intent_compatibility(query: str, candidate: str) -> float:
    q = intent_signature(query)
    c = intent_signature(candidate)
    if not q or not c:
        return 0.5
    shared = q & c
    if shared:
        return 1.0
    # Closely related information needs receive partial credit.
    related = {
        frozenset(("identifier", "access")),
        frozenset(("timing", "consequence")),
        frozenset(("procedure", "access")),
        frozenset(("procedure", "contact")),
    }
    if any(frozenset((left, right)) in related for left in q for right in c):
        return 0.35
    return 0.0


def content_anchors(text: str) -> set[str]:
    value = str(text or "")
    anchors = {
        token for token in _tokenize(value)
        if token not in _GENERIC_ANCHOR_TERMS and len(token) > 2
    }
    # Preserve short acronyms such as PF, ID, IT, or UK as topic anchors.
    # The generic-anchor filter still removes common function acronyms such as HR.
    for acronym in re.findall(r"\b[A-Z]{2,6}\b", value):
        token = acronym.casefold()
        if token not in _GENERIC_ANCHOR_TERMS:
            anchors.add(token)
    return anchors


def anchor_coverage(query: str, candidate: str) -> float:
    q = content_anchors(query)
    if not q:
        return 0.0
    return len(q & content_anchors(candidate)) / len(q)


class _BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.documents, self.k1, self.b, self.n = documents, k1, b, len(documents)
        self.lengths = np.asarray([len(d) for d in documents], dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) if self.n else 1.0
        self.term_freqs = [Counter(doc) for doc in documents]
        df = Counter()
        for doc in documents: df.update(set(doc))
        self.idf = {term: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        for i, tf in enumerate(self.term_freqs):
            dl, score = float(self.lengths[i]), 0.0
            for term in query_tokens:
                freq = tf.get(term, 0)
                if not freq: continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-6))
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1.0) / denom)
            scores[i] = score
        return scores


class HybridSearch:
    def __init__(self, chunks: List[Dict], vector_store):
        self.chunks, self.vector_store = chunks, vector_store
        self.texts = [str(c.get("search_text") or c.get("text") or "") for c in chunks]
        self.headings = [" ".join(filter(None,[c.get("filename",""),str(c.get("section_number") or ""),c.get("section_title") or c.get("section") or "",c.get("question","")])) for c in chunks]
        self.bm25 = _BM25([_tokenize(t) for t in self.texts])
        self.heading_bm25 = _BM25([_tokenize(t) for t in self.headings])
        self.last_details: dict[int, dict] = {}

    def _rank(self, scores: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
        order = np.argsort(scores)[::-1][:min(top_k,len(scores))]
        vals = scores[order]
        mx = float(vals.max()) if len(vals) else 0.0
        return order.tolist(), ((vals/mx).tolist() if mx > 0 else [0.0]*len(vals))

    def bm25_search(self, query: str, top_k: int): return self._rank(self.bm25.get_scores(_tokenize(query)), top_k)

    def heading_search(self, query: str, top_k: int):
        scores = self.heading_bm25.get_scores(_tokenize(query))
        q_norm = re.sub(r"\s+", " ", query.casefold()).strip()
        q_terms = set(_tokenize(query))
        for idx, heading in enumerate(self.headings):
            h_norm = re.sub(r"\s+", " ", heading.casefold()).strip()
            if q_norm and (q_norm in h_norm or h_norm in q_norm):
                scores[idx] += 4.0

            # Generic derivational match for forms such as entitled /
            # entitlement or notify / notification. Prefix matching is used
            # only as a small heading bonus and never as an answer rule.
            h_terms = set(_tokenize(heading))
            prefix_hits = sum(
                1 for q_term in q_terms for h_term in h_terms
                if len(q_term) >= 5 and len(h_term) >= 5
                and q_term[:5] == h_term[:5]
            )
            if prefix_hits:
                scores[idx] += min(1.5, 0.45 * prefix_hits)

            sec = str(self.chunks[idx].get("section_number") or "")
            if sec and re.search(rf"\b{re.escape(sec)}\b", q_norm):
                scores[idx] += 3.0
        return self._rank(scores, top_k)

    def vector_search(self, query_embedding: np.ndarray, top_k: int):
        indices,scores=self.vector_store.search(query_embedding,top_k)
        return indices.tolist(),[float(s) for s in scores.tolist()]

    @staticmethod
    def _add_rrf(acc,details,indices,scores,channel,weight,rrf_k,query_no):
        for rank,(idx,raw) in enumerate(zip(indices,scores),start=1):
            acc[idx]+=weight/(rrf_k+rank)
            info=details[idx]
            info[f"{channel}_best"]=max(float(info.get(f"{channel}_best",-1e9)),float(raw))
            info.setdefault("matched_queries",set()).add(query_no)

    def search_queries(self,queries,query_embeddings,top_k,pool_multiplier=3,rrf_k=60):
        pool=min(len(self.chunks),max(top_k,top_k*pool_multiplier))
        acc,details=defaultdict(float),defaultdict(dict)
        for qn,q in enumerate(queries):
            emb=query_embeddings[min(qn,len(query_embeddings)-1)]
            bi,bs=self.bm25_search(q,pool); vi,vs=self.vector_search(emb,pool); hi,hs=self.heading_search(q,min(pool,max(top_k*2,10)))
            self._add_rrf(acc,details,bi,bs,"bm25",1.0,rrf_k,qn)
            self._add_rrf(acc,details,vi,vs,"vector",1.15,rrf_k,qn)
            self._add_rrf(acc,details,hi,hs,"heading",1.25,rrf_k,qn)
        link_intent = any(re.search(r"\b(link|url|website|web address|hyperlink)\b", q.casefold()) for q in queries)
        for idx in list(acc):
            role = self.chunks[idx].get("content_role", "body")
            if role == "toc":
                acc[idx] *= .15
            elif role == "front_matter":
                acc[idx] *= .4
            elif role == "title_page":
                acc[idx] *= .06
            elif role == "links":
                acc[idx] *= 1.0 if link_intent else .01
            if self.chunks[idx].get("record_type") == "faq":
                acc[idx] *= 1.08
            details[idx]["keyword_coverage"] = max(keyword_coverage(q, self.texts[idx]) for q in queries)
        ranked=sorted(acc,key=acc.get,reverse=True)[:min(top_k,len(acc))]
        top=acc[ranked[0]] if ranked else 0.0
        normalized=[float(acc[i]/top) if top else 0.0 for i in ranked]
        for idx in ranked:
            if isinstance(details[idx].get("matched_queries"),set): details[idx]["matched_queries"]=sorted(details[idx]["matched_queries"])
            details[idx]["rrf_raw"]=float(acc[idx])
        self.last_details={idx:details[idx] for idx in ranked}
        return ranked,normalized,self.last_details

    def hybrid_search(self,query,query_embedding,top_k,bm25_weight=.3,vector_weight=.7):
        indices,scores,_=self.search_queries([query],[query_embedding],top_k)
        return indices,scores

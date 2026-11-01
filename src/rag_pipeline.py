from __future__ import annotations

import hashlib,math,os,re,time
from collections import defaultdict
from dataclasses import dataclass,field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict,List,Optional
import numpy as np

import persistence as pr
from answer_generation import (
    NOT_FOUND, answer_support_score, explicit_link_request, generate_answer_full,
    choose_faq_candidate, generate_document_summary, generate_freeform, rewrite_query, sanitize_answer, verify_answer,
)
from chunker import CHUNKER_VERSION,choose_chunk_params,chunk_text
from confidence import ANSWERED_PARTIAL,ANSWERED_SUPPORTED,GENERATION_FAILURE,REFUSED_INSUFFICIENT_EVIDENCE,RETRIEVAL_FAILURE,confidence_color,confidence_label,evidence_confidence,sigmoid_score
from document_loader import LOADER_VERSION,SUPPORTED_EXTENSIONS,load_document_text
from embedder import embedding_model_name,embed_chunks,embed_query,rerank_model_name,rerank_with_scores
from hybrid_search import HybridSearch,anchor_coverage,content_anchors,expand_query,intent_compatibility,keyword_coverage
from vector_store import VectorStore

CACHE_VERSION="rag-cache-v7-clean-answer-routing"
RERANK_POOL_DEFAULT=int(os.getenv("RAG_RERANK_POOL","30"))


def _requested_answer_type(text: str) -> str:
    """Classify the requested response shape using generic language cues."""
    value = re.sub(r"[^a-z0-9\s]", " ", str(text or "").casefold())
    value = re.sub(r"\s+", " ", value).strip()
    patterns = (
        ("identifier", r"\b(username|user name|user id|login id|account id|employee id|identifier|code)\b"),
        ("consequence", r"\b(what happens|consequence|consequences|missed|miss|late|too late|forgot|forget|failed|fail)\b"),
        ("deadline", r"\b(deadline|due date|by when|when must|how long|time limit|how often|frequency|weekly|monthly|every week|month end)\b"),
        ("contact", r"\b(who|whom|contact|call|email|reach out|connect|speak to)\b"),
        ("access", r"\b(access|open|view|see|locate|find|link|url|website|portal|where can i find|where do i find|where can i view|where can i see|where is|where are)\b"),
        ("procedure", r"\b(how do|how can|steps?|procedure|process|update(?:d|ing)?|change(?:d|ing)?|correct(?:ed|ion|ing)?|edit(?:ed|ing)?|adjust(?:ed|ment|ing)?)\b"),
        ("eligibility", r"\b(can i|may i|eligible|eligibility|entitled|entitlements?|rights?|benefits?|qualif(?:y|ied|ication|ications)|who can|circumstances|under what conditions|conditions|allowed|permitted|withdraw|withdrawal|before retirement)\b"),
        ("purpose", r"\b(purpose|scope|objective|overview|meaning|definition|define)\b"),
        ("amount", r"\b(how much|amount|payment|pay|percentage|rate)\b"),
        ("status", r"\b(status|approved|rejected|result|check whether)\b"),
    )
    for label, pattern in patterns:
        if re.search(pattern, value):
            return label
    return "information"


def _is_multi_topic_question(text: str) -> bool:
    value = re.sub(r"[^a-z0-9\s]", " ", str(text or "").casefold())
    value = re.sub(r"\s+", " ", value).strip()
    content = [token for token in value.split() if len(token) > 2]
    return len(content) >= 6 and bool(
        re.search(r"\b(and|plus|as well as|along with|together with)\b", value)
    )


def _initial_action_question(text: str) -> bool:
    value = re.sub(r"[^a-z0-9\s]", " ", str(text or "").casefold())
    value = re.sub(r"\s+", " ", value).strip()
    return bool(re.search(
        r"\b(first|first step|initially|initial action|immediately|right away|as soon as possible|expected to do first)\b",
        value,
    ))


def _initial_phase_score(text: str) -> float:
    """Generic bonus for chunks describing the opening or earliest action phase."""
    value = re.sub(r"\s+", " ", str(text or "").casefold())
    score = 0.0
    if re.search(r"\bwithin\s+\d+\s+(?:business\s+)?(?:hours?|days?|weeks?)\b", value):
        score += 1.0
    if re.search(r"\b(first\s+\d+\s+(?:hours?|days?|weeks?)|immediately|as soon as possible|without delay)\b", value):
        score += 0.9
    if re.search(r"\b(required to|must|should)\b", value):
        score += 0.25
    if re.search(r"(?:^|\s)1[.)]\s+", value):
        score += 0.15
    return min(1.5, score)


def _topic_parts(question: str) -> list[str]:
    """Split an explicit multi-topic question into small focus phrases.

    The splitter is deliberately structural: it uses only conjunctions from the
    current question and contains no corpus- or policy-specific vocabulary.
    """
    raw = re.sub(r"\s+", " ", str(question or "")).strip()
    if not _is_multi_topic_question(raw):
        return []
    parts = [
        part.strip(" ,.;:-")
        for part in re.split(r"\b(?:and|plus|as well as|along with|together with)\b", raw, flags=re.I)
        if part.strip(" ,.;:-")
    ]
    if len(parts) < 2:
        return []
    cleaned = []
    for part in parts[:4]:
        part = re.sub(
            r"^(?:please\s+)?(?:easily\s+)?(?:explain|describe|summarize|summarise|tell me about|give me|show me)\s+(?:me\s+)?",
            "", part, flags=re.I,
        ).strip()
        if len(part.split()) >= 2:
            cleaned.append(part)
    return cleaned if len(cleaned) >= 2 else []


def _topic_search_variants(question: str) -> list[str]:
    return [f"{question} Focus topic: {part}" for part in _topic_parts(question)]


def _broad_document_topic(part: str) -> bool:
    """Return True when a topic part names a document/policy broadly.

    A broad phrase such as "the leave policy" should be represented by an
    overview, purpose, or entitlement section rather than an arbitrary later
    section that happens to repeat the document subject.
    """
    value = re.sub(r"[^a-z0-9\s]", " ", str(part or "").casefold())
    value = re.sub(r"\s+", " ", value).strip()
    return bool(re.search(r"\b(policy|document|guideline|guidelines|overview)\b", value))


def _section_role_bonus(part: str, section: str) -> float:
    """Generic section-role bonus for broad document topic parts.

    Later operational sections such as return-to-work or contact sections are
    checked before generic words such as ``rights``. Otherwise a heading like
    ``RIGHTS ON AND AFTER RETURNING TO WORK`` can incorrectly tie with the
    actual ``ENTITLEMENTS`` section for a broad policy question.
    """
    if not _broad_document_topic(part):
        return 0.0
    part_norm = re.sub(r"\s+", " ", str(part or "").casefold())
    section_norm = re.sub(r"\s+", " ", str(section or "").casefold())

    later_role_terms = ("return", "contact", "breach", "exception", "monitor")
    if re.search(r"\b(return(?:ing)?|contact|breach(?:es)?|exception(?:al)?|monitor(?:ing)?)\b", section_norm) and not any(
        term in part_norm for term in later_role_terms
    ):
        return -0.75

    if re.search(r"\b(entitlement|entitlements|eligibility|qualification|rights)\b", section_norm):
        return 1.30
    if re.search(r"\b(purpose|scope|overview|objectives?)\b", section_norm):
        return 1.05
    return 0.0


def _should_pre_rewrite(text: str) -> bool:
    """Rewrite indirect scenarios and multi-topic questions before retrieval.

    The trigger is based only on generic language shape. It contains no
    document, policy, department, product, or expected-answer vocabulary.
    """
    value = re.sub(r"[^a-z0-9\s]", " ", str(text or "").casefold())
    value = re.sub(r"\s+", " ", value).strip()
    words = value.split()
    first_person_scenario = bool(
        re.search(r"\b(i|i ve|i have|i am|we|my|our|someone|an employee|the employee)\b", value)
    ) and len(words) >= 8
    indirect_action = bool(
        re.search(r"\b(what should|what do|what am|what are|how should|how do|immediately|first)\b", value)
    ) and len(words) >= 8
    specific_intent = (
        _requested_answer_type(value) in {"identifier", "eligibility", "deadline", "consequence"}
        and len(words) >= 6
    )
    return (
        _is_multi_topic_question(value)
        or first_person_scenario
        or indirect_action
        or specific_intent
    )



def _intent_search_variants(question: str) -> list[str]:
    """Add compact, corpus-neutral search language for the requested answer shape."""
    answer_type = _requested_answer_type(question)
    additions = {
        "identifier": "identifier user id username login id account code",
        "deadline": "deadline due date submission frequency timing weekly monthly",
        "consequence": "consequence result missed deadline late failure impact",
        "contact": "contact person responsible team email phone",
        "procedure": "procedure steps correction edit update process",
        "eligibility": "eligibility entitlement qualification requirements criteria",
        "purpose": "purpose scope objective overview definition",
        "amount": "payment amount rate percentage duration",
        "status": "approval rejection status result",
        "access": "access portal link website location",
    }
    phrase = additions.get(answer_type)
    return [f"{question} {phrase}"] if phrase else []


def _incident_search_variants(question: str) -> list[str]:
    """Normalize accidental-disclosure scenarios into generic incident vocabulary.

    This is not tied to a policy or expected answer. It bridges ordinary user
    language such as sharing information with the wrong recipient to common
    security-document terminology.
    """
    value = re.sub(r"[^a-z0-9\s]", " ", str(question or "").casefold())
    value = re.sub(r"\s+", " ", value).strip()
    disclosure = bool(re.search(
        r"\b(accidentally|mistakenly|wrong person|wrong recipient|unintended|unauthori[sz]ed)\b",
        value,
    )) and bool(re.search(
        r"\b(shared|sent|disclosed|emailed|exposed|revealed|information|data|confidential|personal)\b",
        value,
    ))
    if not disclosure:
        return []
    return [
        f"{question} unauthorized disclosure security incident personal data breach report immediately preserve evidence"
    ]


def _application_search_variants(question: str) -> list[str]:
    """Add generic application/required-document vocabulary when explicitly asked.

    This improves recall for questions such as "how do I apply" or "what
    documents must I submit" without embedding any policy-specific terms.
    """
    value = re.sub(r"[^a-z0-9\s]", " ", str(question or "").casefold())
    value = re.sub(r"\s+", " ", value).strip()
    asks_application = bool(re.search(r"\b(apply|application)\b", value))
    asks_documents = bool(
        re.search(r"\b(documents?|proof|certificate)\b", value)
        and re.search(r"\b(need|required|submit|provide|send|attach)\b", value)
    )
    if not (asks_application or asks_documents):
        return []
    return [
        f"{question} application notification requirements documents provide submit proof deadline"
    ]


def _entity_anchor_terms(text: str) -> set[str]:
    """Extract named systems/brands suitable for cross-FAQ aliases.

    Only mixed-case names and tokens that also occur inside a source URL are
    accepted. Ordinary capitalised words, generic nouns, and short acronyms are
    intentionally excluded because they create unsafe cross-topic bridges.
    """
    value = str(text or "")
    generic = {
        "http", "https", "www", "com", "org", "net", "portal", "login",
        "member", "guide", "recording", "policy", "page",
        "document", "details", "account", "service", "personal", "required",
        "time", "check", "get", "use", "refer", "please",
    }
    url_tokens = set()
    for url in re.findall(r"https?://[^\s\]\)<>\"']+", value):
        url_tokens.update(re.findall(r"[a-z0-9]{4,}", url.casefold()))
    terms = {token for token in url_tokens if token not in generic}

    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b", value):
        normalized = token.casefold().strip("_-")
        internal_upper = (
            any(char.isupper() for char in token[1:])
            and any(char.islower() for char in token)
        )
        appears_in_url = normalized in url_tokens
        if normalized not in generic and (internal_upper or appears_in_url):
            terms.add(normalized)
    return terms


def _resource_terms(text: str) -> list[str]:
    """Extract meaningful terms from a short named-resource phrase."""
    value = re.sub(r"https?://[^\s\]\)<>\"']+", " ", str(text or ""))
    value = re.sub(r"\[[^\]]+\]\([^\)]+\)", lambda m: m.group(0).split("](", 1)[0].lstrip("["), value)
    terms = re.findall(r"[a-z0-9]+", value.casefold())
    generic = {
        "a", "an", "and", "are", "about", "can", "could", "details", "do", "does",
        "find", "for", "from", "get", "give", "how", "i", "in", "is", "it", "link",
        "me", "of", "on", "open", "please", "refer", "see", "show", "tell", "the",
        "this", "to", "view", "website", "where", "which", "with", "you", "your",
        "access", "page", "portal", "resource", "information",
        "policy", "document", "guideline", "guidelines", "guide",
    }
    return [term for term in terms if len(term) > 1 and term not in generic]


def _answer_resource_labels(answer: str) -> list[str]:
    """Derive resource names only from the source FAQ answer."""
    value = str(answer or "")
    labels = [
        re.sub(r"\s+", " ", label).strip()
        for label in re.findall(r"\[([^\]]{2,160})\]\(https?://[^\)]+\)", value)
    ]
    plain = re.sub(r"\[[^\]]+\]\([^\)]+\)", " ", value)
    plain = re.sub(r"https?://[^\s\]\)<>\"']+", " ", plain)
    plain = re.sub(
        r"^\s*(?:please\s+)?(?:refer|use|open|view|see|visit|access)\s+(?:to\s+)?",
        "",
        plain,
        flags=re.I,
    ).strip(" .:-")
    if 2 <= len(plain) <= 180:
        labels.append(plain)
    return list(dict.fromkeys(label for label in labels if label))


def _approximate_term_coverage(source_terms: list[str], target_terms: list[str]) -> float:
    """Return source-term coverage with conservative typo tolerance."""
    if not source_terms or not target_terms:
        return 0.0
    matched = 0
    used: set[int] = set()
    for source in source_terms:
        best_index = None
        best_score = 0.0
        for index, target in enumerate(target_terms):
            if index in used:
                continue
            score = 1.0 if source == target else SequenceMatcher(None, source, target).ratio()
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= 0.84:
            matched += 1
            used.add(best_index)
    return matched / len(source_terms)


def _resource_name_score(question: str, faq_question: str, faq_answer: str) -> float:
    """Score a named resource using only corpus text and the user question.

    This handles short requests such as "tell me about <resource>" and permits
    one small spelling error. No resource names or domain-specific aliases are
    embedded in the runtime.
    """
    q_terms = _resource_terms(question)
    if not q_terms or len(q_terms) > 9:
        return 0.0

    candidates = [faq_question, *_answer_resource_labels(faq_answer)]
    best = 0.0
    for candidate in candidates:
        c_terms = _resource_terms(candidate)
        if not c_terms:
            continue
        # Compare the shorter meaningful phrase against the longer one so
        # request wrappers and FAQ grammar do not reduce the score.
        source = q_terms if len(q_terms) <= len(c_terms) else c_terms
        target = c_terms if source is q_terms else q_terms
        coverage = _approximate_term_coverage(source, target)
        sequence = SequenceMatcher(
            None,
            " ".join(q_terms),
            " ".join(c_terms),
        ).ratio()
        score = 0.75 * coverage + 0.25 * sequence
        best = max(best, score)
    return max(0.0, min(1.0, best))



def _strong_resource_faq_match(question: str, faq_chunks: list[dict]) -> tuple[dict, float] | None:
    """Resolve a uniquely named FAQ resource without model-score dependence.

    The resource name is derived only from indexed FAQ questions, hyperlink
    labels, and answer text. A conservative score and ranking margin are
    required, so generic requests cannot select an unrelated access FAQ.
    """
    requested_type = _requested_answer_type(question)
    if requested_type not in {"information", "access"}:
        return None

    q_terms = _resource_terms(question)
    if not q_terms or len(q_terms) > 9:
        return None
    if len(q_terms) == 1 and len(q_terms[0]) < 5:
        return None

    ranked: list[tuple[float, dict]] = []
    for chunk in faq_chunks:
        if chunk.get("record_type") != "faq" or not str(chunk.get("answer") or "").strip():
            continue
        faq_question = str(chunk.get("question") or "")
        faq_answer = str(chunk.get("answer") or "")
        candidate_type = _requested_answer_type(faq_question)
        if candidate_type not in {"information", "access"}:
            continue
        score = _resource_name_score(question, faq_question, faq_answer)
        if score > 0:
            ranked.append((score, chunk))

    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best_chunk = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0

    minimum = 0.86 if len(q_terms) >= 2 else 0.94
    margin = best_score - second_score
    if best_score < minimum:
        return None
    if margin < 0.16 and best_score < 0.97:
        return None
    return best_chunk, best_score


def _faq_alias_texts(faq_chunks: list[dict]) -> dict[int, str]:
    """Build safe corpus-derived aliases from rare named-entity bridges.

    A bridge is created only when a named system, acronym, product, or proper
    noun appearing in one FAQ answer also occurs in another FAQ question.
    Generic procedural or topical words cannot create aliases, preventing
    unrelated FAQs from being connected merely by words such as update,
    medical, details, deadline, or correction.
    """
    question_anchors = [content_anchors(str(c.get("question") or "")) for c in faq_chunks]
    answer_entities = [_entity_anchor_terms(str(c.get("answer") or "")) for c in faq_chunks]
    entity_frequency = defaultdict(int)
    for values in answer_entities:
        for term in values:
            entity_frequency[term] += 1

    aliases: dict[int, list[str]] = {i: [] for i in range(len(faq_chunks))}
    for target_index, target in enumerate(faq_chunks):
        target_terms = question_anchors[target_index] | _entity_anchor_terms(str(target.get("question") or ""))
        for source_index, source in enumerate(faq_chunks):
            if target_index == source_index:
                continue
            bridge = {
                term for term in target_terms & answer_entities[source_index]
                if len(term) >= 2 and entity_frequency.get(term, 99) <= 4
            }
            if not bridge:
                continue
            source_question = str(source.get("question") or "").strip()
            if source_question and source_question not in aliases[target_index]:
                aliases[target_index].append(source_question)
            if len(aliases[target_index]) >= 3:
                break
    return {index: " ".join(values) for index, values in aliases.items()}



def _mentioned_document_filenames(question: str, chunks: list[dict]) -> set[str]:
    """Return indexed filenames explicitly named in the user question.

    Matching is derived only from the filenames currently present in the index.
    Generic file words are ignored. This prevents a question that names a
    policy/document from being diverted to an unrelated FAQ in another file.
    """
    raw = str(question or "")
    # Split CamelCase filenames before token comparison.
    q_terms = set(re.findall(r"[a-z0-9]+", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw).casefold()))
    generic = {
        "policy", "policies", "document", "documents", "guideline", "guidelines",
        "final", "draft", "version", "issue", "copy", "pdf", "docx", "txt",
        "file", "the", "a", "an", "of", "and", "for", "to", "in", "on",
    }
    q_terms -= generic
    if not q_terms:
        return set()

    filenames = {str(chunk.get("filename") or "") for chunk in chunks if str(chunk.get("filename") or "")}
    matched: set[str] = set()
    for filename in filenames:
        stem = Path(filename).stem
        stem = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
        name_terms = {
            term for term in re.findall(r"[a-z0-9]+", stem.casefold())
            if term not in generic and not term.isdigit()
        }
        if not name_terms:
            continue
        overlap = q_terms & name_terms
        required = 1 if len(name_terms) == 1 else min(2, len(name_terms))
        if len(overlap) >= required:
            matched.add(filename)
    return matched

def _heading_type_match(question: str, heading: str) -> float:
    requested = _requested_answer_type(question)
    if requested == "information":
        return 0.0
    heading_type = _requested_answer_type(heading)
    return 1.0 if heading_type == requested else 0.0

@dataclass
class RetrievedChunk:
    chunk_id:int;text:str;page:int;section:Optional[str];vector_score:float;rerank_score:float=0.0;label:str="";source_type:str="document";filename:str="";parent_id:str="";record_type:str="policy"
@dataclass
class RAGResult:
    answer:str;verification:str;retrieved:List[RetrievedChunk];confidence_score:float;confidence_label:str;confidence_color:str;search_mode:str;elapsed_s:float;pdf_name:str;page_scores:Dict[int,float]=field(default_factory=dict)
@dataclass
class _Candidate:
    chunk:dict;search_score:float;vector_score:float;bm25_score:float;heading_score:float;coverage:float;intent_score:float=0.0;rerank_score:float=-12.0

class RAGPipeline:
    def __init__(self,chunks,vector_store,pdf_name,cfg,summary="",source_type="document"):
        self.chunks=chunks;self.vector_store=vector_store;self.pdf_name=pdf_name;self.cfg=cfg;self.summary=summary;self.source_type=source_type;self.hybrid=HybridSearch(chunks,vector_store)

    def _refusal(self,started,mode,status=REFUSED_INSUFFICIENT_EVIDENCE,score=0.0):
        score=max(0.0,min(1.0,float(score)))
        return RAGResult(NOT_FOUND,status,[],score,confidence_label(score),confidence_color(score),mode,time.perf_counter()-started,self.pdf_name)

    def _search(self, queries, candidate_k, search_mode, include_faq=True):
        if not queries:
            return []
        embeddings = [embed_query(q) for q in queries]
        if search_mode == "vector":
            acc, details = defaultdict(float), defaultdict(dict)
            for emb in embeddings:
                ids, scores = self.vector_store.search(emb, min(candidate_k * 3, len(self.chunks)))
                for rank, (idx, score) in enumerate(zip(ids.tolist(), scores.tolist()), 1):
                    acc[idx] += 1 / (60 + rank)
                    details[idx]["vector_best"] = max(float(details[idx].get("vector_best", -1)), float(score))
            ranked = sorted(acc, key=acc.get, reverse=True)[:candidate_k]
            top = acc[ranked[0]] if ranked else 0
            search_scores = [acc[i] / top if top else 0 for i in ranked]
        elif search_mode == "bm25":
            acc, details = defaultdict(float), defaultdict(dict)
            for q in queries:
                ids, scores = self.hybrid.bm25_search(q, min(candidate_k * 3, len(self.chunks)))
                for rank, (idx, score) in enumerate(zip(ids, scores), 1):
                    acc[idx] += 1 / (60 + rank)
                    details[idx]["bm25_best"] = max(float(details[idx].get("bm25_best", 0)), float(score))
            ranked = sorted(acc, key=acc.get, reverse=True)[:candidate_k]
            top = acc[ranked[0]] if ranked else 0
            search_scores = [acc[i] / top if top else 0 for i in ranked]
        else:
            ranked, search_scores, details = self.hybrid.search_queries(
                queries, embeddings, min(candidate_k, len(self.chunks)), 3
            )

        allow_links = any(explicit_link_request(q) for q in queries)
        metadata_intent = any(
            re.search(r"\b(document title|title of|document name|cover page)\b", q.casefold())
            for q in queries
        )

        out = []
        for idx, search_score in zip(ranked, search_scores):
            d = details.get(idx, {})
            chunk = self.chunks[idx]
            role = str(chunk.get("content_role") or "body")

            # FAQ records are handled in a separate atomic FAQ stage. They are
            # excluded from policy generation to prevent one FAQ question from
            # being combined with another FAQ's answer.
            if not include_faq and chunk.get("record_type") == "faq":
                continue
            if role == "links" and not allow_links:
                continue
            if role == "title_page" and not metadata_intent:
                continue

            coverage = float(
                d.get(
                    "keyword_coverage",
                    max(
                        keyword_coverage(q, chunk.get("search_text") or chunk.get("text", ""))
                        for q in queries
                    ),
                )
            )
            candidate = _Candidate(
                chunk,
                float(search_score),
                float(d.get("vector_best", 0)),
                float(d.get("bm25_best", 0)),
                float(d.get("heading_best", 0)),
                coverage,
            )
            candidate.matched_queries = tuple(d.get("matched_queries") or ())
            out.append(candidate)
        return out

    @staticmethod
    def _route_documents(candidates,max_documents=2):
        if not candidates:return []
        scores=defaultdict(float)
        for rank,c in enumerate(candidates,1):
            f=c.chunk.get("filename") or "__unknown__";scores[f]+=c.search_score/math.sqrt(rank);scores[f]+=.03 if c.chunk.get("record_type")=="faq" else 0
        files=sorted(scores.items(),key=lambda x:x[1],reverse=True);selected={files[0][0]}
        if len(files)>1 and files[1][1]>=files[0][1]*.78 and max_documents>1:selected.add(files[1][0])
        return [c for c in candidates if (c.chunk.get("filename") or "__unknown__") in selected]

    def _rerank(self,question,candidates,rerank_k,final_k):
        if not candidates:return []
        pool=candidates[:max(final_k,min(rerank_k,len(candidates)))]
        texts=["\n".join(filter(None,[c.chunk.get("filename",""),c.chunk.get("section_title") or c.chunk.get("section") or "",c.chunk.get("question",""),c.chunk.get("text","")])) for c in pool]
        order,scores=rerank_with_scores(question,texts,len(pool))
        ranked=[]
        for idx,score in zip(order,scores):
            candidate=pool[idx]
            candidate.rerank_score=float(score)
            heading=str(candidate.chunk.get("section_title") or candidate.chunk.get("section") or "")
            candidate.heading_type_match=_heading_type_match(question, heading)
            candidate.rerank_priority=(
                .58*sigmoid_score(candidate.rerank_score)
                +.32*float(candidate.heading_type_match)
                +.10*max(0.0,min(1.0,float(candidate.heading_score)))
            )
            ranked.append(candidate)
        ranked.sort(key=lambda c: float(getattr(c,"rerank_priority",0.0)), reverse=True)
        return ranked[:max(final_k,min(len(ranked),final_k+3))]

    @staticmethod
    def _candidate_strength(c):
        vector=max(0,min(1,(c.vector_score+1)/2))
        return max(0,min(1,.27*sigmoid_score(c.rerank_score)+.20*vector+.16*c.search_score+.13*c.coverage+.08*c.heading_score+.16*float(getattr(c,"heading_type_match",0.0))))

    def _corrective_candidates(self, question, base_queries, cfg, search_mode):
        first_search = self._search(
            base_queries, int(cfg["candidate_k"]), search_mode, include_faq=False
        )
        first = self._rerank(
            question,
            self._route_documents(first_search),
            int(cfg.get("rerank_k", RERANK_POOL_DEFAULT)),
            int(cfg["final_k"]),
        )
        if first and self._candidate_strength(first[0]) >= float(cfg.get("evidence_threshold", .38)):
            return first

        rewritten = rewrite_query(question)
        queries = list(dict.fromkeys(base_queries + rewritten))[:5]
        if len(queries) == len(base_queries):
            return first

        second_search = self._search(
            queries,
            min(len(self.chunks), int(cfg["candidate_k"]) + 12),
            search_mode,
            include_faq=False,
        )
        second = self._route_documents(second_search)
        return self._rerank(
            question,
            second,
            min(len(second), int(cfg.get("rerank_k", RERANK_POOL_DEFAULT)) + 8),
            int(cfg["final_k"]),
        )

    def _strong_resource_candidate(self, question: str):
        faq_chunks = [
            chunk for chunk in self.chunks
            if chunk.get("record_type") == "faq" and chunk.get("answer")
        ]
        matched = _strong_resource_faq_match(question, faq_chunks)
        if matched is None:
            return None
        chunk, resource_score = matched
        faq_question = str(chunk.get("question") or "")
        faq_answer = str(chunk.get("answer") or "")
        question_lexical = keyword_coverage(question, faq_question)
        answer_lexical = keyword_coverage(question, faq_answer)
        anchor_score = anchor_coverage(question, faq_question)
        candidate_type = _requested_answer_type(faq_question)
        requested_type = _requested_answer_type(question)
        type_match = 1.0 if candidate_type == requested_type else 0.0

        candidate = _Candidate(
            chunk,
            min(1.0, 0.72 + 0.24 * resource_score),
            min(1.0, resource_score),
            resource_score,
            resource_score,
            max(question_lexical, answer_lexical, resource_score),
            max(0.5, intent_compatibility(question, faq_question)),
            4.0,
        )
        candidate.anchor_score = anchor_score
        candidate.question_lexical = question_lexical
        candidate.answer_lexical = answer_lexical
        candidate.alias_text = ""
        candidate.alias_lexical = 0.0
        candidate.resource_score = resource_score
        candidate.type_match = type_match
        candidate.type_conflict = False
        candidate.selected_by_resource_match = True
        return candidate


    def _faq_lookup(self, question, queries, cfg):
        faq = [c for c in self.chunks if c.get("record_type") == "faq" and c.get("answer")]
        if not faq or not cfg.get("use_faq_layer", True):
            return []

        queries = list(dict.fromkeys(q for q in queries if str(q).strip()))[:5]
        qemb = [embed_query(q).reshape(-1) for q in queries]
        user_answer_type = _requested_answer_type(question)
        alias_texts = _faq_alias_texts(faq)
        pre = []
        for faq_index, chunk in enumerate(faq):
            try:
                vector_score = max(
                    float(np.dot(q, self.vector_store.reconstruct(int(chunk.get("chunk_id", -1)))))
                    for q in qemb
                )
            except Exception:
                vector_score = 0.0

            faq_question = str(chunk.get("question") or "")
            faq_answer = str(chunk.get("answer") or "")
            full = str(chunk.get("search_text") or chunk.get("text") or "")
            # Expanded queries are useful for semantic recall, but they must not
            # establish topical safety. Generic intent expansions such as
            # "consequence deadline failure" can otherwise make an unrelated FAQ
            # look lexically grounded. All acceptance gates below therefore use
            # only the user's original wording; rewritten queries affect vector
            # recall only.
            lexical = max(keyword_coverage(q, full) for q in queries)
            question_lexical = keyword_coverage(question, faq_question)
            answer_lexical = keyword_coverage(question, faq_answer)
            fuzzy = SequenceMatcher(None, question.casefold(), faq_question.casefold()).ratio()
            intent_match = intent_compatibility(question, faq_question)
            alias_text = alias_texts.get(faq_index, "")
            anchor_match = anchor_coverage(question, f"{faq_question} {alias_text}")
            alias_lexical = keyword_coverage(question, alias_text) if alias_text else 0.0
            resource_score = _resource_name_score(question, faq_question, faq_answer)

            candidate_answer_type = _requested_answer_type(faq_question)
            type_match = 1.0 if candidate_answer_type == user_answer_type else 0.0
            type_conflict = (
                user_answer_type != "information"
                and candidate_answer_type != "information"
                and candidate_answer_type != user_answer_type
            )
            vector_norm = max(0.0, min(1.0, (vector_score + 1.0) / 2.0))
            pre_score = (
                .24 * vector_norm + .17 * question_lexical + .09 * fuzzy
                + .11 * intent_match + .11 * anchor_match + .14 * type_match
                + .20 * resource_score + .02 * alias_lexical
                + .01 * lexical + .01 * answer_lexical
                - (.22 if type_conflict else 0.0)
            )
            candidate = _Candidate(
                chunk, pre_score, vector_score, lexical, fuzzy,
                max(lexical, question_lexical), intent_match,
            )
            candidate.anchor_score = anchor_match
            candidate.question_lexical = question_lexical
            candidate.answer_lexical = answer_lexical
            candidate.alias_text = alias_text
            candidate.alias_lexical = alias_lexical
            candidate.resource_score = resource_score
            candidate.type_match = type_match
            candidate.type_conflict = type_conflict
            pre.append(candidate)

        pre.sort(key=lambda candidate: candidate.search_score, reverse=True)

        # For a specific requested answer shape, prefer candidates with the same
        # shape when they have at least minimal semantic or topic support. This
        # prevents an access-link FAQ from defeating an identifier FAQ merely
        # because both mention the same portal or system.
        typed_pool = []
        if user_answer_type != "information":
            for candidate in pre:
                vector_norm = max(0.0, min(1.0, (float(candidate.vector_score) + 1.0) / 2.0))
                plausible = (
                    float(getattr(candidate, "type_match", 0.0)) >= 1.0
                    and (
                        float(getattr(candidate, "anchor_score", 0.0)) >= .08
                        or candidate.coverage >= .12
                        or vector_norm >= .58
                        or candidate.intent_score >= .80
                    )
                )
                if plausible:
                    typed_pool.append(candidate)
        pool_source = typed_pool if typed_pool else pre
        pool = pool_source[:min(16, len(pool_source))]
        texts = [
            "\n".join(filter(None, [
                f"Candidate information need: {candidate.chunk.get('question', '')}",
                f"Related corpus terminology: {getattr(candidate, 'alias_text', '')}",
            ]))
            for candidate in pool
        ]
        # Rerank the candidate information need against the original question.
        # Search rewrites can improve recall, but injecting them here can change
        # the user's topic and produce unsafe FAQ matches.
        rerank_query = question

        order, scores = rerank_with_scores(rerank_query, texts, len(pool))
        ranked = []
        for idx, score in zip(order, scores):
            pool[idx].rerank_score = float(score)
            ranked.append(pool[idx])
        ranked.sort(
            key=lambda candidate: (
                1 if float(getattr(candidate, "resource_score", 0.0)) >= .68 else 0,
                1 if (
                    float(getattr(candidate, "anchor_score", 0.0)) >= .06
                    or float(getattr(candidate, "question_lexical", 0.0)) >= .18
                    or float(getattr(candidate, "answer_lexical", 0.0)) >= .28
                ) else 0,
                self._faq_strength(candidate),
            ),
            reverse=True,
        )

        # The local model may reorder genuinely close candidates, but it cannot
        # override the deterministic topic and intent gate in _direct_faq().
        if len(ranked) > 1 and float(getattr(ranked[0], "resource_score", 0.0)) < .68:
            first_strength = self._faq_strength(ranked[0])
            second_strength = self._faq_strength(ranked[1])
            question_match = keyword_coverage(question, ranked[0].chunk.get("question", ""))
            if (first_strength - second_strength) < .075 or question_match < .34:
                selected_index = choose_faq_candidate(
                    question,
                    [candidate.chunk.get("question", "") for candidate in ranked[:6]],
                    [candidate.chunk.get("answer", "") for candidate in ranked[:6]],
                )
                if selected_index is not None and 0 <= selected_index < min(6, len(ranked)):
                    selected = ranked.pop(selected_index)
                    selected.selected_by_disambiguator = True
                    ranked.insert(0, selected)
        return ranked

    @staticmethod
    def _faq_strength(c):
        return max(0.0, min(1.0,
            .31 * sigmoid_score(c.rerank_score)
            + .13 * c.search_score
            + .12 * c.intent_score
            + .09 * float(getattr(c, "anchor_score", 0.0))
            + .12 * float(getattr(c, "type_match", 0.0))
            + .17 * float(getattr(c, "resource_score", 0.0))
            + .06 * c.coverage
        ))

    @staticmethod
    def _direct_faq(question, candidates, threshold):
        if not candidates or candidates[0].chunk.get("record_type") != "faq" or not candidates[0].chunk.get("answer"):
            return None
        top = candidates[0]
        strength = RAGPipeline._faq_strength(top)
        margin = strength - (RAGPipeline._faq_strength(candidates[1]) if len(candidates) > 1 else 0.0)
        candidate_question = str(top.chunk.get("question", ""))
        candidate_answer = str(top.chunk.get("answer", ""))
        question_match = keyword_coverage(question, candidate_question)
        answer_match = keyword_coverage(question, candidate_answer)
        anchor_match = max(
            float(getattr(top, "anchor_score", 0.0)),
            anchor_coverage(question, candidate_question),
        )
        intent_match = intent_compatibility(question, candidate_question)
        semantic = sigmoid_score(top.rerank_score)
        vector_norm = max(0.0, min(1.0, (float(top.vector_score) + 1.0) / 2.0))
        type_match = float(getattr(top, "type_match", 0.0))
        type_conflict = bool(getattr(top, "type_conflict", False))
        same_answer = bool(
            len(candidates) > 1
            and candidate_answer.strip()
            == str(candidates[1].chunk.get("answer", "")).strip()
        )

        # A FAQ must share a substantive topic anchor. Generic question words
        # such as who/how/contact/portal cannot by themselves justify a match.
        # Corpus-derived aliases can supply the anchor when two related FAQ
        # records use different names for the same entity.
        typed_lexical_topic = (
            type_match >= 1.0
            and intent_match >= .80
            and question_match >= .24
            and vector_norm >= .60
        )
        weak_typed_topic = (
            type_match >= 1.0
            and question_match >= .18
            and semantic >= .68
            and vector_norm >= .60
            and margin >= .010
        )
        # Matching only the requested answer shape (for example, consequence or
        # contact) is never enough. The user's original words must also share a
        # substantive anchor, strongly overlap the FAQ question, or overlap the
        # FAQ answer. This blocks unrelated consequence/contact FAQs while still
        # allowing corpus-derived aliases such as payroll portal -> product name.
        requested_type = _requested_answer_type(question)
        candidate_type = _requested_answer_type(candidate_question)

        # Deadline questions often contain only temporal wording (for example,
        # "every week or month end") while the canonical FAQ question says only
        # "submission deadline". In that case there may be no durable entity
        # anchor, so use overlap with the source answer's timing terms together
        # with matching intent and strong semantic/vector evidence. This remains
        # stricter than answer-type matching alone and does not relax contact or
        # consequence FAQ safety.
        temporal_deadline_topic = (
            requested_type == "deadline"
            and candidate_type == "deadline"
            and type_match >= 1.0
            and intent_match >= .80
            and answer_match >= .25
            and semantic >= .68
            and vector_norm >= .60
        )

        # A short "tell me about <named resource>" question may intentionally
        # target an access/link FAQ even though its grammatical answer type is
        # generic information. Require a strong resource-name match so this
        # exception cannot admit unrelated access FAQs.
        resource_score = float(getattr(top, "resource_score", 0.0))
        resource_name_match = (
            requested_type in {"information", "access"}
            and candidate_type in {"information", "access"}
            and resource_score >= .68
            and (anchor_match >= .20 or question_match >= .20)
            and semantic >= .58
            and vector_norm >= .52
        )

        topic_grounded = (
            anchor_match >= .06
            or answer_match >= .30
            or question_match >= .24
            or temporal_deadline_topic
            or resource_name_match
        )
        intent_grounded = (
            intent_match >= .35
            or type_match >= 1.0
            or semantic >= .78
            or resource_name_match
        )
        disambiguated = bool(getattr(top, "selected_by_disambiguator", False))
        rank_reliable = (
            anchor_match >= .08
            or question_match >= .30
            or weak_typed_topic
            or typed_lexical_topic
            or (type_match >= 1.0 and semantic >= .68 and vector_norm >= .62 and margin >= .012)
            or temporal_deadline_topic
            or resource_name_match
            or (disambiguated and type_match >= 1.0 and (anchor_match >= .06 or question_match >= .24 or answer_match >= .30))
            or same_answer
        )
        if type_conflict and type_match < 1.0 and semantic < .82:
            rank_reliable = False

        reliable = topic_grounded and intent_grounded and rank_reliable
        return top if strength >= float(threshold) and reliable else None

    @staticmethod
    def _is_broad_question(question: str) -> bool:
        return bool(re.search(
            r"\b(summary|summarize|summarise|overview|entire|whole|full|all sections|compare|comparison|across documents)\b",
            str(question or "").casefold(),
        ))

    def _select_coherent_candidates(self, question, candidates, final_k):
        """Select focused evidence while preserving explicitly requested topics.

        A direct question stays within one parent section. A multi-topic or broad
        question receives a small, diverse set of parent sections so one strong
        section cannot suppress the other requested topic.
        """
        if not candidates:
            return []

        grouped = defaultdict(list)
        for candidate in candidates:
            chunk = candidate.chunk
            key = (
                chunk.get("filename") or "__unknown__",
                chunk.get("parent_id") or f"chunk::{chunk.get('chunk_id', 0)}",
            )
            grouped[key].append(candidate)

        def group_score(items):
            strengths = sorted((self._candidate_strength(item) for item in items), reverse=True)
            matched_queries = set()
            for item in items:
                matched_queries.update(getattr(item, "matched_queries", ()) or ())
            return (
                strengths[0]
                + .16 * sum(strengths[1:3])
                + .10 * max(item.coverage for item in items)
                + .06 * max(item.search_score for item in items)
                + .035 * min(3, len(matched_queries))
            )

        ranked_groups = sorted(
            grouped.values(), key=group_score, reverse=True
        )
        multi_topic = _is_multi_topic_question(question)
        broad = self._is_broad_question(question)

        if not multi_topic and not broad:
            if _initial_action_question(question):
                # Initial-action questions may have one formal deadline block and
                # one opening instruction block in the same document. Select at
                # most two strongly phased parents so parallel opening actions
                # are not lost, while excluding later-stage policy sections.
                phased_groups = sorted(
                    ranked_groups,
                    key=lambda items: (
                        group_score(items)
                        + .22 * max(_initial_phase_score(item.chunk.get("text", "")) for item in items),
                        max(_initial_phase_score(item.chunk.get("text", "")) for item in items),
                    ),
                    reverse=True,
                )
                selected = []
                selected_ids = set()
                selected_file = phased_groups[0][0].chunk.get("filename") if phased_groups else None
                for items in phased_groups[:4]:
                    if selected_file and items[0].chunk.get("filename") != selected_file:
                        continue
                    phase = max(_initial_phase_score(item.chunk.get("text", "")) for item in items)
                    if selected and phase < .20:
                        continue
                    for item in sorted(
                        items,
                        key=lambda candidate: (
                            _initial_phase_score(candidate.chunk.get("text", "")),
                            self._candidate_strength(candidate),
                            candidate.search_score,
                        ),
                        reverse=True,
                    ):
                        cid = int(item.chunk.get("chunk_id", -1))
                        if cid not in selected_ids:
                            selected.append(item)
                            selected_ids.add(cid)
                            break
                    if len(selected) >= min(2, final_k):
                        break
                if selected:
                    for item in sorted(
                        phased_groups[0],
                        key=lambda candidate: (self._candidate_strength(candidate), candidate.search_score),
                        reverse=True,
                    ):
                        cid = int(item.chunk.get("chunk_id", -1))
                        if cid not in selected_ids and len(selected) < final_k:
                            selected.append(item)
                            selected_ids.add(cid)
                    return selected[:max(1, final_k)]

            best_items = sorted(
                ranked_groups[0],
                key=lambda item: (self._candidate_strength(item), item.search_score),
                reverse=True,
            )
            return best_items[:max(1, final_k)]

        # Explicit multi-topic questions are matched topic-part by topic-part.
        # This prevents a broad phrase such as "the policy" from selecting a
        # later return/contact section merely because it repeats the document
        # subject. Each requested topic receives one coherent parent section.
        topic_parts = _topic_parts(question) if multi_topic else []
        if topic_parts and not broad:
            selected = []
            selected_ids = set()
            used_groups = set()
            for part in topic_parts:
                scored_groups = []
                for group_index, items in enumerate(ranked_groups):
                    if group_index in used_groups:
                        continue
                    first_chunk = items[0].chunk
                    section = str(first_chunk.get("section_title") or first_chunk.get("section") or "")
                    filename = str(first_chunk.get("filename") or "")
                    opening = " ".join(str(item.chunk.get("text") or "")[:700] for item in items[:2])
                    heading_target = section
                    title_coverage = keyword_coverage(part, heading_target)
                    topic_anchor = anchor_coverage(part, f"{section} {opening}")
                    filename_coverage = keyword_coverage(part, filename)
                    value = (
                        .32 * group_score(items)
                        + 1.35 * title_coverage
                        + 1.10 * topic_anchor
                        + .12 * filename_coverage
                        + _section_role_bonus(part, section)
                    )
                    scored_groups.append((value, group_index, items))
                if not scored_groups:
                    continue
                scored_groups.sort(key=lambda row: row[0], reverse=True)
                _value, group_index, items = scored_groups[0]
                used_groups.add(group_index)
                chosen = max(
                    items,
                    key=lambda item: (self._candidate_strength(item), item.search_score),
                )
                cid = int(chosen.chunk.get("chunk_id", -1))
                if cid not in selected_ids:
                    selected.append(chosen)
                    selected_ids.add(cid)
                if len(selected) >= final_k:
                    break
            if selected:
                return selected[:max(1, min(final_k, len(selected)))]

        # Greedily cover distinct rewritten/original queries and parent sections.
        selected = []
        selected_ids = set()
        covered_queries = set()
        remaining = list(ranked_groups)
        max_groups = min(4 if broad else 3, final_k, len(remaining))
        while remaining and len(selected) < max_groups:
            best_index = 0
            best_value = -1.0
            for index, items in enumerate(remaining):
                query_hits = set()
                for item in items:
                    query_hits.update(getattr(item, "matched_queries", ()) or ())
                new_hits = len(query_hits - covered_queries)
                value = group_score(items) + .10 * new_hits
                if value > best_value:
                    best_value = value
                    best_index = index
            items = remaining.pop(best_index)
            items = sorted(
                items,
                key=lambda item: (self._candidate_strength(item), item.search_score),
                reverse=True,
            )
            chosen = items[0]
            selected.append(chosen)
            selected_ids.add(int(chosen.chunk.get("chunk_id", -1)))
            covered_queries.update(getattr(chosen, "matched_queries", ()) or ())

        # For an explicit multi-topic question, the groups chosen above already
        # cover the requested focus phrases. Do not fill the remaining slots with
        # unrelated high-scoring sections from the same document (for example, a
        # return-to-work section when the user asked only about leave and pay).
        if multi_topic and not broad and selected:
            return selected[:max(1, min(final_k, len(selected)))]

        # Fill remaining slots with the strongest chunks from already selected
        # source parents, then with other high-quality candidates.
        for candidate in candidates:
            cid = int(candidate.chunk.get("chunk_id", -1))
            if cid in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(cid)
            if len(selected) >= final_k:
                break
        return selected[:max(1, final_k)]

    @staticmethod
    def _same_parent(a,b):return a.get("filename")==b.get("filename") and a.get("parent_id") and a.get("parent_id")==b.get("parent_id")

    def _expand_context(self,top_chunks,window=1,include_parent=False):
        seen=set();expanded=[];by_id={int(c["chunk_id"]):c for c in self.chunks}
        for ch in top_chunks:
            cid=int(ch["chunk_id"])
            for idx in range(cid-max(0,window),cid+max(0,window)+1):
                n=by_id.get(idx)
                if n is None or idx in seen or (idx!=cid and not self._same_parent(ch,n)):continue
                expanded.append(n);seen.add(idx)
            if include_parent and ch.get("parent_text") and len(ch["parent_text"])>len(ch.get("text","")):
                p=dict(ch);p["chunk_id"]=-(cid+1);p["text"]=ch["parent_text"]
                if p["chunk_id"] not in seen:expanded.append(p);seen.add(p["chunk_id"])
        return expanded

    @staticmethod
    def _format_context(chunks):
        """Return only source text to the generator.

        Filenames, page labels, and SOURCE/SECTION headers remain available in
        RetrievedChunk metadata for audit/export, but are intentionally hidden
        from the small local LLM so they cannot leak into employee-facing text.
        """
        values = []
        seen = set()
        for chunk in chunks:
            if chunk.get("content_role") == "links":
                continue
            text = str(chunk.get("text") or "").strip()
            sig = re.sub(r"\s+", " ", text).casefold()[:500]
            if not text or sig in seen:
                continue
            seen.add(sig)
            values.append(text)
        return values

    @staticmethod
    def _retrieved(candidates,limit):
        out=[]
        for c in candidates[:limit]:
            ch=c.chunk;out.append(RetrievedChunk(int(ch.get("chunk_id",0)),str(ch.get("text","")),int((ch.get("pages") or [1])[0]),ch.get("section_title") or ch.get("section"),float(c.vector_score),float(c.rerank_score),(ch.get("labels") or [f"Page {(ch.get('pages') or [1])[0]}"])[0],ch.get("source_type","document"),ch.get("filename",""),ch.get("parent_id",""),ch.get("record_type","policy")))
        return out

    def _append_related_link_references(self, retrieved, selected_chunks, limit=3):
        """Attach link-only blocks from the same source page for final validation.

        These blocks are never sent to the generator. They are added only to
        ``retrieved`` so the UI can append an exact document URL when relevant.
        """
        if not selected_chunks or limit <= 0:
            return retrieved
        source_pages = {
            (chunk.get("filename", ""), int(page))
            for chunk in selected_chunks
            for page in (chunk.get("pages") or [1])
        }
        seen = {(item.filename, item.chunk_id) for item in retrieved}
        for chunk in self.chunks:
            if chunk.get("content_role") != "links":
                continue
            pages = chunk.get("pages") or [1]
            if not any((chunk.get("filename", ""), int(page)) in source_pages for page in pages):
                continue
            key = (chunk.get("filename", ""), int(chunk.get("chunk_id", 0)))
            if key in seen:
                continue
            retrieved.append(RetrievedChunk(
                int(chunk.get("chunk_id", 0)), str(chunk.get("text", "")),
                int(pages[0]), chunk.get("section_title") or chunk.get("section"),
                0.0, 0.0, (chunk.get("labels") or [f"Page {pages[0]}"])[0],
                chunk.get("source_type", "document"), chunk.get("filename", ""),
                chunk.get("parent_id", ""), chunk.get("record_type", "policy"),
            ))
            seen.add(key)
            limit -= 1
            if limit <= 0:
                break
        return retrieved

    def query_faq_only(self, question: str):
        """Return an atomic FAQ answer when a safely grounded FAQ matches.

        This method lets the UI try the FAQ layer before broad deterministic
        heading routing. It returns ``None`` when the topic, answer type, or
        ranking margin is not reliable, so policy/document retrieval can proceed.
        """
        started = time.perf_counter()
        cfg = self.cfg
        question = str(question or "").strip()
        if not question or not self.chunks or not cfg.get("use_faq_layer", True):
            return None
        if _mentioned_document_filenames(question, self.chunks):
            return None

        base = expand_query(question) or [question]
        base = list(dict.fromkeys(
            base
            + _intent_search_variants(question)
            + _incident_search_variants(question)
            + _application_search_variants(question)
        ))[:7]
        if _should_pre_rewrite(question):
            rewritten = rewrite_query(question)
            if rewritten:
                base = list(dict.fromkeys(base + rewritten))[:6]

        faq = self._strong_resource_candidate(question)
        if faq is None:
            candidates = self._faq_lookup(question, base, cfg)
            threshold = float(cfg.get("faq_evidence_threshold", .36))
            faq = self._direct_faq(question, candidates, threshold)
            if faq is None:
                rewritten = rewrite_query(question)
                if rewritten:
                    expanded = list(dict.fromkeys(base + rewritten))[:6]
                    candidates = self._faq_lookup(question, expanded, cfg)
                    faq = self._direct_faq(question, candidates, max(.31, threshold - .03))
        if faq is None:
            return None

        answer = sanitize_answer(str(faq.chunk.get("answer", "")).strip(), question)
        score = evidence_confidence(
            vector_score=faq.vector_score,
            rrf_score=faq.search_score,
            rerank_score=faq.rerank_score,
            keyword_coverage=max(
                faq.coverage, keyword_coverage(question, faq.chunk.get("question", ""))
            ),
            heading_match=faq.heading_score,
            source_consistency=1,
            answer_support=1,
        )
        retrieved = self._retrieved([faq], 1)
        pages = {retrieved[0].page: 1.0} if retrieved else {}
        return RAGResult(
            answer, ANSWERED_SUPPORTED, retrieved, score, confidence_label(score),
            confidence_color(score), "faq", time.perf_counter() - started,
            self.pdf_name, pages,
        )

    def query(self, question, search_mode="hybrid"):
        started = time.perf_counter()
        cfg = self.cfg
        question = str(question or "").strip()
        if not question or not self.chunks:
            return self._refusal(started, search_mode, RETRIEVAL_FAILURE, 0)

        base = expand_query(question) or [question]
        base = list(dict.fromkeys(
            base + _intent_search_variants(question) + _incident_search_variants(question)
            + _topic_search_variants(question)
            + ([f"{question} earliest initial required actions opening phase deadline"] if _initial_action_question(question) else [])
        ))[:7]
        pre_rewrites = rewrite_query(question) if _should_pre_rewrite(question) else []
        if pre_rewrites:
            base = list(dict.fromkeys(base + pre_rewrites))[:5]
        if cfg.get("use_hyde", False):
            hypothetical = generate_freeform(
                "Write a brief factual answer-shaped search passage without inventing names, numbers, or rules.\n"
                "Question: " + question,
                90,
                0.0,
            )
            if hypothetical and not hypothetical.startswith("[ERROR]"):
                base.append(question + " " + hypothetical)

        # FAQ records are atomic. A question that explicitly names an indexed
        # document stays inside that document and bypasses the global FAQ layer.
        mentioned_files = _mentioned_document_filenames(question, self.chunks)
        faq = self._strong_resource_candidate(question) if not mentioned_files else None
        if not mentioned_files and faq is None:
            faq_candidates = self._faq_lookup(question, base, cfg)
            faq_threshold = float(cfg.get("faq_evidence_threshold", .36))
            faq = self._direct_faq(question, faq_candidates, faq_threshold)
            if faq is None:
                faq_rewrites = rewrite_query(question)
                if faq_rewrites:
                    expanded_faq_queries = list(dict.fromkeys(base + faq_rewrites))[:5]
                    faq_candidates = self._faq_lookup(question, expanded_faq_queries, cfg)
                    faq = self._direct_faq(question, faq_candidates, max(.32, faq_threshold - .025))

        if faq is not None:
            answer = sanitize_answer(str(faq.chunk.get("answer", "")).strip(), question)
            score = evidence_confidence(
                vector_score=faq.vector_score,
                rrf_score=faq.search_score,
                rerank_score=faq.rerank_score,
                keyword_coverage=max(
                    faq.coverage, keyword_coverage(question, faq.chunk.get("question", ""))
                ),
                heading_match=faq.heading_score,
                source_consistency=1,
                answer_support=1,
            )
            retrieved = self._retrieved([faq], 1)
            pages = {retrieved[0].page: 1.0} if retrieved else {}
            return RAGResult(
                answer, ANSWERED_SUPPORTED, retrieved, score, confidence_label(score),
                confidence_color(score), "faq", time.perf_counter() - started, self.pdf_name, pages
            )

        # Do not pass several FAQ records into the generator. If FAQ resolution
        # is ambiguous, search policy/document chunks only; a refusal is safer
        # than stitching one FAQ question to another FAQ answer.
        candidates = self._corrective_candidates(question, base, cfg, search_mode)
        if mentioned_files:
            named_candidates = [
                candidate for candidate in candidates
                if str(candidate.chunk.get("filename") or "") in mentioned_files
            ]
            if named_candidates:
                candidates = named_candidates
        if not candidates:
            return self._refusal(started, search_mode, RETRIEVAL_FAILURE, 0)

        strength = self._candidate_strength(candidates[0])
        threshold = float(cfg.get("evidence_threshold", .38))
        if strength < threshold:
            return self._refusal(
                started, search_mode, REFUSED_INSUFFICIENT_EVIDENCE, strength
            )

        final_k = int(cfg.get("final_k", 4))
        selected = self._select_coherent_candidates(question, candidates, final_k)
        top_chunks = [candidate.chunk for candidate in selected]
        expanded = self._expand_context(
            top_chunks,
            int(cfg.get("context_window", 0)),
            include_parent=(
                self._is_broad_question(question)
                or _initial_action_question(question)
                or _is_multi_topic_question(question)
            ),
        )
        top_ids = {int(chunk["chunk_id"]) for chunk in top_chunks}
        contexts = self._format_context(
            top_chunks + [
                chunk for chunk in expanded
                if int(chunk.get("chunk_id", 0)) not in top_ids
            ]
        )

        answer = generate_answer_full(contexts, question, int(cfg.get("max_tokens", 320)))
        if answer.startswith("[ERROR]"):
            return self._refusal(started, search_mode, GENERATION_FAILURE, strength)
        if NOT_FOUND.casefold() in answer.casefold():
            return self._refusal(
                started, search_mode, REFUSED_INSUFFICIENT_EVIDENCE, strength
            )

        answer = sanitize_answer(answer, question)
        verification = verify_answer(contexts, answer)
        support = answer_support_score(contexts, answer)
        if verification == GENERATION_FAILURE:
            return self._refusal(started, search_mode, GENERATION_FAILURE, strength)

        names = [candidate.chunk.get("filename", "") for candidate in selected]
        consistency = (
            max(names.count(name) for name in set(names)) / max(1, len(names))
            if names else 0
        )
        first = selected[0]
        score = evidence_confidence(
            vector_score=first.vector_score,
            rrf_score=first.search_score,
            rerank_score=first.rerank_score,
            keyword_coverage=first.coverage,
            heading_match=first.heading_score,
            source_consistency=consistency,
            answer_support=support,
        )
        if verification == ANSWERED_PARTIAL:
            score *= .82

        retrieved = self._retrieved(selected, final_k)
        retrieved = self._append_related_link_references(retrieved, top_chunks, limit=3)
        pages = {}
        for candidate, item in zip(selected, retrieved):
            pages[item.page] = max(
                pages.get(item.page, 0), self._candidate_strength(candidate)
            )
        if pages:
            max_page_score = max(pages.values())
            if max_page_score > 0:
                pages = {page: value / max_page_score for page, value in pages.items()}

        bounded_score = max(0, min(1, score))
        return RAGResult(
            answer, verification, retrieved, bounded_score, confidence_label(bounded_score),
            confidence_color(bounded_score), search_mode, time.perf_counter() - started,
            self.pdf_name, pages
        )



def _cache_identity(file_name):return {"cache_version":CACHE_VERSION,"loader_version":LOADER_VERSION,"chunker_version":CHUNKER_VERSION,"embedding_model":os.getenv("RAG_EMBED_MODEL","BAAI/bge-small-en-v1.5"),"rerank_model":os.getenv("RAG_RERANK_MODEL","cross-encoder/ms-marco-MiniLM-L-6-v2"),"file_name":file_name.casefold()}
def get_file_hash(file_bytes,file_name=""):
    h=hashlib.sha256();identity=_cache_identity(file_name)
    for k in sorted(identity):h.update(f"{k}={identity[k]}\n".encode())
    h.update(file_bytes);return h.hexdigest()
def ingest_document(file_bytes,file_name,cache_dir,cfg):
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True);ext=Path(file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:raise ValueError(f"Unsupported file type: {ext}")
    fh=get_file_hash(file_bytes,file_name);cp=cache_dir/f"{fh}_chunks.pkl";fp=cache_dir/f"{fh}_faiss.index";ep=cache_dir/f"{fh}_emb.npy";mp=cache_dir/f"{fh}_manifest.json";tmp=cache_dir/f"{fh}{ext or '.txt'}";tmp.write_bytes(file_bytes)
    units=load_document_text(str(tmp),file_name=file_name);total=sum(len(str(u.get("text","")).split()) for u in units)
    if not total:raise ValueError(f"No extractable text found in {file_name}.")
    chunks=chunk_text(units,*choose_chunk_params(total))
    if not chunks:raise ValueError(f"No searchable chunks could be built from {file_name}.")
    texts=[str(c.get("search_text") or c.get("text") or "") for c in chunks];emb=embed_chunks(texts);vs=VectorStore(emb);vs.save(fp);pr.save_chunks(chunks,cp);pr.save_embeddings(emb,ep);pr.save_manifest(_cache_identity(file_name)|{"file_hash":fh,"chunk_count":len(chunks),"embedding_dimensions":int(emb.shape[1]),"actual_embedding_model":embedding_model_name(),"actual_rerank_model":rerank_model_name()},mp)
    return chunks,vs,generate_document_summary(texts)
def load_document_cache(file_bytes,file_name,cache_dir):
    cache_dir=Path(cache_dir);fh=get_file_hash(file_bytes,file_name);cp=cache_dir/f"{fh}_chunks.pkl";fp=cache_dir/f"{fh}_faiss.index";mp=cache_dir/f"{fh}_manifest.json"
    if cp.exists() and fp.exists() and mp.exists() and pr.manifest_matches(mp,_cache_identity(file_name)):
        chunks=pr.load_chunks(cp);vs=VectorStore.load(fp)
        if len(chunks)==vs.index.ntotal:return chunks,vs
    return None
ingest_pdf=ingest_document
def load_pdf_cache(file_bytes,cache_dir):return load_document_cache(file_bytes,"uploaded.pdf",cache_dir)

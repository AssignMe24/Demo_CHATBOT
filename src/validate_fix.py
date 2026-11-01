from __future__ import annotations

import sys
from pathlib import Path

from answer_generation import (
    NOT_FOUND, _best_initial_phase_actions, _extract_initial_phase_actions, _initial_action_coverage,
    _missing_multi_topic_numeric_facts, _extractive_conditional_requirements,
    _missing_compound_requirement_parts, _append_missing_compound_requirements,
    _requirements_question, _looks_truncated_answer, _unsupported_numeric_sentences,
    apply_source_link_policy, sanitize_answer,
)
from chunker import choose_chunk_params, chunk_text
from document_loader import SUPPORTED_EXTENSIONS, load_document_text
from hybrid_search import anchor_coverage, intent_compatibility
try:
    from rag_pipeline import (
        RAGPipeline, _Candidate, _faq_alias_texts, _heading_type_match,
        _application_search_variants, _incident_search_variants, _mentioned_document_filenames,
        _requested_answer_type, _resource_name_score, _strong_resource_faq_match, _section_role_bonus,
    )
except Exception:
    RAGPipeline = _Candidate = None
    _faq_alias_texts = _heading_type_match = None
    _application_search_variants = _incident_search_variants = _mentioned_document_filenames = None
    _requested_answer_type = _resource_name_score = _strong_resource_faq_match = _section_role_bonus = None


def validate_answer_cleanup() -> list[str]:
    errors: list[str] = []

    copied_metadata = (
        "SOURCE: example.pdf | Page 2 | SECTION: Leave "
        "Question: How should this be recorded? "
        "Answer: Record the day using the required work category. "
        "[Guide](https://example.com/guide)"
    )
    cleaned = sanitize_answer(copied_metadata, "How should this be recorded?")
    if "SOURCE:" in cleaned or "Question:" in cleaned:
        errors.append("source metadata remained in the answer")
    if "Record the day" not in cleaned:
        errors.append("substantive answer text was removed while cleaning metadata")

    evidence_comment = (
        "Answer: Use the payroll portal. "
        "(Evidence supports only this part of the question. "
        "The evidence does not state anything else.)"
    )
    cleaned = sanitize_answer(evidence_comment, "Where can I find it?")
    if "evidence" in cleaned.casefold():
        errors.append("evidence commentary remained in the user-facing answer")

    inline_pdf_header = sanitize_answer(
        "Delete the record when no longer needed; Title: Example Policy Page: 10 "
        "ABC005 Issue 1.0 (C) Example Company - Company Confidential "
        "notify the responsible person immediately.",
        "What are the obligations?",
    )
    if "Title:" in inline_pdf_header or "Issue 1.0" in inline_pdf_header or "Company Confidential" in inline_pdf_header:
        errors.append("a flattened repeating PDF header remained inside the answer")
    if "notify the responsible person" not in inline_pdf_header:
        errors.append("substantive text after an inline PDF header was removed")

    initial_context = (
        "Within 5 business days, employees are required to:\n"
        "▪ Update their resume with recent experience.\n"
        "▪ Update their skills profile, proficiency levels, and certifications.\n"
        "Within the first 60 days, employees should pursue training."
    )
    phase_label, phase_actions = _extract_initial_phase_actions(initial_context)
    if "5 business days" not in phase_label.casefold() or len(phase_actions) != 2:
        errors.append("parallel actions in the earliest source phase were not extracted")
    if _initial_action_coverage("Update the resume.", phase_actions) >= .75:
        errors.append("an incomplete initial-action answer was incorrectly treated as complete")

    competing_initial_contexts = [
        (
            "A later communication contains many contacts. Resume must be updated within 5 business days. "
            "Contact the support team if required."
        ),
        (
            "Within 5 business days, employees are required to:\n"
            "▪ Update their resume with recent experience.\n"
            "▪ Update their skills profile, proficiency levels, and certifications.\n"
            "Within the first 60 days, employees should pursue training."
        ),
    ]
    best_label, best_actions = _best_initial_phase_actions(
        competing_initial_contexts,
        "I was moved to the bench. What should I do first?",
    )
    if "5 business days" not in best_label.casefold() or len(best_actions) != 2:
        errors.append("the most complete opening-phase action block was not selected across chunks")
    if not any("skills profile" in action.casefold() for action in best_actions):
        errors.append("a parallel initial skills-update action was omitted")

    multi_context = (
        "Ordinary leave lasts 26 weeks. Additional leave lasts a further 26 weeks, "
        "for 52 weeks in total. Pay is available for up to 39 weeks. "
        "For the first 6 weeks it is paid at 90% of average earnings, and the standard "
        "rate applies for the remaining 33 weeks."
    )
    missing_numeric = _missing_multi_topic_numeric_facts(
        "Ordinary leave lasts 26 weeks and pay lasts 39 weeks.",
        multi_context,
        "Explain the leave and the pay.",
    )
    if not any("52 weeks" in fact for fact in missing_numeric) or not any("33 weeks" in fact for fact in missing_numeric):
        errors.append("missing multi-topic phase/total facts were not detected")

    broad_pay_context = (
        "Statutory pay is payable for up to 39 weeks. "
        "This period can begin from 14 days before placement. "
        "For the first 6 weeks it is paid at 90% of average earnings and the standard rate applies for the remaining 33 weeks."
    )
    broad_missing = _missing_multi_topic_numeric_facts(
        "Pay is available for 39 weeks, with 6 weeks at 90% and 33 weeks at the standard rate.",
        broad_pay_context,
        "Explain the leave and the pay.",
    )
    if any("14 days" in fact for fact in broad_missing):
        errors.append("a procedural start-date detail was appended to a broad multi-topic summary")

    source = "Please use [Example Portal](https://example.com/portal)."
    kept = apply_source_link_policy(
        "Use [Example Portal](https://example.com/portal).",
        "Where can I find the item?",
        source,
    )
    if "https://example.com/portal" not in kept:
        errors.append("a relevant source-backed URL was removed")

    invented = apply_source_link_policy(
        "Use [Example Portal](https://invented.example/portal).",
        "Where can I find the item?",
        source,
    )
    if "invented.example" in invented:
        errors.append("an invented URL was not removed")

    appended = apply_source_link_policy(
        "Use the Example Portal.",
        "Where can I find the item?",
        source,
    )
    if "https://example.com/portal" not in appended:
        errors.append("a relevant source link was not appended")

    unrelated_source = "Company website: [Company](https://example.com)."
    unrelated = apply_source_link_policy(
        "The purpose is to define employee leave eligibility.",
        "What is the purpose of the leave policy?",
        unrelated_source,
    )
    if "https://example.com" in unrelated:
        errors.append("an unrelated cover/company URL was appended")

    compact_steps = (
        "Step1:\u00a0Sign in\u00a0Step2:\u00a0Open the request Step3: Submit it "
        "Personal Details: Name / Date Note: Keep the confirmation."
    )
    formatted = sanitize_answer(compact_steps, "How do I complete the process?")
    if "\n2. Open the request" not in formatted or "\n3. Submit it" not in formatted:
        errors.append("compact procedure steps were not formatted as a readable list")
    if "- **Personal Details:**" not in formatted or "- **Note:**" not in formatted:
        errors.append("trailing procedure notes were not separated from the steps")

    compact_bullets = sanitize_answer(
        "• First requirement • Second requirement",
        "What are the requirements?",
    )
    if not compact_bullets.startswith("- First requirement") or "\n- Second requirement" not in compact_bullets:
        errors.append("compact bullet items were not formatted as separate Markdown lines")

    requirement_context = (
        "3.1 The employee must notify the manager within 7 days. "
        "3.2 The employee must provide a matching certificate for a local case. "
        "3.3 If the case is overseas then the employee must provide the authority notification "
        "and the expected arrival date within 28 days."
    )
    requirement_answer = _extractive_conditional_requirements(
        requirement_context,
        "What documents do I need to submit?",
    )
    if "matching certificate" not in requirement_answer or "authority notification" not in requirement_answer:
        errors.append("focused document requirements were not retained")
    if "notify the manager" in requirement_answer:
        errors.append("unrelated notification timing was included in a document-requirement answer")
    if requirement_answer.rstrip().endswith("within"):
        errors.append("a requirement sentence was cut in the middle")

    if not _requirements_question("How to apply for the requested leave"):
        errors.append("application wording did not activate the complete requirements prompt")

    compound_context = (
        "If the case is from overseas then the applicant must provide the authority notification "
        "and the expected arrival date within 28 days of receiving it and the actual arrival date."
    )
    incomplete_compound = (
        "For an overseas case, provide the authority notification and the expected arrival date."
    )
    missing_compound = _missing_compound_requirement_parts(incomplete_compound, compound_context)
    if not any("28 days" in part for part in missing_compound) or not any("actual arrival" in part for part in missing_compound):
        errors.append("later items in a compound source requirement were not detected as missing")
    repaired_compound = _append_missing_compound_requirements(incomplete_compound, compound_context)
    if "28 days" not in repaired_compound or "actual arrival date" not in repaired_compound:
        errors.append("a partial compound requirement was not restored from the source clause")

    punctuation_hyphen = sanitize_answer(
        "Inform the team (SPOC - Alex Morgan) within 24 hours.",
        "What should I do?",
    )
    if "SPOC\n- Alex" in punctuation_hyphen:
        errors.append("a punctuation hyphen was incorrectly converted into a bullet")

    link_hyphen = sanitize_answer(
        "See [Weekend - Holiday Guide](https://example.com/guide).",
        "How should I record it?",
    )
    if "Weekend\n- Holiday" in link_hyphen:
        errors.append("a hyphen inside a link label was incorrectly converted into a bullet")

    if not _looks_truncated_answer(
        "This is a sufficiently long generated answer explaining several supported points. "
        "It includes the main conditions and then stops abruptly. If an"
    ):
        errors.append("truncated answer detection did not catch an unfinished sentence")

    numeric_context = (
        "All qualifying employees may take up to 26 weeks additional leave. "
        "Statutory pay is payable for up to 39 weeks during leave."
    )
    numeric_answer = "Additional leave can be taken for up to 39 weeks."
    if not _unsupported_numeric_sentences(numeric_answer, numeric_context):
        errors.append("a number attached to the wrong subject was not detected")

    leave_pay_context = (
        "All qualifying employees may take 26 weeks ordinary leave and a further 26 weeks additional leave, "
        "for 52 weeks total. Statutory Adoption Pay is payable for up to 39 weeks."
    )
    if not _unsupported_numeric_sentences(
        "The adoption leave policy allows employees to take up to 39 weeks of leave.",
        leave_pay_context,
    ):
        errors.append("a pay duration incorrectly attached to leave was not detected")

    if intent_compatibility(
        "I do not know my username", "What is the user ID for my login?"
    ) <= intent_compatibility(
        "I do not know my username", "How do I access the portal?"
    ):
        errors.append("identifier intent did not outrank access intent")

    if intent_compatibility(
        "Is the deadline weekly or monthly?", "When is the submission deadline?"
    ) <= intent_compatibility(
        "Is the deadline weekly or monthly?", "What happens if I forget to submit?"
    ):
        errors.append("timing intent did not outrank consequence intent")

    if anchor_coverage(
        "I shared confidential personal information with the wrong person",
        "I worked on a weekend. How should I log it?",
    ) != 0:
        errors.append("unrelated FAQ topics unexpectedly shared a content anchor")

    if anchor_coverage(
        "I was moved to the bench and need my first action",
        "Can I withdraw money from my retirement fund?",
    ) != 0:
        errors.append("unrelated bench and withdrawal topics unexpectedly shared a content anchor")

    if _requested_answer_type is not None:
        if _requested_answer_type(
            "I noticed an error after approval. Can it still be corrected?"
        ) != "procedure":
            errors.append("corrected/edit wording was not classified as a procedure")

        if _requested_answer_type(
            "When is an employee entitled to this benefit?"
        ) != "eligibility":
            errors.append("entitlement wording was not classified as eligibility")

        if _requested_answer_type(
            "What are the main rights and benefits?"
        ) != "eligibility":
            errors.append("rights/benefits wording was not classified as eligibility")

        if _section_role_bonus("the leave policy", "2. ENTITLEMENTS") <= _section_role_bonus(
            "the leave policy", "10. RIGHTS ON AND AFTER RETURNING TO WORK"
        ):
            errors.append("a broad policy topic did not prefer entitlement/overview sections over later return sections")

        if _requested_answer_type(
            "Is the submission deadline weekly or at month end?"
        ) != "deadline":
            errors.append("weekly/month-end wording was not classified as a deadline")

        if _requested_answer_type(
            "What happens if I forget and submit late?"
        ) != "consequence":
            errors.append("missed/late wording was not classified as a consequence")


        if _requested_answer_type(
            "Where can I view the current resource?"
        ) != "access":
            errors.append("view/locate wording was not classified as resource access")

        if _resource_name_score is not None:
            correct_resource = _resource_name_score(
                "Tell me about Global Holiday Calender?",
                "Where can I view the Global Holiday Calendar?",
                "Please refer [Global Holiday Calendar - 2025](https://example.com/calendar).",
            )
            unrelated_resource = _resource_name_score(
                "Tell me about Global Holiday Calender?",
                "How can I find network hospitals?",
                "Please refer [Network Hospital List](https://example.com/hospitals).",
            )
            if correct_resource < .68 or correct_resource <= unrelated_resource:
                errors.append("a short named resource with a minor typo did not outrank unrelated resources")

            if _strong_resource_faq_match is not None:
                resource_chunks = [
                    {
                        "record_type": "faq",
                        "question": "Where can I view the Global Holiday Calendar?",
                        "answer": "Please refer [Global Holiday Calendar - 2025](https://example.com/calendar).",
                    },
                    {
                        "record_type": "faq",
                        "question": "How can I find network hospitals?",
                        "answer": "Please refer [Network Hospital List](https://example.com/hospitals).",
                    },
                ]
                matched_resource = _strong_resource_faq_match(
                    "Tell me about Global Holiday Calender?", resource_chunks
                )
                if matched_resource is None or "Holiday Calendar" not in matched_resource[0].get("question", ""):
                    errors.append("the deterministic named-resource resolver did not select the unique corpus resource")
                if _strong_resource_faq_match("Tell me about policy?", resource_chunks) is not None:
                    errors.append("a generic resource request was accepted by the deterministic resolver")

        if _requested_answer_type(
            "Under what circumstances can I use the fund before retirement?"
        ) != "eligibility":
            errors.append("circumstances/withdrawal wording was not classified as eligibility")

        if _heading_type_match(
            "When is an employee entitled to leave?", "2. ENTITLEMENTS"
        ) < 1.0:
            errors.append("eligibility questions did not prefer an entitlement heading")

        if not _incident_search_variants(
            "I accidentally shared confidential personal information with the wrong person."
        ):
            errors.append("accidental disclosure scenario did not receive an incident search bridge")

        if not _application_search_variants(
            "What documents do I need to submit to apply?"
        ):
            errors.append("application/document questions did not receive a requirements search bridge")

        if _mentioned_document_filenames is not None:
            named = _mentioned_document_filenames(
                "How can employability be enhanced according to Bench Policy?",
                [
                    {"filename": "Bench Policy_Final 1.pdf"},
                    {"filename": "HR_Chatbot_FAQs_Final.docx"},
                ],
            )
            if "Bench Policy_Final 1.pdf" not in named:
                errors.append("a filename with a numeric/version suffix was not detected in the question")

        synthetic_faqs = [
            {"question": "How do I access the payroll portal?", "answer": "Use ExamplePay."},
            {"question": "What is the user ID for my ExamplePay login?", "answer": "Use the employee code."},
        ]
        aliases = _faq_alias_texts(synthetic_faqs)
        if "payroll portal" not in aliases.get(1, "").casefold():
            errors.append("related FAQ records did not create a corpus-derived terminology bridge")

        unrelated_alias_faqs = [
            {"question": "How can I edit an approved record?", "answer": "Ask a manager to reject it."},
            {"question": "How do I update a correction in another account?", "answer": "Click Manage and select the declaration."},
        ]
        unrelated_aliases = _faq_alias_texts(unrelated_alias_faqs)
        if any(unrelated_aliases.values()):
            errors.append("generic procedure words created an unsafe cross-topic FAQ alias")

        unrelated = _Candidate(
            {"record_type": "faq", "answer": "Call the emergency contact.",
             "question": "Whom should I contact in a medical emergency?"},
            .75, .65, .2, .2, .25, 1.0, 2.0,
        )
        unrelated.anchor_score = 0.0
        unrelated.type_match = 1.0
        unrelated.type_conflict = False
        if RAGPipeline._direct_faq(
            "Whom should I connect with while on the bench?", [unrelated], .30
        ) is not None:
            errors.append("a same-type but topically unrelated FAQ was accepted")

        unrelated_consequence = _Candidate(
            {"record_type": "faq", "answer": "Missing a submission deadline may prevent recording time.",
             "question": "What happens if I forget to submit on time?"},
            .88, .78, .4, .4, .4, 1.0, 3.0,
        )
        unrelated_consequence.anchor_score = 0.0
        unrelated_consequence.type_match = 1.0
        unrelated_consequence.type_conflict = False
        if RAGPipeline._direct_faq(
            "What happens if an employee does not return to work?", [unrelated_consequence], .30
        ) is not None:
            errors.append("an unrelated consequence FAQ was accepted without an original-topic anchor")

        payroll = _Candidate(
            {"record_type": "faq", "answer": "Use the employee code.",
             "question": "What is the user ID for my ExamplePay login?"},
            .72, .64, .2, .2, .30, 1.0, 2.0,
        )
        payroll.anchor_score = .34
        payroll.type_match = 1.0
        payroll.type_conflict = False
        if RAGPipeline._direct_faq(
            "I can access the payroll portal but do not know my username.", [payroll], .30
        ) is None:
            errors.append("a corpus-linked identifier FAQ was rejected")

        deadline = _Candidate(
            {"record_type": "faq", "answer": "Submit weekly and finally at month end.",
             "question": "When is the submission deadline?"},
            .74, .66, .25, .20, .30, 1.0, 2.2,
        )
        deadline.anchor_score = 0.0
        deadline.type_match = 1.0
        deadline.type_conflict = False
        deadline.question_lexical = .25
        if RAGPipeline._direct_faq(
            "Is there a deadline every week or only at month end?", [deadline], .30
        ) is None:
            errors.append("a same-topic deadline FAQ was rejected")

        resource = _Candidate(
            {"record_type": "faq", "answer": "Please refer [Calendar](https://example.com/calendar).",
             "question": "Where can I view the Global Holiday Calendar?"},
            .72, .65, .20, .70, .40, 0.0, 2.0,
        )
        resource.anchor_score = .66
        resource.type_match = 0.0
        resource.type_conflict = False
        resource.question_lexical = .40
        resource.answer_lexical = .20
        if RAGPipeline._direct_faq(
            "Tell me about Global Holiday Calender?", [resource], .30
        ) is None:
            errors.append("a strongly named resource FAQ with a minor spelling variation was rejected")

        # Explicit multi-topic questions should not be padded with unrelated
        # high-scoring parent sections after the requested focus groups are covered.
        fake_candidates = []
        for idx, parent in enumerate(("leave", "pay", "eligibility", "return", "contact")):
            candidate = _Candidate(
                {"chunk_id": idx, "filename": "example.pdf", "parent_id": parent,
                 "text": parent, "section": parent},
                .90 - idx * .03, .70, .30, .20, .45, .80, 2.5 - idx * .10,
            )
            candidate.matched_queries = (idx,)
            fake_candidates.append(candidate)
        pipeline = RAGPipeline.__new__(RAGPipeline)
        focused = pipeline._select_coherent_candidates(
            "Clearly explain the employee leave arrangements and statutory payment details",
            fake_candidates,
            5,
        )
        if len(focused) > 3:
            errors.append("an explicit multi-topic answer was padded with unrelated parent sections")


    return errors


def validate_document(path: Path) -> list[str]:
    errors: list[str] = []
    units = load_document_text(str(path), file_name=path.name)
    total_words = sum(len(str(unit.get("text", "")).split()) for unit in units)
    chunks = chunk_text(units, *choose_chunk_params(total_words))
    if not chunks:
        return ["no searchable chunks produced"]

    for index, chunk in enumerate(chunks):
        if chunk.get("chunk_id") != index:
            errors.append(f"chunk {index}: non-contiguous chunk_id")
        if not str(chunk.get("text", "")).strip():
            errors.append(f"chunk {index}: empty text")
        if chunk.get("record_type") == "faq":
            if not str(chunk.get("question", "")).strip():
                errors.append(f"chunk {index}: FAQ question missing")
            if not str(chunk.get("answer", "")).strip():
                errors.append(f"chunk {index}: FAQ answer missing")
        if chunk.get("content_role") == "links" and not str(chunk.get("text", "")).casefold().startswith("document links"):
            errors.append(f"chunk {index}: invalid link-only role")
    return errors


def validate_app_routing() -> list[str]:
    """Static regression checks for deterministic route precedence.

    ``app.py`` is a Streamlit entry point and should not be imported by the
    validator. These checks confirm that broad/multi-topic questions cannot be
    intercepted by the one-sentence extractor and that exact section routing is
    attempted first.
    """
    errors: list[str] = []
    app_path = Path(__file__).with_name("app.py")
    source = app_path.read_text(encoding="utf-8")

    route_start = source.find("for route in (")
    route_end = source.find("):", route_start)
    route_block = source[route_start:route_end] if route_start >= 0 and route_end >= 0 else ""
    if not route_block or route_block.find("heading_answer") > route_block.find("high_overlap_sentence_answer"):
        errors.append("exact heading routing does not run before the one-sentence extractor")

    fn_start = source.find("def high_overlap_sentence_answer")
    fn_end = source.find("\ndef deterministic_policy_answer", fn_start)
    fn_block = source[fn_start:fn_end] if fn_start >= 0 and fn_end >= 0 else ""
    if "is_multi_topic_request(question)" not in fn_block:
        errors.append("multi-topic questions are not excluded from one-sentence extraction")
    if "is_summary_request(question)" not in fn_block:
        errors.append("summary questions are not excluded from one-sentence extraction")
    if "required documents" not in fn_block or "application" not in fn_block:
        errors.append("application/document requirements are not excluded from one-sentence extraction")

    heading_start = source.find("def heading_answer(")
    heading_end = source.find("\ndef estimate_available_content_words", heading_start)
    heading_block = source[heading_start:heading_end] if heading_start >= 0 and heading_end >= 0 else ""
    strong_pos = heading_block.find("strong_heading_request =")
    multi_pos = heading_block.find("is_multi_topic_request(question)")
    if strong_pos < 0 or multi_pos < strong_pos:
        errors.append("single section titles containing 'and' are still rejected as multi-topic before heading matching")
    named_pos = heading_block.find("_strong_named_section_chunks")
    generic_pos = heading_block.find("find_section_chunks_by_topic")
    if named_pos < 0 or generic_pos < 0 or named_pos > generic_pos:
        errors.append("explicit section-title matching does not run before generic topic scoring")
    return errors


def main(folder: str | None = None) -> None:
    failed = False
    cleanup_errors = validate_answer_cleanup()
    print("Answer cleanup:", "PASS" if not cleanup_errors else "FAIL")
    for error in cleanup_errors:
        print("  -", error)
    failed = failed or bool(cleanup_errors)

    routing_errors = validate_app_routing()
    print("App routing:", "PASS" if not routing_errors else "FAIL")
    for error in routing_errors:
        print("  -", error)
    failed = failed or bool(routing_errors)

    if folder:
        root = Path(folder)
        files = [
            path for path in root.iterdir()
            if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        ]
        for path in sorted(files):
            try:
                errors = validate_document(path)
            except Exception as exc:
                errors = [str(exc)]
            print(path.name + ":", "PASS" if not errors else "FAIL")
            for error in errors:
                print("  -", error)
            failed = failed or bool(errors)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

# Step 17 — Deterministic Resource Match and Compound Requirement Completeness

This build fixes the two remaining issues found in `chat_history(20260806-002359).db`.

## What was fixed

### 1. Short named FAQ resources no longer depend on embedding/reranker thresholds

A question such as:

`Tell me about Global Holiday Calender?`

is matched against resource names derived dynamically from:

- indexed FAQ questions,
- hyperlink labels, and
- FAQ answer text.

The resolver allows a small spelling error only when the best resource match is strong and clearly separated from the next candidate. Generic requests such as `Tell me about policy` are rejected. No resource name, URL, FAQ number, or expected answer is hardcoded.

### 2. Application questions now use the requirements prompt

Words such as `apply`, `application`, `notify`, `notification`, and `provide` now activate the complete requirements flow.

### 3. Joined source requirements are checked for omitted later items

When one source clause contains several mandatory items joined by `and` or `or`, the answer is checked against the matching answer sentence. If the local model keeps the first item but drops a later item, the incomplete sentence is replaced by the complete source clause.

This is generic and source-driven. It does not contain Adoption Leave terms, dates, document names, or expected answers.

## Files changed

- `rag_pipeline.py`
- `answer_generation.py`
- `validate_fix.py`

The ZIP includes all runtime files so the source folder stays version-consistent.

## Apply the package

1. Stop Streamlit with `Ctrl+C`.
2. Extract the ZIP.
3. Replace all matching files inside the application's `src` folder.
4. Keep these existing items unchanged:
   - `chat.html`
   - `documents`
   - `chat_history.db`
   - `feedback.db`
5. Run:

```bat
venv\Scripts\activate
rmdir /s /q src\__pycache__ 2>nul
cd src
python validate_fix.py ..\documents
streamlit run app.py
```

The embedding cache does not need to be deleted.

## Expected validation

```text
Answer cleanup: PASS
App routing: PASS
Adoption Leave Policy.pdf: PASS
Bench Policy_Final 1.pdf: PASS
DataProtectionPolicy (1).pdf: PASS
HR_Chatbot_FAQs_Final.docx: PASS
```

## Retest

```text
Tell me about Global Holiday Calender?
How to apply for adoption leave
```

The first answer should return the exact source-backed Global Holiday Calendar link. The second should retain every item in the overseas-adoption requirement, including the expected-arrival timing and the actual arrival date.

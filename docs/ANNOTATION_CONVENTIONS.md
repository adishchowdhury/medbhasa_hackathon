# Annotation conventions used by JUSL Health Home

These are the decisions we applied **consistently** to every one of the 232 notes. They implement the organisers' `entity_schema.md` v1.0 and
`annotation_guidelines.pdf`; wherever the schema left a case open we wrote the decision down here (the guidelines ask teams to do this).
The worked example from the organisers (`resources/worked_example/`) was used as the tie-breaker.

Notation: `{TYPE|span}` is the markup used in `data/round1/annotation_sources/*.txt`.

| # | Rule | Example (Bengali) |
|---|---|---|
| C1 | Age, sex, occupation, section headers (`প্রধান অভিযোগ:`, `Dx:` …), advice sentences, food, lifestyle risk factors (smoking, alcohol) are **not** entities → `O`. | `৪৫ বছর বয়সী পুরুষ রোগী`, `লবণ কম খান` |
| C2 | **TEMPORAL**: onset, duration, time-of-day of a symptom, follow-up interval. Tag the whole phrase including `ধরে / পর / আগে / থেকে`. A medication course length (`৭ দিন`) is a separate TEMPORAL span after the dosage. | `{TMP|গত তিন দিন ধরে}`, `{TMP|২ সপ্তাহ পর}`, `{TMP|রাতে} {SX|কাশি}` |
| C3 | **DOSAGE**: strength, frequency, meal timing, route and "as needed" as **one** contiguous span; the time of day belongs to DOSAGE when it describes when to take the drug. | `{MED|মেটফরমিন} {DOS|৫০০ মি.গ্রা. দিনে ২ বার}`, `{DOS|শিরায়}`, `{DOS|সকালে খালি পেটে}` |
| C4 | **MEDICATION**: generic or brand name, drug class, vaccine, ORS, IV fluid, blood product. The dosage **form** that directly follows the name (`সিরাপ`, `ক্রিম`, `ইনহেলার`, `ড্রপ` …) is part of the MEDICATION span. Strength is not. | `{MED|সালবিউটামল ইনহেলার}`, `{MED|পারমেথ্রিন ক্রিম} {DOS|৫%}` |
| C5 | **TEST**: named investigations and the *name* of a vital sign (`রক্তচাপ`, `নাড়ির গতি`, `তাপমাত্রা`, `ওজন`, `SpO₂`, `BMI`). A body part in front of an imaging name stays `BODY_SITE` (as in the organisers' worked example: `{BODY|বুকের} {TEST|এক্স-রে}`). | `{TEST|রক্তচাপ} {RES|১১০/৭০ মি.মি. পারদ}` |
| C6 | **TEST_RESULT**: numeric value **with** its unit, qualitative result, and findings reported by an investigation (imaging / lab / ECG). | `{RES|৯.২%}`, `{RES|পজিটিভ}`, `{TEST|আল্ট্রাসনোগ্রাফি}-তে {BODY|পিত্তথলিতে} {RES|পাথর}` |
| C7 | **BODY_SITE**: anatomical location incl. laterality/position words and the attached case ending. A site is **split from the symptom** it carries: `{BODY|পেটে} {SX|ব্যথা}`. Exception: lexicalised symptoms whose name contains the organ (`মাথা ঘোরা`, `বুক ধড়ফড়`, `হাত কাঁপা`, `চুল পড়া`) stay one SYMPTOM. | `{BODY|ডান পায়ের বুড়ো আঙুলে}`, `{BODY|বুকে} {SX|ব্যথা}` |
| C8 | **SYMPTOM**: patient complaints **and** observed physical-exam signs (`wheeze`, `ফোলা`, `লাল`, `পুঁজ`). A severity adjective is inside the span. | `{SX|তীব্র ব্যথা}`, `{BODY|টনসিল} {SX|লাল} ও {SX|ফোলা}` |
| C9 | **DIAGNOSIS**: named disease/condition, confirmed or suspected, also when it appears in the history. Abbreviations in brackets are separate spans. A past-history disease is a DIAGNOSIS even though it is not today's problem. | `{DX|টাইপ ২ ডায়াবেটিস}`, `{DX|প্রস্টেট বৃদ্ধি} ({DX|BPH})` |
| C10 | **PROCEDURE**: therapeutic / interventional acts (surgery, endoscopy, dressing, nebulisation, physiotherapy, plaster, counselling, stent). Diagnostic tests and imaging are TEST; if unsure → TEST (schema §3). | `{PROC|ড্রেসিং}`, `{PROC|নেবুলাইজেশন}`, `{PROC|আপার জিআই এন্ডোস্কপি}` |
| C11 | **NEGATION_UNCERTAINTY**: only the marker (`নেই`, `না`, `নয়`, `অস্বীকার`, `সন্দেহ`, `হতে পারে`, `সম্ভাব্য`, `যায়নি`) — never `কোনো`, never advice verbs such as `বন্ধ`, `নিষেধ`, `বর্জন`. The negated entity keeps its own type (schema §3). | `{SX|জ্বর} {NEG|নেই}`, `{DX|ডেঙ্গু}র {NEG|সন্দেহ}` |
| C12 | Punctuation tokens are always `O`; they are never inside a span. Content in brackets is tagged as separate spans. | `{DX|মূত্রনালির সংক্রমণ} ({DX|UTI})` |
| C13 | A token with an attached hyphen suffix is tagged as one whole token. | `{TEST|ইসিজি}-তে` → token `ইসিজি-তে` |
| C14 | Spelling variants (`মেটফর্মিন` / `মেটফরমিন`) are **not** normalised; each occurrence is tagged as written (guidelines §6). | — |
| C15 | English / code-mixed terms are tagged by meaning, not script. | `{SX|wheeze}`, `{TEST|HbA1c}` |

## Known ambiguities (to be reviewed by a clinician)

* **C7 vs. lexicalised symptoms** – the list of "organ-in-name" idioms is small and arbitrary; other teams (and the hidden gold) may split `মাথা ঘোরা` into `BODY_SITE + SYMPTOM`.
* **C6** – units are inside TEST_RESULT; the organisers' worked example only shows the bare number (`১৪০/৯০`). A scorer that expects bare numbers will count our longer spans as wrong.
* **C9** – history-only diagnoses are tagged DIAGNOSIS; a clinician could argue for a separate "history" label that the schema does not have.
* **C2 / C3** – `সকালে`, `রাতে` are TEMPORAL when they describe a symptom and DOSAGE when they describe when to take a drug; the two readings are separated only by context.
* **Round-1 notes n001–n040** – eight tags were corrected to follow these rules; every change is listed in `round1_deliverables/round1_fixes_log.tsv`.

## Process

1. Each note and its entity spans were written as inline markup in batches of about 20 (`batch_03 … batch_11`); the annotation method is stated in `round1_deliverables/SOURCE_DECLARATION.md`.
2. A deterministic script (`src/medbhasha/corpus.py`) strips the markup, tokenises with the same tokenizer later used on the test notes, and projects spans to BIO tags. It **fails** if a span cuts through a token, overlaps another span, or contains punctuation.
3. After every batch we ran the official `validate.py`; a global consistency audit (same surface form tagged with different types; symptom spans that contain a body-site word) led to the C7 clean-up.
4. **No clinician reviewed these annotations, and no second annotator exists** → no inter-annotator agreement can be reported.

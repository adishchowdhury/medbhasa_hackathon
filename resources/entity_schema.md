# MedBhasha — Entity Schema & Annotation Guidelines
### Version 1.0 · Released [October 12, 2026]

This document defines the entity categories every team must use when annotating
their corpus in Round 1. All submitted annotations are scored against this
exact schema — read it fully before you start tagging, and check the worked
example in `worked_example/` alongside this file.

If anything here is ambiguous for a specific note, open an issue in this
repository rather than guessing — your question and the answer become visible
to every team, which keeps scoring fair.

---

## 1. Tagging scheme

Annotations use **BIO** tagging at the token level (BIOES is also accepted —
see `validate.py --scheme BIOES`):

| Prefix | Meaning |
|---|---|
| `B-<TYPE>` | First token of an entity span |
| `I-<TYPE>` | Continuation token of the same entity span |
| `O` | Not part of any entity |

A span is one or more consecutive tokens of the **same** entity type. Do not
tag overlapping or nested spans — if a phrase could fit two categories, pick
the single most specific one (see §3, Edge Cases).

---

## 2. Entity categories

| # | Tag value | Category | Definition | Includes | Example terms (bn) |
|---|---|---|---|---|---|
| 1 | `DIAGNOSIS` | Problem / Diagnosis | A named disease, condition, or diagnosis attributed to the patient | Confirmed or suspected diagnoses | ডায়াবেটিস, উচ্চ রক্তচাপ, নিউমোনিয়া |
| 2 | `SYMPTOM` | Symptom | A subjective complaint or observed sign reported by or about the patient | Pain, fever, cough, fatigue, etc. | জ্বর, কাশি, ব্যথা, দুর্বলতা |
| 3 | `MEDICATION` | Medication | The name of a drug or chemical substance | Brand **and** generic names — tag both; do not try to resolve one to the other | মেটফরমিন, নাপা, ইনসুলিন |
| 4 | `DOSAGE` | Dosage / Frequency / Route | The amount, frequency, or route of a medication | Strength (mg/ml), frequency ("twice daily"), route ("oral", "IV") | ৫০০ মিগ্রা, দিনে দুইবার, মুখে খাওয়ার |
| 5 | `TEST` | Test / Investigation | A named lab test or imaging/diagnostic procedure ordered or performed | Lab tests, imaging (X-ray, USG, CT, MRI), vitals checks | রক্তচাপ পরীক্ষা, এক্স-রে, রক্ত পরীক্ষা |
| 6 | `TEST_RESULT` | Test Result / Value | The reported value or outcome of a test | Numeric values, qualitative results ("positive", "normal") | ১৪০/৯০, পজিটিভ, স্বাভাবিক |
| 7 | `BODY_SITE` | Body Site / Anatomy | An anatomical location referenced in the text | Organs, body regions | বুক, পেট, মাথা, কিডনি |
| 8 | `PROCEDURE` | Procedure | A medical or surgical procedure performed (not a diagnostic test — see §3) | Surgery, endoscopy, catheterization, dressing change | এন্ডোস্কপি, অস্ত্রোপচার |
| 9 | `TEMPORAL` | Temporal expression | An expression of onset, duration, frequency-in-time, or follow-up timing | "for 3 days", "last week", "follow up in 2 weeks" | গত তিন দিন ধরে, দুই সপ্তাহ পর |
| 10 | `NEGATION_UNCERTAINTY` | Negation / Uncertainty marker | A word or phrase marking negation or diagnostic uncertainty | "denies", "no complaint of", "rule out", "?diabetes" | নেই, অস্বীকার করে, সন্দেহ |

---

## 3. Edge cases & decisions

- **Test vs. Procedure:** imaging and lab investigations (X-ray, USG, blood
  tests) are `TEST`, not `PROCEDURE`. `PROCEDURE` is reserved for
  therapeutic/interventional acts (surgery, endoscopy, dressing, injection
  as an act). If you're unsure whether something is diagnostic or
  therapeutic, default to `TEST`.
- **Negation scope:** tag only the negation/uncertainty word or short marker
  phrase itself as `NEGATION_UNCERTAINTY` (e.g., "নেই", "অস্বীকার করে"). Do
  **not** extend the tag across the symptom/diagnosis it negates — that stays
  tagged under its own category (e.g., `SYMPTOM`) as normal. The negation
  relationship itself is not encoded in this schema version.
- **Brand vs. generic medication:** tag both the same way, as `MEDICATION`.
  We are not distinguishing brand/generic in the tag itself this season —
  note it in free text in your EDA/report if it's relevant to your analysis.
- **Multi-word spans:** tag every token of a multi-word entity — the first
  token `B-<TYPE>`, every following token in the same span `I-<TYPE>`. Do not
  tag only the head word.
- **Ambiguous category:** if a phrase genuinely fits two categories, choose
  the more specific one and move on — don't spend annotation time
  deliberating. Note the decision briefly in your report's limitations
  section if a pattern of ambiguity comes up often in your data.

---

## 4. File format

Submit your annotated corpus as a single CSV (or TSV) file, one row per
token:

```
note_id,token,tag
n001,রোগীর,O
n001,গত,B-TEMPORAL
n001,তিন,I-TEMPORAL
...
```

- Column names must be exactly `note_id`, `token`, `tag` (lowercase).
- One blank line or a `note_id` change marks a new note — there is no
  separate sentence boundary marker; sentence structure is not scored.
- Tag values must exactly match the `TAG VALUE` column in §2 above
  (case-sensitive, e.g. `B-DIAGNOSIS`, not `b-diagnosis` or `B-Diagnosis`).
- Punctuation tokens get `O`.

Run `validate.py` on your file before submission — it checks column
structure, tag vocabulary, and BIO sequence validity, and produces the
`validation_report.txt` required in your Round 1 zip.

See `worked_example/` in this repository for a complete, correctly tagged
sample note in this exact format.

---

## 5. Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | [October 12, 2026] | Initial release |

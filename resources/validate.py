#!/usr/bin/env python3
"""
MedBhasha — Annotation Format Validator
=========================================

Checks a team's annotated corpus for format correctness BEFORE submission.
Run this locally and include the generated `validation_report.txt` in your
final zip, as required by the submission checklist.

USAGE
-----
    python validate.py --input annotated_data.csv --scheme BIO

    --input   Path to your annotated corpus (CSV or TSV, one row per token)
    --scheme  "BIO" (default) or "BIOES"
    --sep     Column separator, default "," (use "\\t" for tab-separated files)

EXPECTED FILE FORMAT
---------------------
One row per token, with a blank row (or a change in `note_id`) marking
sentence/note boundaries. Required columns:

    note_id, token, tag

Example:
    note_id,token,tag
    n001,রোগী,O
    n001,ডায়াবেটিস,B-DIAGNOSIS
    n001,আছে,O
    n002,মেটফরমিন,B-MEDICATION
    n002,৫০০মিগ্রা,B-DOSAGE

Tag values must exactly match the released entity schema (edit the
ENTITY_TYPES list below once the final schema is published) and follow
valid BIO/BIOES structure — no orphan I-/E- tags, no mismatched entity
types within a span, etc.

This script does not check annotation QUALITY (that's done by the
clinical advisor) — only structural/format correctness.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# CONFIG — update this list to match the officially released entity schema.
# Use the exact category names that will appear in tags, e.g. "DIAGNOSIS"
# for tags "B-DIAGNOSIS" / "I-DIAGNOSIS".
# ---------------------------------------------------------------------------
ENTITY_TYPES = [
    "DIAGNOSIS",
    "SYMPTOM",
    "MEDICATION",
    "DOSAGE",
    "TEST",
    "TEST_RESULT",
    "BODY_SITE",
    "PROCEDURE",
    "TEMPORAL",
    "NEGATION_UNCERTAINTY",
]

REQUIRED_COLUMNS = ["note_id", "token", "tag"]


def load_rows(path, sep):
    """Read the file into a list of dict rows, preserving row order."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=sep)
        rows = list(reader)
    return rows, reader.fieldnames


def check_columns(fieldnames, errors):
    fieldnames = fieldnames or []
    missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing:
        errors.append(
            f"[FATAL] Missing required column(s): {missing}. "
            f"Found columns: {fieldnames}"
        )
    return not missing


def build_tag_vocab(scheme):
    prefixes = ["B", "I"] if scheme == "BIO" else ["B", "I", "E", "S"]
    vocab = {"O"}
    for p in prefixes:
        for et in ENTITY_TYPES:
            vocab.add(f"{p}-{et}")
    return vocab


def group_by_note(rows):
    notes = defaultdict(list)
    for i, row in enumerate(rows):
        note_id = row.get("note_id", "").strip()
        notes[note_id].append((i, row))
    return notes


def validate_tag_vocab(rows, valid_tags, errors, warnings):
    bad_tag_counter = Counter()
    empty_token_lines = []
    for i, row in enumerate(rows, start=2):  # +2: header is line 1
        tag = (row.get("tag") or "").strip()
        token = (row.get("token") or "").strip()
        note_id = (row.get("note_id") or "").strip()

        if not note_id:
            errors.append(f"[Line {i}] Missing note_id.")
        if not token:
            empty_token_lines.append(i)
        if tag not in valid_tags:
            bad_tag_counter[tag] += 1
            if len(bad_tag_counter) <= 20:  # avoid flooding the report
                errors.append(
                    f"[Line {i}] Unrecognized tag '{tag}' for note '{note_id}'. "
                    f"Check spelling/casing against the released schema."
                )

    if empty_token_lines:
        warnings.append(
            f"{len(empty_token_lines)} row(s) have an empty token field "
            f"(lines: {empty_token_lines[:10]}{'...' if len(empty_token_lines) > 10 else ''})."
        )
    if len(bad_tag_counter) > 20:
        errors.append(
            f"[FATAL] {len(bad_tag_counter)} distinct invalid tag values found in total "
            f"(only first 20 shown above). Most common: {bad_tag_counter.most_common(5)}"
        )
    return bad_tag_counter


def validate_bio_sequence(notes, scheme, errors, warnings):
    """
    Check that tag sequences are structurally valid within each note:
      - No I-/E- tag appears without a preceding B-/I- of the SAME entity type
      - (BIOES only) S- and E- used consistently
    """
    for note_id, entries in notes.items():
        prev_tag = "O"
        prev_type = None
        for line_no, row in entries:
            tag = (row.get("tag") or "").strip()
            if tag == "O" or tag not in _all_valid_tags_cache:
                prev_tag, prev_type = tag, None
                continue

            prefix, _, etype = tag.partition("-")

            prev_prefix = prev_tag.partition("-")[0] if prev_tag != "O" else "O"

            if prefix in ("I", "E"):
                if prev_type != etype or prev_prefix not in _continuation_allowed[scheme].get(prefix, ()):
                    errors.append(
                        f"[Note {note_id}, line {line_no + 2}] Tag '{tag}' does not "
                        f"follow a valid preceding tag ('{prev_tag}') of the same entity "
                        f"type — orphan or mismatched continuation."
                    )
            prev_tag, prev_type = tag, etype


# Which previous tag prefixes are allowed to precede a continuation tag
_continuation_allowed = {
    "BIO": {"I": ("B", "I")},
    "BIOES": {"I": ("B", "I"), "E": ("B", "I")},
}


def summarize(rows, notes, errors, warnings):
    entity_counter = Counter()
    for row in rows:
        tag = (row.get("tag") or "").strip()
        if tag.startswith(("B-", "S-")):
            entity_counter[tag[2:]] += 1

    print("\n--- Corpus Summary ---")
    print(f"Total notes:  {len(notes)}")
    print(f"Total tokens: {len(rows)}")
    print("Entity counts:")
    if entity_counter:
        for etype, count in entity_counter.most_common():
            print(f"   {etype:<25} {count}")
    else:
        print("   (none found — check tagging)")

    missing_types = [et for et in ENTITY_TYPES if et not in entity_counter]
    if missing_types:
        warnings.append(
            f"No annotations found for entity type(s): {missing_types}. "
            f"This may be fine (rare categories) or may indicate a labeling gap."
        )


def main():
    parser = argparse.ArgumentParser(description="MedBhasha annotation format validator")
    parser.add_argument("--input", required=True, help="Path to annotated corpus file")
    parser.add_argument("--scheme", choices=["BIO", "BIOES"], default="BIO")
    parser.add_argument("--sep", default=",", help="Column separator (default: ',')")
    parser.add_argument("--report", default="validation_report.txt", help="Output report path")
    args = parser.parse_args()

    sep = "\t" if args.sep in ("\\t", "tab") else args.sep

    errors, warnings = [], []

    try:
        rows, fieldnames = load_rows(args.input, sep)
    except FileNotFoundError:
        print(f"[FATAL] File not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] Could not read file: {e}")
        sys.exit(1)

    if not check_columns(fieldnames, errors):
        _write_report(args.report, errors, warnings, passed=False)
        sys.exit(1)

    global _all_valid_tags_cache
    _all_valid_tags_cache = build_tag_vocab(args.scheme)

    validate_tag_vocab(rows, _all_valid_tags_cache, errors, warnings)
    notes = group_by_note(rows)
    validate_bio_sequence(notes, args.scheme, errors, warnings)
    summarize(rows, notes, errors, warnings)

    passed = len(errors) == 0
    _write_report(args.report, errors, warnings, passed)

    print(f"\n{'PASSED' if passed else 'FAILED'} — see {args.report} for full details.")
    print(f"Errors: {len(errors)} | Warnings: {len(warnings)}")
    sys.exit(0 if passed else 1)


def _write_report(path, errors, warnings, passed):
    with open(path, "w", encoding="utf-8") as f:
        f.write("MedBhasha Annotation Validation Report\n")
        f.write("=======================================\n\n")
        f.write(f"Result: {'PASS' if passed else 'FAIL'}\n")
        f.write(f"Errors: {len(errors)}\n")
        f.write(f"Warnings: {len(warnings)}\n\n")

        if errors:
            f.write("ERRORS\n------\n")
            for e in errors:
                f.write(f"- {e}\n")
            f.write("\n")

        if warnings:
            f.write("WARNINGS\n--------\n")
            for w in warnings:
                f.write(f"- {w}\n")


if __name__ == "__main__":
    main()

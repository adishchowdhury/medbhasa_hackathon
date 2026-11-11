# Data directory

| Path | What it is | Provenance |
|---|---|---|
| `round1/annotated_corpus.csv` | **Final annotated corpus**, BIO, columns `note_id,token,tag` (official format) | team-authored synthetic notes (LLM-assisted), see `round1_deliverables/SOURCE_DECLARATION.md` |
| `round1/notes_raw.csv` | plain text of every note (`note_id,text`) used by the retriever. For `n001–n040` the text is re-joined from tokens (the Round-1 raw text was not kept). | same |
| `round1/note_metadata.csv` | `note_id,domain,source_category,generator,batch,annotation_method` for all notes (the Round-1 file covered only 20 notes) | same |
| `round1/annotation_sources/*.txt` | hand-annotated notes in span markup `{DX\|…}`; `scripts/build_corpus.py` turns them into the BIO CSV | same |
| `round1/round1_submitted_annotated_corpus.csv` | the 40 notes exactly as submitted in Round 1 (kept for the fix log) | same |
| `round2/test_notes.csv` | 1 691 **unlabeled** notes released by the organisers (BOM removed) | COMSYS Hackathon-8 organisers; used only as input |
| `round2/test_queries.csv` | 30 Bengali test queries released by the organisers (BOM removed) | COMSYS Hackathon-8 organisers |

Everything under `round1/` is fictional; no real patient information is present.

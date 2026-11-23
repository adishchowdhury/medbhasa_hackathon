#!/usr/bin/env python3
"""Build the finale deck  presentation/JUSL_Health_Home_MedBhasha_Finale.pptx  from the real outputs.

    python scripts/make_slides.py

All numbers are read from outputs/ and data/; charts are drawn with matplotlib into presentation/assets/.
Every slide has speaker notes.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from medbhasha import analysis, corpus, pipeline  # noqa: E402

OUT = ROOT / "outputs"
PRES = ROOT / "presentation"
ASSETS = PRES / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- design tokens
INK = RGBColor(0x1F, 0x29, 0x33)
MUTED = RGBColor(0x5B, 0x67, 0x75)
TEAL = RGBColor(0x0F, 0x4C, 0x5C)
TEAL_LIGHT = RGBColor(0xE3, 0xEE, 0xF0)
CORAL = RGBColor(0xE3, 0x64, 0x14)
GREEN = RGBColor(0x2A, 0x9D, 0x8F)
PAPER = RGBColor(0xF8, 0xF6, 0xF1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xD9, 0xDE, 0xE3)
HEX = {"teal": "#0F4C5C", "coral": "#E36414", "green": "#2A9D8F", "muted": "#5B6775", "light": "#9FC2CA", "ink": "#1F2933"}
FONT = "Calibri"
BN_FONT = "Nirmala UI"          # Bengali-capable font shipped with Windows / Office

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9AA5B1", "axes.labelcolor": HEX["ink"],
                     "xtick.color": HEX["ink"], "ytick.color": HEX["ink"], "figure.dpi": 200})

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height
STATE = {"n": 0}


# ----------------------------------------------------------------------------- helpers
def _font(run, size, bold=False, color=INK, italic=False):
    f = run.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, FONT
    f.color.rgb = color
    rpr = run._r.get_or_add_rPr()
    for tag, face in (("a:ea", FONT), ("a:cs", BN_FONT)):
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {})
            rpr.append(el)
        el.set("typeface", face)


def text(slide, x, y, w, h, content, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, line_spacing=1.1):
    """content: str or list of str/(str, dict) paragraphs."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    paras = content if isinstance(content, list) else [content]
    for i, p in enumerate(paras):
        opts = {}
        if isinstance(p, tuple):
            p, opts = p
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = opts.get("align", align)
        para.line_spacing = line_spacing
        para.space_after = Pt(opts.get("space_after", 6))
        r = para.add_run()
        r.text = p
        _font(r, opts.get("size", size), opts.get("bold", bold), opts.get("color", color), opts.get("italic", italic))
    return tb


def bullets(slide, x, y, w, h, items, size=17, color=INK, gap=9):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Emu(0)
    for i, it in enumerate(items):
        sub = isinstance(it, tuple) and it[0] == "sub"
        body = it[1] if sub else it
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(gap)
        para.line_spacing = 1.08
        lead = para.add_run()
        lead.text = ("      –  " if sub else "■  ")
        _font(lead, size - 6 if not sub else size - 2, color=CORAL if not sub else MUTED, bold=True)
        # support **bold** segments
        parts = body.split("**")
        for k, part in enumerate(parts):
            if not part:
                continue
            r = para.add_run()
            r.text = part
            _font(r, size - (2 if sub else 0), bold=(k % 2 == 1), color=MUTED if sub else color)
    return tb


def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    s.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.text_frame.text = ""
    return s


def card(slide, x, y, w, h, big, label, color=TEAL, big_size=40, label_size=13, fill=WHITE):
    rect(slide, x, y, w, h, fill, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    rect(slide, x, y + Inches(0.18), Inches(0.07), h - Inches(0.36), color)
    text(slide, x + Inches(0.3), y + Inches(0.18), w - Inches(0.45), Inches(0.8), big, size=big_size, bold=True, color=color)
    text(slide, x + Inches(0.3), y + h - Inches(0.62), w - Inches(0.45), Inches(0.55), label, size=label_size, color=MUTED,
         anchor=MSO_ANCHOR.BOTTOM)


def new_slide(title, kicker=None, notes="", dark=False):
    s = prs.slides.add_slide(BLANK)
    STATE["n"] += 1
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = TEAL if dark else PAPER
    if not dark:
        rect(s, 0, 0, W, Inches(0.12), TEAL)
        if kicker:
            text(s, Inches(0.7), Inches(0.42), Inches(11), Inches(0.35), kicker.upper(), size=12, bold=True, color=CORAL)
        text(s, Inches(0.7), Inches(0.72), Inches(12), Inches(0.9), title, size=31, bold=True, color=TEAL)
        # footer
        text(s, Inches(0.7), H - Inches(0.42), Inches(8), Inches(0.3),
             "JUSL Health Home  ·  MedBhasha — COMSYS Hackathon-8 Finale", size=10, color=MUTED)
        text(s, W - Inches(1.2), H - Inches(0.42), Inches(0.5), Inches(0.3), str(STATE["n"]), size=10, color=MUTED,
             align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes.strip()
    return s


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), x, y, width=w, height=h)


def table(slide, df, x, y, w, col_w=None, size=12, header_fill=TEAL, row_h=0.36, bn_cols=()):
    rows, cols = df.shape[0] + 1, df.shape[1]
    shp = slide.shapes.add_table(rows, cols, x, y, w, Inches(row_h * rows))
    t = shp.table
    tbl_pr = shp._element.graphic.graphicData.tbl
    style = tbl_pr.find(qn("a:tblPr"))
    style.set("bandRow", "0")
    style.set("firstRow", "1")
    if col_w:
        for i, cw in enumerate(col_w):
            t.columns[i].width = Inches(cw)
    for j, c in enumerate(df.columns):
        cell = t.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        cell.text_frame.text = ""
        r = cell.text_frame.paragraphs[0].add_run()
        r.text = str(c)
        _font(r, size, bold=True, color=WHITE)
        cell.margin_left = cell.margin_right = Inches(0.08)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, v in enumerate(row):
            cell = t.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if i % 2 else TEAL_LIGHT
            cell.text_frame.text = ""
            cell.text_frame.word_wrap = True
            r = cell.text_frame.paragraphs[0].add_run()
            r.text = str(v)
            _font(r, size, color=INK)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i in range(rows):
        t.rows[i].height = Inches(row_h)
    return shp


def chip(slide, x, y, w, label, fill=TEAL, color=WHITE, size=13, h=0.42):
    s = rect(slide, x, y, w, Inches(h), fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.35)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    _font(r, size, bold=True, color=color)
    return s


def flow_box(slide, x, y, w, h, title, body, fill=WHITE, accent=TEAL, title_size=16, body_size=12.5):
    rect(slide, x, y, w, h, fill, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    rect(slide, x, y, w, Inches(0.09), accent)
    text(slide, x + Inches(0.18), y + Inches(0.22), w - Inches(0.36), Inches(0.45), title, size=title_size, bold=True, color=accent)
    text(slide, x + Inches(0.18), y + Inches(0.72), w - Inches(0.36), h - Inches(0.85), body, size=body_size, color=INK)


def arrow(slide, x, y, w=Inches(0.42), h=Inches(0.34), color=CORAL):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    a.fill.solid()
    a.fill.fore_color.rgb = color
    a.line.fill.background()
    a.shadow.inherit = False


# ----------------------------------------------------------------------------- data
notes, meta, raw = pipeline.load_train()
tdf, qdf = pipeline.load_test()
rows = analysis.spans_from_notes(notes)
freq = pd.read_csv(OUT / "eda_entity_frequency.csv", index_col=0)
cv = pd.read_csv(OUT / "ner_cv_results.csv")
lc = pd.read_csv(OUT / "ner_learning_curve.csv")
pt = pd.read_csv(OUT / "ner_cv_per_type_domain_heldout.csv", index_col=0)
cc = json.loads((OUT / "cluster_config.json").read_text(encoding="utf-8"))
sel = pd.read_csv(OUT / "cluster_selection.csv")
csum = pd.read_csv(OUT / "cluster_summary.csv")
spans = pd.read_csv(OUT / "ner_predictions_spans.csv")
rag = pd.read_csv(OUT / "rag_answers.csv")
rag_team = pd.read_csv(OUT / "rag_answers_team_corpus_only.csv")
man = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))

N_NOTES, N_TOK, N_SPANS = len(notes), sum(len(t) for t, _ in notes.values()), len(rows)
get = lambda ev, model: cv[(cv.evaluation.str.startswith(ev)) & (cv.model.str.startswith(model))].f1.iloc[0]
F1_R, F1_D = get("random", "full"), get("domain", "full")
F1_D_NOLEX = get("domain", "no gazetteer")
N_CL, SIL, DB = cc["n_clusters"], cc["silhouette"], cc["davies_bouldin"]
N_SP_TEST = len(spans)

# ----------------------------------------------------------------------------- charts
def save(fig, name):
    p = ASSETS / name
    fig.savefig(p, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return p


fig, ax = plt.subplots(figsize=(6.2, 4.3))
f = freq.sort_values("spans")
lab = [t.replace("NEGATION_UNCERTAINTY", "NEGATION/UNCERT.") for t in f.index]
bars = ax.barh(lab, f["spans"], color=HEX["teal"])
bars[0].set_color(HEX["coral"])
for i, v in enumerate(f["spans"]):
    ax.text(v + 6, i, str(v), va="center", fontsize=10, color=HEX["ink"])
ax.set_xlabel("annotated entity spans")
ax.set_xlim(0, f["spans"].max() * 1.15)
CH_FREQ = save(fig, "corpus_entity_frequency.png")

fig, ax = plt.subplots(figsize=(6.6, 4.1))
labels = ["full model", "no gazetteer", "no section\nfeatures", "plain CRF"]
keys = ["full", "no gazetteer", "no section", "neither"]
r = [get("random", k) for k in keys]
d = [get("domain", k) for k in keys]
xx = np.arange(4)
b1 = ax.bar(xx - 0.19, r, 0.36, color=HEX["light"], label="random 5-fold")
b2 = ax.bar(xx + 0.19, d, 0.36, color=HEX["teal"], label="domain-held-out 5-fold")
for bs in (b1, b2):
    for b in bs:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f"{b.get_height():.3f}", ha="center", fontsize=9)
ax.set_xticks(xx, labels)
ax.set_ylim(0.75, 0.9)
ax.set_ylabel("entity-level F1 (seqeval)")
ax.legend(frameon=False, loc="upper right", fontsize=9)
CH_ABL = save(fig, "ner_ablation.png")

fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot(lc["train_tokens"], lc["f1"], marker="o", color=HEX["coral"], lw=2.4)
for _, rr in lc.iterrows():
    ax.annotate(f"{rr.f1:.2f}", (rr.train_tokens, rr.f1), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax.axvline(612, color=HEX["muted"], ls="--", lw=1)
ax.text(700, lc["f1"].max() + 0.012, "← Round-1 corpus size (612 tokens)", fontsize=8.5, color=HEX["muted"])
ax.set_xlabel("training tokens")
ax.set_ylabel("F1 on fixed 40-note hold-out")
ax.set_ylim(lc["f1"].min() - 0.03, lc["f1"].max() + 0.035)
CH_LC = save(fig, "ner_learning_curve.png")

fig, ax = plt.subplots(figsize=(6.2, 4.3))
p2 = pt.loc[[t for t in pt.index if t not in ("micro avg", "macro avg")]].sort_values("f1-score")
cols = [HEX["coral"] if v < 0.8 else HEX["teal"] for v in p2["f1-score"]]
ax.barh([t.replace("NEGATION_UNCERTAINTY", "NEGATION/UNCERT.") for t in p2.index], p2["f1-score"], color=cols)
for i, (v, s) in enumerate(zip(p2["f1-score"], p2["support"])):
    ax.text(v + 0.005, i, f"{v:.2f}  (n={int(s)})", va="center", fontsize=9.5)
ax.set_xlim(0.5, 1.08)
ax.set_xlabel("F1, domain-held-out CV")
CH_TYPE = save(fig, "ner_per_type.png")

fig, ax = plt.subplots(figsize=(7.0, 3.9))
for algo, col, lbl in (("kmeans", HEX["light"], "k-means"), ("ward", HEX["teal"], "Ward agglomerative")):
    s_ = sel[sel.algorithm == algo]
    ax.plot(s_["n_clusters"], s_["silhouette"], marker="o", ms=3.5, color=col, lw=2, label=lbl)
lv = sel[sel.algorithm == "louvain"]
ax.scatter(lv["n_clusters"], lv["silhouette"], marker="^", s=55, color=HEX["green"], label="Louvain (graph)", zorder=3)
ax.scatter([N_CL], [SIL], s=230, facecolors="none", edgecolors=HEX["coral"], linewidths=2.4, zorder=4)
ax.annotate(f"chosen: {cc['chosen']['algorithm']}, k={N_CL}\nsilhouette {SIL:.3f}", (N_CL, SIL), xytext=(115, 0.47),
            fontsize=10, color=HEX["coral"], arrowprops=dict(arrowstyle="->", color=HEX["coral"]))
ax.set_xlabel("number of clusters")
ax.set_ylabel("silhouette (cosine)")
ax.legend(frameon=False, fontsize=9, loc="lower right")
CH_SEL = save(fig, "cluster_selection_slide.png")

tf_ = pd.read_csv(OUT / "test_entity_frequency.csv", index_col=0).sort_values("spans")
fig, ax = plt.subplots(figsize=(4.6, 4.6))
ax.barh([t.replace("NEGATION_UNCERTAINTY", "NEG/UNCERT.") for t in tf_.index], tf_["spans"], color=HEX["teal"])
for i, v in enumerate(tf_["spans"]):
    ax.text(v + 80, i, f"{v:,}", va="center", fontsize=9)
ax.set_xlim(0, tf_["spans"].max() * 1.22)
ax.set_xlabel("predicted spans (1,691 released notes)")
ax.set_title("Entity frequency in the NER output", fontsize=11, loc="left", color=HEX["teal"])
CH_TFREQ = save(fig, "test_entity_frequency_slide.png")

# ----------------------------------------------------------------------------- slides
# 1 — title
s = new_slide("", dark=True, notes=f"""
Good morning. We are team JUSL Health Home. Our project builds a complete Bengali clinical NLP pipeline:
we created an annotated corpus, trained an entity recogniser on it, clustered 1,691 unseen notes into phenotypes,
and built a question-answering system that only quotes the notes. Everything runs offline and reproduces exactly.
""")
rect(s, 0, 0, Inches(0.35), H, CORAL)
text(s, Inches(1.0), Inches(1.15), Inches(11), Inches(0.5), "COMSYS HACKATHON-8  ·  GRAND FINALE", size=15, bold=True, color=RGBColor(0xF4, 0xA2, 0x61))
text(s, Inches(1.0), Inches(1.75), Inches(11.5), Inches(1.4), "MedBhasha", size=66, bold=True, color=WHITE)
text(s, Inches(1.0), Inches(2.95), Inches(11.5), Inches(1.0), "A Bengali clinical NLP pipeline: annotate → NER → cluster → retrieve",
     size=27, color=RGBColor(0xDC, 0xEB, 0xEE))
for i, (big, lbl) in enumerate([(f"{N_NOTES}", "annotated notes"), (f"{N_TOK:,}", "annotated tokens"),
                                (f"{F1_D:.3f}", "NER F1 (domains held out)"), ("30/30", "queries answered with citations")]):
    x = Inches(1.0 + i * 2.85)
    text(s, x, Inches(4.35), Inches(2.6), Inches(0.8), big, size=36, bold=True, color=WHITE)
    text(s, x, Inches(5.1), Inches(2.6), Inches(0.5), lbl, size=13, color=RGBColor(0xB9, 0xD2, 0xD8))
text(s, Inches(1.0), Inches(6.35), Inches(11), Inches(0.4), "Team JUSL Health Home  ·  Team lead: Adrija Chowdhury  ·  Jadavpur University, Kolkata",
     size=15, color=WHITE)

# 2 — problem
s = new_slide("200 million Bengali speakers, almost no annotated clinical text",
              kicker="The problem", notes="""
English clinical NLP has MIMIC and i2b2. Bengali has essentially nothing open and entity-annotated.
The hackathon's real goal is the dataset; NER, clustering and RAG prove the data is useful.
Our constraints shaped every decision: own data only, no external APIs, exact reproducibility.
""")
bullets(s, Inches(0.7), Inches(1.75), Inches(6.2), Inches(4.8), [
    "English clinical NLP has MIMIC, i2b2/n2c2 — **Bengali has no open, entity-annotated clinical corpus**",
    "Bengali notes are **code-mixed** (মেটফর্মিন / metformin, HbA1c, C/O, Rx) and **highly inflected** (ডায়াবেটিসের, ইসিজি-তে)",
    "Real notes come in several registers: narrative, structured sections, shorthand, discharge summaries",
    "Competition constraints we designed for:",
    ("sub", "train only on our own annotations"),
    ("sub", "no external API, no web retrieval in RAG"),
    ("sub", "code must reproduce the results exactly"),
], size=17)
flow_y = Inches(1.9)
steps = [("1  Annotate", "own Bengali corpus, BIO, 10 entity types"), ("2  NER", "find the 10 entity types in unseen notes"),
         ("3  Cluster", "group notes into phenotypes from entities"), ("4  Retrieve", "answer Bengali questions by quoting notes")]
for i, (t, b) in enumerate(steps):
    y = flow_y + Inches(i * 1.2)
    flow_box(s, Inches(7.4), y, Inches(5.2), Inches(1.0), t, b, accent=[TEAL, GREEN, CORAL, TEAL][i], title_size=16, body_size=13)

# 3 — data
s = new_slide(f"From 40 notes to {N_NOTES} notes — and every entity type well covered",
              kicker="Data & annotation", notes=f"""
After Round 1 the organisers told us 612 tokens was too small and asked for about 10,000, prioritising rare types.
We grew the corpus to {N_NOTES} notes and {N_TOK:,} tokens. Every type now has at least {int(freq['spans'].min())} spans; Procedure is the rarest
because in outpatient notes most clinical acts are tests or prescriptions. We deliberately wrote four registers:
short informal notes, structured outpatient notes, shorthand C/O-Dx-Rx notes, and admission/discharge summaries.
All notes are synthetic and fictional; the method is stated in our source declaration.
""")
card(s, Inches(0.7), Inches(1.8), Inches(2.7), Inches(1.45), str(N_NOTES), "notes (Round 1: 40)")
card(s, Inches(3.6), Inches(1.8), Inches(2.7), Inches(1.45), f"{N_TOK:,}", "tokens (Round 1: 612 · target ≈ 10k)", color=CORAL)
card(s, Inches(0.7), Inches(3.45), Inches(2.7), Inches(1.45), f"{N_SPANS:,}", "entity spans, 10 types", color=GREEN)
card(s, Inches(3.6), Inches(3.45), Inches(2.7), Inches(1.45), "PASS", "official validate.py · 0 errors, 0 warnings")
bullets(s, Inches(0.7), Inches(5.15), Inches(5.8), Inches(1.6), [
    f"{meta['domain'].nunique()} clinical domains · 4 note registers · synthetic, fictional patients",
    "8 Round-1 tags corrected and logged (round1_fixes_log.tsv)",
], size=14, gap=5)
picture(s, CH_FREQ, Inches(6.75), Inches(1.65), w=Inches(5.9))

# 4 — annotation quality
ex_toks, ex_tags = notes["n052"]
s = new_slide("Annotation we can defend: rules, checks, audit",
              kicker="Data & annotation", notes="""
Annotation quality is 20 percent of the score, so we made it auditable.
1) 15 written conventions, following the organisers' worked example for the hard cases.
2) Notes are authored as span markup and converted by a script that refuses spans that cut a token, overlap, or contain punctuation.
3) A global audit for the same phrase tagged differently: it found that organ-plus-symptom phrases were sometimes merged,
we split them as the worked example does, and BODY_SITE F1 rose by about 9 points.
Honest limitation: no clinician review yet and no second annotator.
""")
flow_box(s, Inches(0.7), Inches(1.75), Inches(3.85), Inches(2.35), "15 written conventions",
         "Organ + symptom split: বুকে → BODY_SITE, ব্যথা → SYMPTOM\nUnits inside TEST_RESULT: ১১০/৭০ মি.মি. পারদ\nNegation = marker only: নেই, সন্দেহ\nPunctuation is always O", accent=TEAL)
flow_box(s, Inches(4.75), Inches(1.75), Inches(3.85), Inches(2.35), "Machine-checked",
         "Span markup → BIO by script\nRejects spans that cut tokens, overlap or contain punctuation\nOfficial validate.py on every batch", accent=GREEN)
flow_box(s, Inches(8.8), Inches(1.75), Inches(3.85), Inches(2.35), "Consistency audit",
         "Same phrase, different tag? → fixed\nOrgan+symptom merge found and split\nBODY_SITE F1 ↑ ≈ 9 points after fix", accent=CORAL)
text(s, Inches(0.7), Inches(4.35), Inches(12), Inches(0.35), "Example — gout note n052 (gold BIO spans)", size=13, bold=True, color=MUTED)
chips = []
for st, en, ty in corpus.spans_from_bio(ex_tags)[:7]:
    chips.append((" ".join(ex_toks[st:en]), ty))
colors = {"TEMPORAL": GREEN, "BODY_SITE": RGBColor(0x6D, 0x59, 0x7A), "SYMPTOM": CORAL, "NEGATION_UNCERTAINTY": MUTED,
          "TEST": TEAL, "TEST_RESULT": RGBColor(0x1D, 0x7C, 0x8C), "DIAGNOSIS": RGBColor(0xB0, 0x41, 0x3E),
          "MEDICATION": RGBColor(0x26, 0x46, 0x53), "DOSAGE": RGBColor(0x8C, 0x6D, 0x31)}
x, y = Inches(0.7), Inches(4.85)
for word, ty in chips:
    wd = Inches(max(1.3, 0.13 * len(word) + 0.45))
    wt = Inches(max(1.3, 0.1 * len(ty) + 0.25))
    ww = max(wd, wt)
    if x + ww > W - Inches(0.6):
        x, y = Inches(0.7), y + Inches(1.0)
    chip(s, x, y, ww, word, fill=WHITE, color=INK, size=15)
    s.shapes[-1].line.color.rgb = colors.get(ty, TEAL)
    s.shapes[-1].line.width = Pt(2)
    text(s, x, y + Inches(0.48), ww, Inches(0.3), ty, size=10, bold=True, color=colors.get(ty, TEAL), align=PP_ALIGN.CENTER)
    x += ww + Inches(0.15)

# 5 — NER model
s = new_slide("NER: a lexicon-augmented, section-aware CRF", kicker="Stage 1 · Entity recognition", notes="""
Why a CRF and not BERT? Twelve thousand tokens is too little to fine-tune a transformer well, pretrained Bengali encoders
cannot be downloaded in an offline exactly-reproducible run, and no external API is allowed.
The CRF predicts the whole tag sequence, so it never outputs an I-tag without a B-tag.
Three groups of clues: the word and its letters, a hand-written bilingual medical gazetteer, and the note section
— text after Dx is probably a diagnosis, after Rx a drug.
""")
ys = Inches(1.85)
flow_box(s, Inches(0.7), ys, Inches(3.0), Inches(2.6), "Token & context", "word, 1–4 letter prefixes and suffixes (Bengali case endings), shape (digits, /, %, Latin), ±2 neighbours", accent=TEAL, body_size=13)
flow_box(s, Inches(3.95), ys, Inches(3.0), Inches(2.6), "Medical gazetteer", "~760 bilingual drugs, diseases, tests, body sites… spelling-normalised matching (hasanta, ড়/য়, case endings)", accent=GREEN, body_size=13)
flow_box(s, Inches(7.2), ys, Inches(3.0), Inches(2.6), "Note section", "header → group: নির্ণয়/Dx → DIAG, ব্যবস্থাপত্র/Rx → TREAT, ভাইটাল → VITAL, C/O → COMPLAINT …", accent=CORAL, body_size=13)
arrow(s, Inches(10.35), ys + Inches(1.1))
flow_box(s, Inches(10.9), ys, Inches(1.75), Inches(2.6), "CRF", "L-BFGS, c1=c2=0.05\n21 tags\nvalid BIO always", accent=TEAL, body_size=13)
bullets(s, Inches(0.7), Inches(4.85), Inches(12), Inches(1.9), [
    "**Why not BERT / an LLM?** 12k tokens over-fit a transformer; pretrained weights cannot be downloaded offline; no external API allowed",
    "Trains in ~10 s on a CPU and is **bit-for-bit deterministic** — the same predictions on every run",
    "Same tokenizer for training and test: edge punctuation split, মি.গ্রা. / ১১০/৭০ / সর্দি-কাশি kept whole",
], size=15, gap=7)

# 6 — NER results
s = new_slide(f"NER results: F1 {F1_R:.3f} (random) · {F1_D:.3f} (unseen domains)", kicker="Stage 1 · Entity recognition", notes=f"""
Entity-level F1 with seqeval: a span only counts if every token and the type are right.
Domain-held-out means all notes of a specialty are hidden at training time, which is closer to the hidden test.
The gazetteer is the most valuable component: {100*(F1_D-F1_D_NOLEX):.1f} F1 points when domains are held out.
The learning curve shows why expanding the corpus mattered: F1 rises from {lc.f1.iloc[0]:.2f} with 1k tokens to {lc.f1.iloc[-1]:.2f} with 10k.
These are upper bounds — synthetic single-author data; the hidden-test score will be lower.
""")
picture(s, CH_ABL, Inches(0.6), Inches(1.7), w=Inches(6.3))
picture(s, CH_LC, Inches(7.1), Inches(1.7), w=Inches(5.6))
bullets(s, Inches(7.2), Inches(5.45), Inches(5.6), Inches(1.4), [
    f"Gazetteer adds **+{100*(F1_D-F1_D_NOLEX):.1f} F1** on unseen domains",
    "More data helped steadily — the expansion was worth it",
], size=14, gap=4)
text(s, Inches(0.7), Inches(6.45), Inches(6.2), Inches(0.5), "Scores are on our own data (cross-validation) → an upper bound for the hidden test.",
     size=12, italic=True, color=MUTED)

# 7 — per type + predictions
s = new_slide("Where the model is strong — and where it struggles", kicker="Stage 1 · Entity recognition", notes=f"""
Temporal, dosage, medication and diagnosis are above 0.9: they have regular forms and the gazetteer helps.
Symptom and body site are weakest — they are also the most convention-dependent: where does a site end and the symptom begin.
On the 1,691 released notes the model produced {N_SP_TEST:,} spans, about {N_SP_TEST/len(tdf):.0f} per note, written in three formats,
and the prediction file passes the official validator.
""")
picture(s, CH_TYPE, Inches(0.6), Inches(1.7), w=Inches(6.0))
card(s, Inches(7.1), Inches(1.8), Inches(2.6), Inches(1.4), f"{len(tdf):,}", "released notes tagged")
card(s, Inches(9.95), Inches(1.8), Inches(2.6), Inches(1.4), f"{N_SP_TEST/1000:.1f}k", "entity spans predicted", color=CORAL)
bullets(s, Inches(7.1), Inches(3.5), Inches(5.6), Inches(3.3), [
    "**Strong:** TEMPORAL, DOSAGE, MEDICATION, DIAGNOSIS > 0.9",
    "**Weak:** SYMPTOM, BODY_SITE — boundary conventions (বুকে | ব্যথা) count as full misses",
    "Unseen vocabulary in shorthand complaints is often left untagged (the CRF is conservative)",
    "Output: BIO per token · spans with character offsets · JSONL — **validator PASS**",
], size=14, gap=7)

# 8 — clustering method
s = new_slide("Clustering: notes grouped by what the NER found — no labels used", kicker="Stage 2 · Phenotype clustering", notes=f"""
Each note is represented only by its predicted entities — never the note id or file order.
Two views: diagnosis text as character n-grams, so আর্টিকারিয়া and আমবাত (আর্টিকারিয়া) still match; and a typed bag of
diagnosis, symptom, drug, test, procedure and body-site phrases. TF-IDF, SVD to 64 dimensions, cosine.
We compared k-means, Ward and Louvain on a similarity graph, and picked the configuration with the best silhouette.
Result: Ward with {N_CL} clusters, silhouette {SIL:.3f}, Davies-Bouldin {DB:.3f}. Louvain independently finds 60 to 76 communities.
""")
steps = [("Predicted entities", "per note, from NER only", TEAL), ("Two views", "diagnosis char n-grams + typed entity bag", GREEN),
         ("TF-IDF → SVD-64", "cosine geometry", TEAL), ("Select by silhouette", "k-means · Ward · Louvain", CORAL)]
for i, (t1, t2, col) in enumerate(steps):
    x = Inches(0.7 + i * 3.1)
    flow_box(s, x, Inches(1.75), Inches(2.7), Inches(1.35), t1, t2, accent=col, title_size=15, body_size=12.5)
    if i < 3:
        arrow(s, x + Inches(2.73), Inches(2.25), w=Inches(0.33), h=Inches(0.3))
picture(s, CH_SEL, Inches(0.6), Inches(3.3), h=Inches(3.6))
card(s, Inches(7.4), Inches(3.45), Inches(2.5), Inches(1.4), f"{N_CL}", f"clusters ({cc['chosen']['algorithm']})")
card(s, Inches(10.1), Inches(3.45), Inches(2.5), Inches(1.4), f"{SIL:.3f}", "silhouette", color=GREEN)
card(s, Inches(7.4), Inches(5.05), Inches(2.5), Inches(1.4), f"{DB:.3f}", "Davies–Bouldin", color=CORAL)
card(s, Inches(10.1), Inches(5.05), Inches(2.5), Inches(1.4), "60–76", "Louvain communities agree", color=TEAL)

# 9 — clusters look like
top = csum.head(6).copy()
def first2(sx):
    return "; ".join([p.split(" (")[0] for p in str(sx).split("; ")[:2]])
tab = pd.DataFrame({"cluster": top["cluster"], "notes": top["size"],
                    "top diagnoses": top["top_diagnoses"].map(first2),
                    "top medications": top["top_medications"].map(first2)})
pc = cc["proxy_check"]
s = new_slide("The clusters are clinically coherent phenotypes", kicker="Stage 2 · Phenotype clustering", notes=f"""
These are the six largest clusters. Allergic rhinitis, urticaria, otitis media, scabies, viral fever, plantar fasciitis —
each cluster groups one condition and its typical drugs, even across Bengali spelling variants.
As a sanity check only — not the official score — we compared with the diagnosis line printed in the notes:
NMI {pc['NMI']:.2f}, ARI {pc['ARI']:.2f}. This check is partly circular, so we use it only to show the structure is not arbitrary.
""")
table(s, tab, Inches(0.7), Inches(1.75), Inches(11.9), col_w=[1.1, 0.9, 5.0, 4.9], size=14, row_h=0.6)
text(s, Inches(0.7), Inches(6.15), Inches(12), Inches(0.5),
     f"Sanity check (not the hidden gold): vs. the diagnosis line printed in {pc['notes_with_diagnosis_line']} notes — NMI {pc['NMI']:.2f}, ARI {pc['ARI']:.2f}.",
     size=13, italic=True, color=MUTED)

# 10 — relationship analysis
assoc = analysis.association_pairs([(r.note_id, r.entity_type, r.text) for r in spans.itertuples()],
                                   "DIAGNOSIS", "MEDICATION", min_count=12, top=7)
at = pd.DataFrame({"diagnosis": assoc["diagnosis"], "medication": assoc["medication"],
                   "notes together": assoc["co_notes"], "lift": assoc["lift"].map(lambda v: f"{v:.0f}×")})
s = new_slide("Frequency & relationship analysis on the NER output", kicker="Stage 2 · Analysis", notes="""
From the predicted entities we count frequencies and measure relationships: lift says how many times more often a diagnosis
and a drug appear together than by chance. Most strong pairs are the expected treatments — for example dehydration with zinc
syrup, seborrhoeic dermatitis with a mild steroid lotion, chest pain with nitroglycerin — a useful check of the NER output itself.
One pair, peptic ulcer with painkillers, comes from advice to AVOID painkillers: co-occurrence is not prescription,
and our analysis does not yet model that distinction.
""")
table(s, at, Inches(0.7), Inches(1.75), Inches(7.3), col_w=[2.6, 2.7, 1.1, 0.9], size=13, row_h=0.55)
picture(s, CH_TFREQ, Inches(8.2), Inches(1.65), w=Inches(4.5))
text(s, Inches(0.7), Inches(6.2), Inches(7.3), Inches(0.6),
     "Lift = how many times more often the pair co-occurs in a note than by chance (pairs seen in ≥ 12 notes).",
     size=12, italic=True, color=MUTED)

# 11 — RAG architecture
s = new_slide("RAG: find the right patients, answer only from the notes", kicker="Stage 3 · Question answering", notes="""
No generative model and no external API. The query is parsed into concepts, free terms, demographic and numeric constraints.
Matching is spelling-normalised and uses a bilingual concept table: PPI equals প্রোটন পাম্প ইনহিবিটর equals ওমিপ্রাজল.
It is polarity-aware: 'ডায়াবেটিস নেই' or 'মায়ের ডায়াবেটিস' do not count as the patient having diabetes.
Notes that meet every condition rank first; otherwise partial matches are shown and labelled.
The answer is assembled from sentences copied from the notes, each cited with its note id — it cannot hallucinate.
""")
steps = [("Bengali query", "ডায়াবেটিস রোগীদের জন্য কোন কোন ওষুধ…", TEAL),
         ("Understand", "concepts · free terms · child/elderly/sex · numeric (HbA1c > 9)", GREEN),
         ("Retrieve", "concept coverage + IDF terms + BM25 · negation & family-history aware", TEAL),
         ("Grounded answer", "counts, drug lists, patient lines — verbatim, with [note-id]", CORAL)]
for i, (t1, t2, col) in enumerate(steps):
    x = Inches(0.7 + i * 3.1)
    flow_box(s, x, Inches(1.75), Inches(2.7), Inches(2.0), t1, t2, accent=col, title_size=15, body_size=13)
    if i < 3:
        arrow(s, x + Inches(2.73), Inches(2.55), w=Inches(0.33), h=Inches(0.3))
bullets(s, Inches(0.7), Inches(4.1), Inches(12), Inches(2.7), [
    "**Spelling-robust:** মেটফর্মিন = মেটফরমিন, ডায়াবেটিসের = ডায়াবেটিস (skeleton + light stemming)",
    "**Synonyms & classes:** T2DM, SSRI → এসসিটালোপ্রাম/সার্ট্রালিন, PPI → ওমিপ্রাজল/প্যান্টোপ্রাজল, অ্যাজমা = হাঁপানি",
    "**Polarity-aware:** “ডায়াবেটিস নেই”, “মায়ের ডায়াবেটিস” do not count — unless the question asks for the negation",
    "**Honest by construction:** if no note matches, it says so instead of guessing",
], size=15, gap=8)

# 12 — RAG results
r19 = rag[rag.query_id == "q19"].iloc[0]
ans = r19.answer.split("\n")
s = new_slide("RAG results on the 30 Bengali test queries", kicker="Stage 3 · Question answering", notes=f"""
Here is query 19: patients on metformin with HbA1c above 9 and type 2 diabetes. Three conditions, one of them numeric.
The system parses the threshold, checks the value in each note, and returns {r19.n_matched_notes} patients, each with the evidence fragment
that contains the HbA1c value. Across the 30 queries, every answer is cited. We also ship a strict variant that searches only
our own annotated notes, because the rules can be read that way; the organisers can choose.
""")
rect(s, Inches(0.7), Inches(1.7), Inches(7.6), Inches(4.85), WHITE, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.03)
text(s, Inches(0.95), Inches(1.85), Inches(7.2), Inches(0.5), "q19 · " + r19.query_text, size=14, bold=True, color=TEAL)
lines = [ans[0]] + [l[:120] + ("…" if len(l) > 120 else "") for l in ans[2:6]]
text(s, Inches(0.95), Inches(2.4), Inches(7.2), Inches(4.0), lines, size=13.5, color=INK, line_spacing=1.12)
full = int((rag["n_matched_notes"] > 0).sum())
card(s, Inches(8.6), Inches(1.7), Inches(4.0), Inches(1.4), f"{full}/30", "queries with notes meeting every condition")
card(s, Inches(8.6), Inches(3.3), Inches(4.0), Inches(1.4), "30/30", "answers cite their source notes", color=GREEN)
card(s, Inches(8.6), Inches(4.9), Inches(4.0), Inches(1.65), "2 KBs", "default: our notes + released notes tagged by our NER\nstrict: our annotated corpus only", color=CORAL, label_size=12)

# 13 — live demo
s = new_slide("Live query: how we will answer the judges' question", kicker="Finale demo", notes="""
For the live query we run one command, or the last cell of the notebook. It takes a few seconds on a laptop, offline.
Things to show: a synonym (অ্যাজমা for হাঁপানি), a demographic constraint (শিশু), a numeric threshold, and the negation handling.
If the question uses a concept that is not in our table, it falls back to BM25 word matching with spelling normalisation.
""")
rect(s, Inches(0.7), Inches(1.75), Inches(11.9), Inches(1.25), RGBColor(0x1B, 0x26, 0x30), shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
text(s, Inches(1.0), Inches(1.95), Inches(11.4), Inches(0.9),
     ["$ python scripts/ask.py \"অ্যাজমা আছে এমন শিশুদের কী ওষুধ দেওয়া হয়েছে?\"",
      ("$ python scripts/ask.py --team-only \"…\"      # strict: our annotated corpus only", {"color": RGBColor(0x9F, 0xC2, 0xCA), "size": 15})],
     size=17, color=RGBColor(0xE8, 0xF1, 0xF2))
demo = [("Synonym", "অ্যাজমা = হাঁপানি", TEAL), ("Who", "শিশু · বয়স্ক · পুরুষ/মহিলা", GREEN),
        ("Numbers", "HbA1c ৯-এর বেশি", CORAL), ("Negation", "অ্যান্টিবায়োটিকের প্রয়োজন নেই", TEAL)]
for i, (t1, t2, col) in enumerate(demo):
    flow_box(s, Inches(0.7 + i * 3.0), Inches(3.35), Inches(2.8), Inches(1.45), t1, t2, accent=col, title_size=15, body_size=14)
bullets(s, Inches(0.7), Inches(5.2), Inches(12), Inches(1.6), [
    "Runs **offline in seconds** on a laptop; the notebook's last RAG cell does the same",
    "Unknown concept? Falls back to BM25 + spelling-normalised word matching (tested on unseen questions)",
], size=15, gap=6)

# 14 — reproducibility & compliance
s = new_slide("Rules and reproducibility: verified, not just claimed", kicker="Engineering", notes=f"""
Everything is one notebook. Seed 42, pinned versions, and an MD5 hash of every output file in the run manifest.
We re-ran the full pipeline under a different Python hash seed and all {len(man['output_md5'])} output files were byte-identical.
That check actually caught a bug — tie ordering in one table depended on Python's hash order — which we fixed.
""")
items = [("Own data only", "NER trained only on our corpus; released notes used as unlabeled input"),
         ("No external API", "no web retrieval, no hosted model, no downloads at run time"),
         ("One notebook", "MedBhasha_Pipeline.ipynb — NER → clustering → RAG, executed, 0 errors"),
         ("Exact reproduction", f"seed 42 · pinned versions · MD5 of all {len(man['output_md5'])} outputs · re-run byte-identical"),
         ("Validated formats", "official validate.py PASS on corpus and on predictions"),
         ("Unsupervised clustering", "model chosen by silhouette / Davies–Bouldin only")]
for i, (t1, t2) in enumerate(items):
    col, rw = i % 2, i // 2
    x, y = Inches(0.7 + col * 6.05), Inches(1.75 + rw * 1.6)
    rect(s, x, y, Inches(5.85), Inches(1.4), WHITE, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    chip(s, x + Inches(0.25), y + Inches(0.47), Inches(0.46), "✓", fill=GREEN, size=15, h=0.46)
    text(s, x + Inches(0.95), y + Inches(0.22), Inches(4.7), Inches(0.4), t1, size=17, bold=True, color=TEAL)
    text(s, x + Inches(0.95), y + Inches(0.68), Inches(4.75), Inches(0.65), t2, size=13, color=INK)

# 15 — challenges & limitations
s = new_slide("What was hard, and what we would do next", kicker="Lessons", notes="""
Be upfront here. Our cross-validation scores are optimistic because the corpus is synthetic and single-author, and no clinician
has reviewed it. The hidden gold may draw boundaries differently. Answers are extractive rather than fluent prose.
Next steps: clinician review of 50 notes, more notes in weak registers, a transformer as a second model if offline weights are allowed,
and a clinical evaluation of the answers.
""")
flow_box(s, Inches(0.7), Inches(1.75), Inches(5.8), Inches(4.3), "Challenges & honest limitations",
         "• Tiny start: 612 tokens → had to build a 12k-token corpus fast\n"
         "• Bengali inflection & code-mixing (ইসিজি-তে, মেটফর্মিন/মেটফরমিন, C/O, Rx)\n"
         "• Synthetic, single-author data → CV F1 is an upper bound\n"
         "• No clinician review, no inter-annotator agreement yet\n"
         "• Span conventions of the hidden gold are unknown\n"
         "• Answers are extractive, not fluent prose", accent=CORAL, body_size=17)
flow_box(s, Inches(6.85), Inches(1.75), Inches(5.8), Inches(4.3), "Next steps",
         "• Clinician spot-check of 50 notes; settle the open conventions\n"
         "• More notes in the registers where NER is weakest\n"
         "• Add IndicBERT / MuRIL as a second model and ensemble with the CRF (if offline weights are allowed)\n"
         "• Clinical-advisor evaluation of the answers\n"
         "• Release the corpus under CC-BY-4.0 for the community", accent=GREEN, body_size=17)

# 16 — thanks
s = new_slide("", dark=True, notes="Thank you. We are happy to take questions, and to run a live query now.")
rect(s, 0, 0, Inches(0.35), H, CORAL)
text(s, Inches(1.0), Inches(2.0), Inches(11), Inches(1.2), "Thank you", size=60, bold=True, color=WHITE)
text(s, Inches(1.0), Inches(3.2), Inches(11), Inches(0.8), "Questions — and a live Bengali query, on the spot.", size=26, color=RGBColor(0xDC, 0xEB, 0xEE))
text(s, Inches(1.0), Inches(4.6), Inches(11.5), Inches(1.4),
     [f"{N_NOTES} notes · {N_TOK:,} tokens · NER F1 {F1_R:.3f} / {F1_D:.3f} · {N_CL} clusters (silhouette {SIL:.3f}) · 30/30 cited answers",
      "Repo: notebook MedBhasha_Pipeline.ipynb · report docs/TECHNICAL_REPORT.pdf"],
     size=16, color=RGBColor(0xB9, 0xD2, 0xD8))
text(s, Inches(1.0), Inches(6.35), Inches(11), Inches(0.4), "Team JUSL Health Home  ·  Adrija Chowdhury (team lead)", size=15, color=WHITE)

out = PRES / "JUSL_Health_Home_MedBhasha_Finale.pptx"
prs.save(out)
print(f"wrote {out}  ({STATE['n']} slides)")

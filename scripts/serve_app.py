#!/usr/bin/env python3
"""Optional bonus-track demo: a tiny local web page for live queries (standard library only).

    python scripts/serve_app.py [--port 8765] [--team-only]

Open http://127.0.0.1:8765 and type a Bengali question. Needs outputs/ner_predictions_spans.csv
(created by the pipeline) unless --team-only is given. Nothing leaves your machine.
"""
import argparse
import html
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medbhasha.ask import ask, build_retriever  # noqa: E402

PAGE = """<!doctype html><html lang="bn"><head><meta charset="utf-8"><title>MedBhasha — JUSL Health Home</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:system-ui,'Noto Sans Bengali',sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;line-height:1.55}}
textarea{{width:100%;font-size:1.05rem;padding:.6rem}}button{{padding:.55rem 1.1rem;font-size:1rem}}
pre{{white-space:pre-wrap;background:#f4f6f8;padding:1rem;border-radius:8px}}small{{color:#555}}</style></head><body>
<h2>MedBhasha — ক্লিনিক্যাল নোট থেকে প্রশ্নোত্তর</h2>
<p><small>Local, offline, extractive answers with [note-id] citations. Knowledge base: {kb}.</small></p>
<form method="post"><textarea name="q" rows="3" placeholder="যেমন: হাঁপানি আছে ও রাতে কাশি হয় এমন রোগীদের জন্য কোন কোন ওষুধ নথিভুক্ত করা আছে?">{q}</textarea>
<p><button>উত্তর দিন</button></p></form>{out}</body></html>"""


def make_handler(retriever, kb):
    class H(BaseHTTPRequestHandler):
        def _send(self, q="", out=""):
            body = PAGE.format(q=html.escape(q), out=out, kb=kb).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            q = parse_qs(self.rfile.read(n).decode("utf-8")).get("q", [""])[0].strip()
            if not q:
                return self._send()
            r = ask(retriever, q)
            out = (f"<h3>উত্তর</h3><pre>{html.escape(r['answer'])}</pre>"
                   f"<p><small>detected concepts: {html.escape(', '.join(r['concepts']) or '-')} · "
                   f"matched notes: {r['n_matched']}</small></p>")
            self._send(q, out)

        def log_message(self, *a):
            pass
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--team-only", action="store_true")
    a = ap.parse_args()
    R = build_retriever(a.team_only)
    kb = "team corpus only" if a.team_only else "team corpus + released notes annotated by our NER"
    print(f"serving on http://127.0.0.1:{a.port}  ({kb})")
    HTTPServer(("127.0.0.1", a.port), make_handler(R, kb)).serve_forever()

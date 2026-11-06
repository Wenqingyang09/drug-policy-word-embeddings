"""
find_source.py

Finds which original document a quoted sentence comes from.
It searches every .txt, .pdf and .docx file in the Desktop folders whose
names start with "Data Source" (including subfolders) for a phrase and prints
the file name, the surrounding text, and for Chinese laws the nearest article number (第…条) before the match.

The intermediate corpus files (china_corpus*, usa_corpus*) are skipped.

Matching ignores line breaks, extra spaces, hyphenation at line ends and
upper/lower case.

Run (one phrase at a time, keep the quotation marks):
  conda activate thesis
  python ~/Desktop/find_source.py "设置社区戒毒康复工作站"
  python ~/Desktop/find_source.py "a few words copied from the US sentence"
"""

import os
import re
import sys
import logging
import unicodedata

logging.disable(logging.WARNING)   # hide PDF colour warnings

DESK = os.path.expanduser("~/Desktop")
SKIP_DIRS = {"results_v1", "__pycache__", ".git"}
# intermediate corpus files: they cannot tell which document a sentence is from
SKIP_PREFIX = ("china_corpus", "usa_corpus")


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"-\s*\n\s*", "", s)          # hyphen at line end
    s = re.sub(r"\s+", " ", s).lower()
    s = re.sub(r"(?<=[一-鿿]) (?=[一-鿿])", "", s)  # spaces inside Chinese
    return s


def read_pdf(path):
    for lib in ("pypdf", "fitz", "pdfplumber"):   # fastest first
        try:
            if lib == "pdfplumber":
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    return "\n".join(p.extract_text() or "" for p in pdf.pages)
            if lib == "pypdf":
                from pypdf import PdfReader
                return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
            import fitz
            return "\n".join(p.get_text() for p in fitz.open(path))
        except Exception:
            continue
    return ""


def read_docx(path):
    try:
        import docx
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    except Exception:
        return ""


def read(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return open(path, encoding="utf-8", errors="ignore").read()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    return ""


if len(sys.argv) < 2:
    sys.exit('Give a phrase, e.g.  python find_source.py "设置社区戒毒康复工作站"')
phrase = norm(" ".join(sys.argv[1:]))
print(f"Searching for: {phrase}\n")

hits = 0
# only the data folders (names starting with "Data Source") are searched
roots = [os.path.join(DESK, d) for d in sorted(os.listdir(DESK))
         if d.startswith("Data Source") and os.path.isdir(os.path.join(DESK, d))]
for root, dirs, files in (w for r in roots for w in os.walk(r)):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
    for fn in sorted(files):
        if (not fn.lower().endswith((".txt", ".pdf", ".docx")) or fn.startswith(("~$", ".~"))
                or fn.startswith(SKIP_PREFIX)):
            continue
        path = os.path.join(root, fn)
        print(f"   reading {os.path.relpath(path, DESK)}", flush=True)
        text = norm(read(path))
        for m in re.finditer(re.escape(phrase), text):
            hits += 1
            before = text[:m.start()]
            art = re.findall(r"第[一二三四五六七八九十百零〇0-9]+条", before)
            print("=" * 70)
            print(f"File   : {os.path.relpath(path, DESK)}")
            if art:
                print(f"Article: {art[-1]} (nearest article number before the match)")
            print(f"Context: ...{text[max(0, m.start() - 200):m.end() + 200]}...")
print("=" * 70)
print(f"{hits} match(es).")
if hits == 0:
    print("No match: try a shorter piece of the sentence.")
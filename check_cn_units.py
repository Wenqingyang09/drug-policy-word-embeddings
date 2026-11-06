"""
check_cn_units.py

Read-only check of what one training unit is in the Chinese corpus.
Uses exactly the same rules as results_summary.py (split at line breaks,
keep lines of at least 5 characters with more than 3 tokens).

Prints: number of lines and units, unit length in characters and tokens,
how many units start with an article number (第…条) or a numbered item,
and 20 random units (first 120 characters).

Run:
  conda activate thesis
  python ~/Desktop/check_cn_units.py
"""

import os
import re
import random
import unicodedata
import numpy as np
import jieba

jieba.add_word("吸毒人员")
DESK = os.path.expanduser("~/Desktop")


def noise(w):
    if not w.strip():
        return True
    if all(unicodedata.category(c)[0] in ("P", "S", "Z") for c in w):
        return True
    return re.fullmatch(r"[0-9.,%·]+", w) is not None


lines = open(os.path.join(DESK, "china_corpus_clean_v3.txt"), encoding="utf-8").read().split("\n")
units = []
for s in lines:
    s = s.strip()
    if len(s) < 5:
        continue
    ws = [w for w in jieba.cut(s) if not noise(w)]
    if len(ws) > 3:
        units.append((s, len(ws)))

chars = [len(s) for s, _ in units]
toks = [n for _, n in units]
art = sum(1 for s, _ in units if re.match(r"第[一二三四五六七八九十百零〇0-9]+条", s))
item = sum(1 for s, _ in units if re.match(r"[（(][一二三四五六七八九十0-9]+[）)]", s))
end = sum(1 for s, _ in units if s[-1] in "。；：！？")

print(f"lines in file: {len(lines)}   training units kept: {len(units)}")
print(f"characters per unit: median {np.median(chars):.0f}, mean {np.mean(chars):.0f}, max {max(chars)}")
print(f"tokens per unit:     median {np.median(toks):.0f}, mean {np.mean(toks):.1f}, max {max(toks)}")
print(f"units starting with an article number (第…条): {art} ({100 * art / len(units):.0f}%)")
print(f"units starting with a numbered item（一）(1): {item} ({100 * item / len(units):.0f}%)")
print(f"units ending with 。；：！？: {end} ({100 * end / len(units):.0f}%)")
print("20 random units:")
random.seed(1)
for s, n in random.sample(units, 20):
    print(f"  [{n} tokens] {s[:120]}")
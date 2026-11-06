"""
example_sentences.py

Draws 5 random sentences (fixed seed) containing the most medical and the most
punitive concept in each corpus, for the note under Figure 1.

  US: addiction (most medical), drug (most punitive, "Drug (general)")
  CN: 康复 (most medical), 贩毒 (most punitive)

Run:
  conda activate thesis
  python ~/Desktop/example_sentences.py
"""

import os
import re
import random

DESK = os.path.expanduser("~/Desktop")

us = open(os.path.join(DESK, "usa_corpus_clean.txt"), encoding="utf-8").read()
us = re.sub(r"-\n", "", us)
us = re.sub(r"\s+", " ", us)
us_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", us)]

cn = open(os.path.join(DESK, "china_corpus_clean_v3.txt"), encoding="utf-8").read()
cn_sents = [s.strip() for s in re.split(r"[。；\n]", cn)]

for label, sents, term, pattern in [
    ("US most medical", us_sents, "addiction", r"\baddiction\b"),
    ("US most punitive", us_sents, "drug", r"\bdrugs?\b"),
    ("CN most medical", cn_sents, "康复", "康复"),
    ("CN most punitive", cn_sents, "贩毒", "贩毒"),
]:
    pool = [s for s in sents if re.search(pattern, s, re.I) and 40 <= len(s) <= 300]
    random.seed(1)
    pick = random.sample(pool, min(5, len(pool)))
    print("=" * 70)
    print(f"{label}: '{term}' ({len(pool)} candidate sentences)")
    for i, s in enumerate(pick, 1):
        print(f"  {i}. {s}")
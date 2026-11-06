"""
robustness_detail.py

Per-concept directions under set-ups B, C and D (same 10 seeded models),
to identify which concepts change between set-ups. Takes several minutes.
Run: python ~/Desktop/robustness_detail.py
"""

import os
import re
import csv
import unicodedata
from collections import defaultdict
import numpy as np
import jieba
from gensim.models import Word2Vec

jieba.add_word("吸毒人员")
DESK = os.path.expanduser("~/Desktop")
OUT = os.path.join(DESK, "results_v1")
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(1, 11))
THR = 9

# ---------------------------------------------------------------- data
us_text = open(os.path.join(DESK, "usa_corpus_clean.txt"), encoding="utf-8").read().lower()
us_text = re.sub(r"[^a-z\s]", " ", us_text)
US_SENTS = [s.split() for s in us_text.split("\n") if len(s.strip()) > 10]


def noise(w):
    if not w.strip():
        return True
    if all(unicodedata.category(c)[0] in ("P", "S", "Z") for c in w):
        return True
    return re.fullmatch(r"[0-9.,%·]+", w) is not None


CN_SENTS = []
for s in open(os.path.join(DESK, "china_corpus_clean_v3.txt"), encoding="utf-8").read().split("\n"):
    s = s.strip()
    if len(s) < 5:
        continue
    ws = [w for w in jieba.cut(s) if not noise(w)]
    if len(ws) > 3:
        CN_SENTS.append(ws)

# ------------------------------------------------------- concepts/anchors
T = {
    "US": {"Addiction": "addiction", "Abuse": "abuse", "Overdose": "overdose",
           "Prevention": "prevention", "Opioid": "opioid",
           "Rehabilitation": "rehabilitation", "Drug user": "users",
           "Prescription": "prescription", "Cocaine": "cocaine", "Heroin": "heroin",
           "Trafficking": "trafficking", "Drug (general)": "drug", "Marijuana": "marijuana"},
    "CN": {"Addiction": "成瘾", "Abuse": "滥用", "Prevention": "预防", "Opioid": "阿片类",
           "Rehabilitation": "康复", "Drug user": "吸毒人员", "Prescription": "处方",
           "Cocaine": "可卡因", "Heroin": "海洛因", "Trafficking": "贩毒",
           "Drug (general)": "毒品", "Marijuana": "大麻"},
}
FULL = {
    "US": (["treatment", "patient", "disorder", "recovery", "therapy", "rehabilitation",
            "health", "care"],
           ["punishment", "offense", "offender", "criminal", "arrest", "sanction",
            "enforcement"]),
    "CN": (["治疗", "患者", "康复", "戒毒", "医疗", "健康", "药物", "成瘾"],
           ["处罚", "犯罪", "违法", "惩治", "刑罚", "逮捕", "罚款"]),
}


def setup_anchors(c, k):
    med, pun = FULL[c]
    if k == "B":
        return med, pun
    t = set(T[c].values())
    med = [w for w in med if w not in t]
    pun = [w for w in pun if w not in t]
    if k == "D" and c == "CN":
        med = [w for w in med if w != "戒毒"]
    return med, pun


def unit(v):
    return v / np.linalg.norm(v)


def dimension(wv, med, pun):
    return unit(np.mean([unit(wv[w]) for w in med], 0) - np.mean([unit(wv[w]) for w in pun], 0))


def concept_scores(wv, c, d):
    return {k: float(np.dot(unit(wv[t]), d)) for k, t in T[c].items() if t in wv}


# ---------------------------------------------------------------- train
print("Training 10 models per country ...")
WV = {"US": [], "CN": []}
for s in SEEDS:
    for c, sents in (("US", US_SENTS), ("CN", CN_SENTS)):
        WV[c].append(Word2Vec(sentences=sents, vector_size=100, window=5, min_count=5,
                              workers=1, epochs=10, sg=1, seed=s).wv)
    print(f"   seed {s} done")


def run_setup(med_pun):
    """med_pun: {country: (med, pun)} -> {country: [scores per seed]}"""
    return {c: [concept_scores(wv, c, dimension(wv, *med_pun[c])) for wv in WV[c]] for c in WV}


SHARED = [k for k in T["CN"] if k in T["US"]]


def direction(res, k, thr):
    lower = sum(1 for u, n in zip(res["US"], res["CN"]) if n[k] < u[k])
    if lower >= thr:
        return "CN lower", lower
    if len(SEEDS) - lower >= thr:
        return "CN higher", lower
    return "undetermined", lower


def h_summary(res, thr):
    d = [direction(res, k, thr)[0] for k in SHARED]
    rehab_pos = sum(1 for r in res["CN"] if r["Rehabilitation"] > 0)
    return d.count("CN lower"), d.count("CN higher"), d.count("undetermined"), rehab_pos


def sign(vals):
    pos = sum(v > 0 for v in vals)
    if pos >= THR:
        return "positive"
    if len(vals) - pos >= THR:
        return "negative"
    return "near zero"


RES = {k: run_setup({c: setup_anchors(c, k) for c in WV}) for k in "BCD"}

print("=" * 90)
print("Per-concept direction (number of runs with CN lower) by set-up")
print(f"   {'Concept':<16}" + "".join(f"{k + ' runs':>10}{k + ' (9)':>16}" for k in "BCD"))
rows = []
for k in SHARED:
    line, row = f"   {k:<16}", [k]
    for st in "BCD":
        d, low = direction(RES[st], k, THR)
        line += f"{low:>10}{d:>16}"
        row += [low, d]
    print(line)
    rows.append(row)
with open(os.path.join(OUT, "robustness_by_concept.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["Concept"] + [x for st in "BCD" for x in (st + "_CN_lower_runs", st + "_direction")])
    w.writerows(rows)
print(f"\nSaved: {os.path.join(OUT, 'robustness_by_concept.csv')}")
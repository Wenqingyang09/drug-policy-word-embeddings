"""
calibration_check.py

Checks whether H1 could be an artefact of the two dimensions being shifted or
scaled differently (for example because legal texts lean punitive overall).

For each of the 10 seeded models per country (set-up C) it computes:
  1. the mean and SD of the scores of all vocabulary words (frequency >= 20,
     anchors excluded): the baseline of each axis
  2. each concept's percentile rank within its own corpus vocabulary, which
     removes any overall shift or difference in scale between the two axes
  3. H1 on percentile ranks with the same rule (>= 9 of 10 runs)
  4. the share of all 100 US x CN model pairs in which China is lower,
     for raw scores and for percentile ranks (the main rule pairs runs by seed
     number, which is arbitrary)

Takes several minutes.  Run: python ~/Desktop/calibration_check.py
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

# Data
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

# Concepts/anchors
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


# Train
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
main_raw = RES["C"]

print("=" * 90)
med_pun = {c: setup_anchors(c, "C") for c in WV}
pct = {"US": [], "CN": []}
for c in WV:
    excl = set(FULL[c][0]) | set(FULL[c][1])
    vocab = [w for w in WV[c][0].index_to_key
             if WV[c][0].get_vecattr(w, "count") >= 20 and w not in excl]
    base_means, base_sds = [], []
    for wv in WV[c]:
        d = dimension(wv, *med_pun[c])
        allsc = np.array([float(np.dot(unit(wv[w]), d)) for w in vocab])
        base_means.append(allsc.mean())
        base_sds.append(allsc.std())
        sc = concept_scores(wv, c, d)
        pct[c].append({k: float((allsc < v).mean() * 100) for k, v in sc.items()})
    print(f"1. {c} axis baseline over {len(vocab)} words: mean {np.mean(base_means):.3f}, "
          f"SD {np.mean(base_sds):.3f}")

print("=" * 90)
print("2-4. Percentile ranks within own corpus and H1 checks")
print(f"   {'Concept':<16}{'US pct':>8}{'CN pct':>8}{'pct: CN lower runs':>20}{'pct dir (9)':>14}"
      f"{'raw 100-pair':>14}{'pct 100-pair':>14}")
rows, dirs = [], []
for k in T["US"]:
    usp = [r[k] for r in pct["US"]]
    if k not in T["CN"]:
        print(f"   {k:<16}{np.mean(usp):>8.1f}{'absent':>8}")
        rows.append([k, round(np.mean(usp), 1), "", "", "", "", ""])
        continue
    cnp = [r[k] for r in pct["CN"]]
    low = sum(1 for a, b in zip(usp, cnp) if b < a)
    d = "CN lower" if low >= THR else ("CN higher" if 10 - low >= THR else "undetermined")
    dirs.append(d)
    raw100 = np.mean([n[k] < u[k] for u in main_raw["US"] for n in main_raw["CN"]])
    pct100 = np.mean([b < a for a in usp for b in cnp])
    print(f"   {k:<16}{np.mean(usp):>8.1f}{np.mean(cnp):>8.1f}{low:>17}/10{d:>14}"
          f"{raw100:>14.2f}{pct100:>14.2f}")
    rows.append([k, round(np.mean(usp), 1), round(np.mean(cnp), 1), low, d,
                 round(raw100, 2), round(pct100, 2)])
print(f"\n   H1 on percentile ranks: CN lower {dirs.count('CN lower')}, "
      f"CN higher {dirs.count('CN higher')}, undetermined {dirs.count('undetermined')}")
with open(os.path.join(OUT, "calibration_check.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["Concept", "US percentile", "CN percentile", "pct CN lower runs",
                "pct direction", "raw share of 100 pairs CN lower", "pct share of 100 pairs CN lower"])
    w.writerows(rows)
print(f"Saved: {os.path.join(OUT, 'calibration_check.csv')}")
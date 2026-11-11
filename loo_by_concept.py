"""
loo_by_concept.py

Leave-one-out check by concept (main analysis: original US line units,
set-up C, nine-run rule). This script records the direction of every shared
concept for each dropped anchor.

The 10 US and 10 Chinese models are retrained with exactly the same data and
settings as results_summary.py. Nothing in results_v1 is changed.

Output: printed and saved to ~/Desktop/results_v2_check/loo_by_concept.txt
and loo_by_concept.csv. Takes about 10 to 15 minutes.

Run:
  conda activate thesis
  python ~/Desktop/loo_by_concept.py
"""

import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import re
import csv
import unicodedata
from collections import Counter
import numpy as np
import jieba
from gensim.models import Word2Vec

jieba.add_word("吸毒人员")
DESK = os.path.expanduser("~/Desktop")
OUT = os.path.join(DESK, "results_v2_check")
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(1, 11))
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# Data, as in results_summary.py
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
SHARED = [k for k in T["CN"] if k in T["US"]]


def anchors_c(c):
    t = set(T[c].values())
    return [w for w in FULL[c][0] if w not in t], [w for w in FULL[c][1] if w not in t]


def unit(v):
    return v / np.linalg.norm(v)


def dimension(wv, med, pun):
    return unit(np.mean([unit(wv[w]) for w in med], 0) - np.mean([unit(wv[w]) for w in pun], 0))


def scores(wvs, c, med, pun):
    return [{k: float(np.dot(unit(wv[t]), dimension(wv, med, pun))) for k, t in T[c].items()
             if t in wv} for wv in wvs]


def direction(us, cn, k):
    lower = sum(1 for u, n in zip(us, cn) if n[k] < u[k])
    if lower >= 9:
        return "CN lower"
    if len(SEEDS) - lower >= 9:
        return "CN higher"
    return "undetermined"


say("Training 10 models per country ...")
WV = {"US": [], "CN": []}
for s in SEEDS:
    for c, sents in (("US", US_SENTS), ("CN", CN_SENTS)):
        WV[c].append(Word2Vec(sentences=sents, vector_size=100, window=5, min_count=5,
                              workers=1, epochs=10, sg=1, seed=s).wv)
say("   done")

base = {c: scores(WV[c], c, *anchors_c(c)) for c in WV}
BASE_DIR = {k: direction(base["US"], base["CN"], k) for k in SHARED}

rows = []
totals = []
for c in WV:
    med, pun = anchors_c(c)
    for pole, lst in (("medical", med), ("punitive", pun)):
        for a in lst:
            m2 = [x for x in med if x != a] if pole == "medical" else med
            p2 = [x for x in pun if x != a] if pole == "punitive" else pun
            res = dict(base)
            res[c] = scores(WV[c], c, m2, p2)
            d = {k: direction(res["US"], res["CN"], k) for k in SHARED}
            cnt = Counter(d.values())
            totals.append((cnt["CN lower"], cnt["CN higher"], cnt["undetermined"]))
            rows.append([c, pole, a] + [d[k] for k in SHARED])

say("=" * 90)
say("Main analysis (no anchor dropped): " +
    ", ".join(f"{k}: {v}" for k, v in BASE_DIR.items()))
say(f"H1 totals over the {len(rows)} leave-one-out runs: "
    f"CN lower {min(t[0] for t in totals)}-{max(t[0] for t in totals)}, "
    f"CN higher {min(t[1] for t in totals)}-{max(t[1] for t in totals)}, "
    f"undetermined {min(t[2] for t in totals)}-{max(t[2] for t in totals)}")
say("=" * 90)
say("Per concept: in how many of the leave-one-out runs each direction occurs,")
say("and which dropped anchors change the direction of the main analysis")
for i, k in enumerate(SHARED):
    col = [r[3 + i] for r in rows]
    cnt = Counter(col)
    changed = [f"{r[0]} {r[2]} -> {r[3 + i]}" for r in rows if r[3 + i] != BASE_DIR[k]]
    say(f"   {k:<15} main: {BASE_DIR[k]:<13} lower {cnt['CN lower']:>2}  higher "
        f"{cnt['CN higher']:>2}  undetermined {cnt['undetermined']:>2}   "
        f"changes: {'; '.join(changed) if changed else 'none'}")

with open(os.path.join(OUT, "loo_by_concept.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["country", "pole", "dropped anchor"] + SHARED)
    w.writerows(rows)
with open(os.path.join(OUT, "loo_by_concept.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
print(f"\nSaved in {OUT}")
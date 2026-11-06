"""
results_summary.py

Produces the main results, the set-up, threshold and leave-one-out checks, the
pole words and the nearest neighbours from the same 10 seeded models per country.

Rules:
  - direction CN vs US: determined if it holds in >= 9 of 10 runs
  - sign within a country: positive/negative if it holds in >= 9 of 10 runs
  - H1: most determined shared concepts lower in CN
  - H2: CN Rehabilitation > 0 in >= 9 of 10 runs
  - validation: 15 highest / lowest words, frequency >= 20, anchors excluded
  - neighbours: kept if in the top 15 in >= 8 of 10 runs

Output files are written to ~/Desktop/results_v1/. Takes several minutes.

Run:
  conda activate thesis
  python ~/Desktop/results_summary.py
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
main = RES["C"]

# ---------------------------------------------------------- 1. main table
print("=" * 100)
print("1. Main results, set-up C, mean (sd) over 10 runs")
hdr = ["Concept", "US mean", "US sd", "US sign", "CN mean", "CN sd", "CN sign",
       "Diff mean", "Diff min", "Diff max", "CN lower runs", "Direction"]
rows = []
for k in T["US"]:
    us = [r[k] for r in main["US"]]
    row = [k, np.mean(us), np.std(us), sign(us)]
    if k in T["CN"]:
        cn = [r[k] for r in main["CN"]]
        diff = [n - u for n, u in zip(cn, us)]
        dirn, low = direction(main, k, THR)
        row += [np.mean(cn), np.std(cn), sign(cn), np.mean(diff), min(diff), max(diff),
                f"{low}/10", dirn]
    else:
        row += ["", "", "absent", "", "", "", "", "US only"]
    rows.append(row)
    f = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
    print("   " + " | ".join(f(x) for x in row))
with open(os.path.join(OUT, "main_results.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(hdr)
    w.writerows([[round(x, 4) if isinstance(x, float) else x for x in r] for r in rows])

# ------------------------------------------------------- 2. robustness
print("=" * 100)
print("2. H1 and H2 by set-up and threshold (CN lower / CN higher / undetermined; CN rehab > 0 runs)")
rob = []
for k in "BCD":
    for thr in (8, 9, 10):
        lo, hi, un, rp = h_summary(RES[k], thr)
        rob.append([k, thr, lo, hi, un, rp])
        print(f"   set-up {k}, threshold {thr}: {lo} / {hi} / {un}; rehab > 0 in {rp}/10; "
              f"rehab mean {np.mean([r['Rehabilitation'] for r in RES[k]['CN']]):.3f}")

print("   Leave-one-out (set-up C, threshold 9)")
loo = []
for c in WV:
    med, pun = setup_anchors(c, "C")
    for pole, lst in (("medical", med), ("punitive", pun)):
        for a in lst:
            mp = {cc: setup_anchors(cc, "C") for cc in WV}
            m2 = [x for x in med if x != a] if pole == "medical" else med
            p2 = [x for x in pun if x != a] if pole == "punitive" else pun
            mp[c] = (m2, p2)
            res = run_setup(mp)
            lo, hi, un, rp = h_summary(res, THR)
            rm = np.mean([r["Rehabilitation"] for r in res["CN"]])
            loo.append([c, pole, a, lo, hi, un, rp, round(rm, 4)])
            print(f"     {c} drop {a:<14} {lo}/{hi}/{un}  rehab>0 {rp}/10  rehab mean {rm:.3f}")
with open(os.path.join(OUT, "robustness.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["setup", "threshold", "CN lower", "CN higher", "undetermined", "rehab>0 runs"])
    w.writerows(rob)
    w.writerow([])
    w.writerow(["LOO country", "pole", "dropped", "CN lower", "CN higher", "undetermined",
                "rehab>0 runs", "rehab mean"])
    w.writerows(loo)

# ------------------------------------------------------- 3. validation
print("=" * 100)
print("3. Dimension validation: 15 highest and lowest words (freq >= 20, anchors excluded)")
poles_rows = []
for c in WV:
    med, pun = setup_anchors(c, "C")
    excl = set(FULL[c][0]) | set(FULL[c][1])
    vocab = [w for w in WV[c][0].index_to_key
             if WV[c][0].get_vecattr(w, "count") >= 20 and w not in excl]
    acc = defaultdict(list)
    for wv in WV[c]:
        d = dimension(wv, med, pun)
        for w in vocab:
            acc[w].append(float(np.dot(unit(wv[w]), d)))
    mean = sorted(((np.mean(v), w) for w, v in acc.items()), reverse=True)
    top, bottom = mean[:15], mean[-15:][::-1]
    print(f"   {c} medical end : " + ", ".join(f"{w} ({s:.2f})" for s, w in top))
    print(f"   {c} punitive end: " + ", ".join(f"{w} ({s:.2f})" for s, w in bottom))
    poles_rows += [[c, "medical", w, round(s, 4)] for s, w in top]
    poles_rows += [[c, "punitive", w, round(s, 4)] for s, w in bottom]
with open(os.path.join(OUT, "poles.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["country", "end", "word", "mean score"])
    w.writerows(poles_rows)

# ------------------------------------------------------- 4. neighbours
print("=" * 100)
print("4. Stable nearest neighbours (top 15 in >= 8 of 10 runs)")
nb_rows = []
for c, word in (("US", "rehabilitation"), ("CN", "康复"), ("US", "users"), ("CN", "吸毒人员")):
    cnt, sims = defaultdict(int), defaultdict(list)
    for wv in WV[c]:
        for x, s in wv.most_similar(word, topn=15):
            cnt[x] += 1
            sims[x].append(s)
    stable = sorted(((np.mean(sims[x]), x, cnt[x]) for x in cnt if cnt[x] >= 8), reverse=True)
    print(f"   {c} {word}: " + ", ".join(f"{x} ({s:.2f}, {n}/10)" for s, x, n in stable))
    nb_rows += [[c, word, x, n, round(s, 4)] for s, x, n in stable]
with open(os.path.join(OUT, "neighbours.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["country", "word", "neighbour", "runs in top 15", "mean similarity"])
    w.writerows(nb_rows)

# per-run scores for plotting
with open(os.path.join(OUT, "per_run_scores.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["country", "seed", "concept", "score"])
    for c in WV:
        for s, r in zip(SEEDS, main[c]):
            for k, v in r.items():
                w.writerow([c, s, k, round(v, 5)])
print(f"\nAll files saved in {OUT}")
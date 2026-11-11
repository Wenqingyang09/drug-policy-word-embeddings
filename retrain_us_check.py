"""
retrain_us_check.py

Robustness checks:
  (1) training units: the US corpus was split into training units at the line
      breaks left by OCR. Here the lines are joined and the text is split into
      sentences instead.
  (2) institutional names: names of federal agencies and programmes that
      contain a target word or an anchor word (e.g. Substance Abuse and Mental
      Health Services Administration, Centers for Disease Control and
      Prevention) are replaced by a neutral placeholder token.

Four US variants are trained, 10 runs each, with exactly the same settings as
results_summary.py (skip-gram, 100 dimensions, window 5, min_count 5, 10 epochs,
workers 1, seeds 1-10):
  line          original line units (baseline, for comparison)
  sentence      sentence units
  line_noname   line units, institutional names replaced
  sent_noname   sentence units, institutional names replaced
The Chinese model is trained once more in the same run (same settings and
preprocessing as results_summary.py), so that all models share the same
Python hash seed. The Chinese corpus is not changed.

Nothing in results_v1 is overwritten. Output goes to
~/Desktop/results_v2_check/:
  summary.txt          everything printed on screen
  concepts_by_variant.csv  per concept and set-up: means, sides, difference, direction
  h1_h2_by_variant.csv     H1 counts and H2 runs for set-ups B, C, D and
                           thresholds 8, 9, 10
  name_replacements.csv    how often each institutional name was replaced
  sentence_samples.txt     30 random sentence units, to check the segmentation
  percentile_by_variant.csv  percentile ranks and H1 on ranks (set-up C)

Takes about 20 to 40 minutes (50 models, single worker for reproducibility).

Run:
  conda activate thesis
  python ~/Desktop/retrain_us_check.py
"""

import os
import sys

# Fix Python's hash seed so that gensim training is reproducible across runs.
# The script restarts itself once with PYTHONHASHSEED=0 if it is not set.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import re
import csv
import random
import unicodedata
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


# US text
RAW = open(os.path.join(DESK, "usa_corpus_clean.txt"), encoding="utf-8").read()


def to_tokens(unit_text):
    t = re.sub(r"[^a-z\s]", " ", unit_text.lower())
    return t.split()


def line_units(text):
    """Exactly as in results_summary.py."""
    t = re.sub(r"[^a-z\s]", " ", text.lower())
    return [s.split() for s in t.split("\n") if len(s.strip()) > 10]


ABBR = r"\b(Dr|Mr|Mrs|Ms|No|Nos|vs|etc|Inc|Jr|Sr|St|Fig|Vol|pp|Sec|Dept|Gov|approx)\."


def sentence_units(text):
    t = re.sub(r"-[ \t]*\n[ \t]*(?=[a-z])", "", text)          # hyphen at line end
    t = re.sub(r"\b((?:[A-Za-z]\.){2,})",                        # U.S., e.g., i.e.
               lambda m: m.group(1).replace(".", ""), t)
    t = re.sub(ABBR, r"\1", t, flags=re.I)                        # Dr. No. etc.
    out = []
    for block in re.split(r"\n\s*\n", t):                         # blank line = hard break
        block = re.sub(r"\s*\n\s*", " ", block)                   # join lines
        for s in re.split(r"(?<=[.!?])[\"'”’)]*\s+", block):
            if len(s.strip()) > 10:
                out.append(s)
    return [to_tokens(s) for s in out if to_tokens(s)]


# Proper names of agencies, programmes, plans, surveys, data systems, laws and
# journals that contain a target word or an anchor word. Built from the list of
# capitalised phrases produced by check_names_units.py (frequency >= 5), plus a
# few spelling variants. Policy goals and chapter titles written by the
# strategies themselves (e.g. "Break the Cycle of Drug Use") are NOT names and
# are kept, because they are part of the framing being measured. Generic terms
# (e.g. "criminal justice system", "prescription drug monitoring programs") are
# also kept.
NAMES = [
    # federal agencies and offices
    "substance abuse and mental health services administration",
    "center for substance abuse prevention", "center for substance abuse treatment",
    "center for behavioral health statistics and quality",
    "national institute on drug abuse", "national institute on alcohol abuse and alcoholism",
    "national institutes of health", "national center for health statistics",
    "centers for disease control and prevention", "center for disease control and prevention",
    "national center for injury prevention and control",
    "department of health and human services", "health and human services",
    "health resources and services administration", "indian health service",
    "veterans health administration", "food and drug administration",
    "office of national drug control policy", "director of national drug control policy",
    "drug enforcement administration", "immigration and customs enforcement",
    "office of juvenile justice and delinquency prevention",
    "national coordinator for health information technology",
    "national drug intelligence center", "financial crimes enforcement network",
    "bureau of international narcotics and law enforcement affairs",
    "national prevention council",
    "california department of corrections and rehabilitation",
    # programmes, task forces, plans and documents
    "national drug control strategy", "national drug control program",
    "national drug control budget", "national drug threat assessment",
    "national heroin threat assessment", "interagency assessment of cocaine movement",
    "high intensity drug trafficking areas", "high intensity drug trafficking area",
    "organized crime drug enforcement task forces", "organized crime drug enforcement task force",
    "border enforcement security task forces", "border enforcement security task force",
    "domestic highway enforcement", "advanced roadside impaired driving enforcement",
    "substance abuse prevention and treatment block grant",
    "prescription drug abuse prevention plan", "prescription drug abuse action plan",
    "national treatment plan", "treatment alternatives for safe communities",
    "access to recovery", "drug free communities", "drug market intervention",
    "brief intervention and referral to treatment", "north american drug dialogue",
    "national impaired driving prevention month", "national substance abuse prevention month",
    "inter american drug abuse control commission", "national drug court institute",
    "national association of drug court professionals",
    "national center on addiction and substance abuse",
    # surveys and data systems
    "national survey on drug use and health", "drug abuse warning network",
    "arrestee drug abuse monitoring", "treatment episode data set",
    "national survey of substance abuse treatment services",
    "drug and alcohol services information system",
    "system to retrieve information on drug evidence", "federal wide drug seizure system",
    "community drug early warning system", "national drug early warning system",
    "national roadside survey of alcohol and drug use", "national hospital care survey",
    # laws
    "affordable care act", "mental health parity and addiction equity act",
    "secure and responsible drug disposal act",
    # international bodies and journals cited in the documents
    "world health organization", "american journal of public health",
    "drug and alcohol dependence", "drug alcohol depend",
    "journal of substance abuse treatment", "subst abuse treat",
    "international journal of drug policy", "journal of drug education",
]
CONNECTORS = {"of", "and", "for", "on", "the", "to", "in"}


def name_pattern(n):
    """Content words must start with a capital letter or be in capitals, so that
    generic lower-case phrases (e.g. 'access to recovery services') are not
    replaced. Connectors may be in any case. Words may be separated by spaces,
    hyphens or a line break."""
    parts = []
    for w in n.split():
        if w in CONNECTORS:
            parts.append(f"(?:{w}|{w.capitalize()}|{w.upper()})")
        else:
            parts.append(f"(?:{w.capitalize()}|{w.upper()})")
    return r"\b" + r"[\s\-]+".join(parts) + r"\b"


def replace_names(text):
    """Replace each name by the token 'orgname'. Longer names are replaced first."""
    counts = {}
    for n in sorted(NAMES, key=len, reverse=True):
        text, k = re.subn(name_pattern(n), " orgname ", text)
        counts[n] = k
    return text, counts


RAW_NONAME, NAME_COUNTS = replace_names(RAW)

US_VARIANTS = {
    "line": line_units(RAW),
    "sentence": sentence_units(RAW),
    "line_noname": line_units(RAW_NONAME),
    "sent_noname": sentence_units(RAW_NONAME),
}

say("=" * 90)
say("0. Training units")
for v, sents in US_VARIANTS.items():
    lens = [len(s) for s in sents]
    say(f"   {v:<12} units {len(sents):>7}   tokens {sum(lens):>8}   "
        f"mean length {np.mean(lens):5.1f}   median {np.median(lens):4.0f}   max {max(lens)}   "
        f"units > 100 tokens {sum(l > 100 for l in lens)}")
end_punct = sum(1 for l in RAW.split("\n") if l.strip()[-1:] in ".!?")
n_lines = sum(1 for l in RAW.split("\n") if l.strip())
say(f"   original non-empty lines ending in . ! ? : {end_punct}/{n_lines} "
    f"({100 * end_punct / n_lines:.0f}%)")
say("   institutional names replaced:")
for n, k in NAME_COUNTS.items():
    say(f"     {k:>5}  {n}")
for w in ("abuse", "prevention", "rehabilitation", "drug", "trafficking"):
    a = sum(s.count(w) for s in US_VARIANTS["line"])
    b = sum(s.count(w) for s in US_VARIANTS["line_noname"])
    say(f"   frequency of '{w}': {a} -> {b} after replacing names")

random.seed(1)
with open(os.path.join(OUT, "sentence_samples.txt"), "w", encoding="utf-8") as fh:
    for s in random.sample(US_VARIANTS["sentence"], 30):
        fh.write(" ".join(s) + "\n\n")
with open(os.path.join(OUT, "name_replacements.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["name", "replacements"])
    w.writerows(NAME_COUNTS.items())

# CN text
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

# Concepts and anchors
# identical to results_summary.py
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


def train(sents, seed):
    return Word2Vec(sentences=sents, vector_size=100, window=5, min_count=5,
                    workers=1, epochs=10, sg=1, seed=seed).wv


# Train
say("=" * 90)
say("Training 10 Chinese models ...")
CN_WV = []
for s in SEEDS:
    CN_WV.append(train(CN_SENTS, s))
say("   done")

US_WV = {}
for v, sents in US_VARIANTS.items():
    say(f"Training 10 US models, variant '{v}' ...")
    US_WV[v] = [train(sents, s) for s in SEEDS]
    missing = [t for t in set(T["US"].values()) | set(FULL["US"][0]) | set(FULL["US"][1])
               if t not in US_WV[v][0]]
    if missing:
        say(f"   WARNING, not in vocabulary: {missing}")
    say("   done")


def scores(wvs, c, setup):
    out = []
    for wv in wvs:
        med, pun = [w for w in setup_anchors(c, setup)[0] if w in wv], \
                   [w for w in setup_anchors(c, setup)[1] if w in wv]
        out.append(concept_scores(wv, c, dimension(wv, med, pun)))
    return out


def direction(us, cn, k, thr):
    lower = sum(1 for u, n in zip(us, cn) if n.get(k, np.nan) < u.get(k, np.nan))
    if lower >= thr:
        return "CN lower", lower
    if len(SEEDS) - lower >= thr:
        return "CN higher", lower
    return "undetermined", lower


CN_RES = {k: scores(CN_WV, "CN", k) for k in "BCD"}

# Results
def sign(vals):
    pos = sum(v > 0 for v in vals)
    if pos >= 9:
        return "M"
    if len(vals) - pos >= 9:
        return "P"
    return "0"


say("=" * 90)
say("1. H1 (CN lower / CN higher / undetermined) and H2 (runs with CN rehabilitation > 0)")
h_rows, c_rows = [], []
US_RES = {}
for v in US_VARIANTS:
    for k in "BCD":
        us = scores(US_WV[v], "US", k)
        US_RES[(v, k)] = us
        cn = CN_RES[k]
        for thr in (8, 9, 10):
            d = [direction(us, cn, c, thr)[0] for c in SHARED]
            rp = sum(1 for r in cn if r["Rehabilitation"] > 0)
            h_rows.append([v, k, thr, d.count("CN lower"), d.count("CN higher"),
                           d.count("undetermined"), rp])
            if thr == 9:
                say(f"   {v:<12} set-up {k}: {d.count('CN lower')} / {d.count('CN higher')} / "
                    f"{d.count('undetermined')}    H2 {rp}/10   "
                    f"CN rehab mean {np.mean([r['Rehabilitation'] for r in cn]):.3f}")
        for c in T["US"]:
            uv = [r[c] for r in us]
            if c in T["CN"]:
                cv = [r[c] for r in cn]
                dirn, low = direction(us, cn, c, 9)
                c_rows.append([v, k, c, round(np.mean(uv), 4), sign(uv), round(np.mean(cv), 4),
                               sign(cv), round(np.mean(cv) - np.mean(uv), 4), f"{low}/10", dirn])
            else:
                c_rows.append([v, k, c, round(np.mean(uv), 4), sign(uv), "", "", "", "",
                               "US only"])

say("=" * 90)
say("2. Per concept: US mean (side) / CN mean (side) / difference / runs CN lower / direction")
say("   side: M = positive in >= 9 runs, P = negative in >= 9 runs, 0 = neither")
for k in "BCD":
    say(f"   ---- set-up {k}")
    for c in T["US"]:
        say(f"   {c}")
        for r in c_rows:
            if r[1] == k and r[2] == c:
                say(f"      {r[0]:<12} US {r[3]:>7} ({r[4]})   CN {str(r[5]):>7} ({r[6]})   "
                    f"diff {str(r[7]):>7}   {r[8]:>5}   {r[9]}")

# check that the 'line' baseline reproduces results_v1 (set-up C)
V1 = os.path.join(DESK, "results_v1", "per_run_scores.csv")
if os.path.exists(V1):
    v1 = {}
    for r in csv.DictReader(open(V1, encoding="utf-8-sig")):
        v1[(r["country"], int(r["seed"]), r["concept"])] = float(r["score"])
    diffs = []
    for c, res in (("US", US_RES[("line", "C")]), ("CN", CN_RES["C"])):
        for s, r in zip(SEEDS, res):
            for k, v in r.items():
                if (c, s, k) in v1:
                    diffs.append(abs(v - v1[(c, s, k)]))
    say("=" * 90)
    say(f"Check against results_v1: largest absolute difference in per-run scores "
        f"= {max(diffs):.5f} over {len(diffs)} scores (0.00001 or less = exact reproduction)")

say("=" * 90)
say("3. Direction changes relative to the 'line' baseline (threshold 9)")
for k in "BCD":
    base = {r[2]: r[9] for r in c_rows if r[0] == "line" and r[1] == k}
    for v in ("sentence", "line_noname", "sent_noname"):
        ch = [(r[2], base[r[2]], r[9]) for r in c_rows
              if r[0] == v and r[1] == k and r[9] != base[r[2]]]
        say(f"   set-up {k}  {v:<12} " + ("no change" if not ch else
            "; ".join(f"{c}: {a} -> {b}" for c, a, b in ch)))

say("=" * 90)
say("4. Concepts with the same direction in all 4 variants x 3 set-ups (12 combinations)")
for target in ("CN lower", "CN higher"):
    ok = [c for c in SHARED
          if all(r[9] == target for r in c_rows if r[2] == c)]
    say(f"   {target} in all 12: {', '.join(ok) if ok else 'none'}")


say("=" * 90)
say("5. Percentile check (set-up C): each concept's percentile rank among all words with")
say("   frequency >= 20 in its own model, anchors excluded; H1 re-assessed on ranks")
from collections import defaultdict


def pct_scores(wvs, c, setup):
    med, pun = setup_anchors(c, setup)
    excl = set(FULL[c][0]) | set(FULL[c][1])
    out = []
    for wv in wvs:
        d = dimension(wv, [w for w in med if w in wv], [w for w in pun if w in wv])
        vocab = [w for w in wv.index_to_key if wv.get_vecattr(w, "count") >= 20 and w not in excl]
        allsc = np.array([float(np.dot(unit(wv[w]), d)) for w in vocab])
        r = {}
        for k, t in T[c].items():
            if t in wv:
                r[k] = 100 * float(np.mean(allsc < float(np.dot(unit(wv[t]), d))))
        out.append(r)
    return out


CN_PCT = pct_scores(CN_WV, "CN", "C")
say(f"   CN rehabilitation percentile, mean over runs: "
    f"{np.mean([r['Rehabilitation'] for r in CN_PCT]):.1f}")
p_rows = []
for v in US_VARIANTS:
    up = pct_scores(US_WV[v], "US", "C")
    d = [direction(up, CN_PCT, c, 9) for c in SHARED]
    lo = [c for c, (x, _) in zip(SHARED, d) if x == "CN lower"]
    hi = [c for c, (x, _) in zip(SHARED, d) if x == "CN higher"]
    un = [c for c, (x, _) in zip(SHARED, d) if x == "undetermined"]
    say(f"   {v:<12} {len(lo)} / {len(hi)} / {len(un)}   higher: {', '.join(hi) or '-'}   "
        f"undetermined: {', '.join(un) or '-'}   "
        f"US rehabilitation percentile {np.mean([r['Rehabilitation'] for r in up]):.1f}")
    for c, (x, low) in zip(SHARED, d):
        p_rows.append([v, c, round(np.mean([r[c] for r in up]), 1),
                       round(np.mean([r[c] for r in CN_PCT]), 1), f"{low}/10", x])
with open(os.path.join(OUT, "percentile_by_variant.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["variant", "concept", "US percentile", "CN percentile", "runs CN lower", "direction"])
    w.writerows(p_rows)

say("=" * 90)
say("6. Dimension validity by variant (set-up C): 10 highest and 10 lowest words,")
say("   frequency >= 20, anchors excluded, mean over 10 runs")
for v in US_VARIANTS:
    med, pun = setup_anchors("US", "C")
    excl = set(FULL["US"][0]) | set(FULL["US"][1])
    acc = defaultdict(list)
    for wv in US_WV[v]:
        d = dimension(wv, [w for w in med if w in wv], [w for w in pun if w in wv])
        for w in wv.index_to_key:
            if wv.get_vecattr(w, "count") >= 20 and w not in excl:
                acc[w].append(float(np.dot(unit(wv[w]), d)))
    mean = sorted(((np.mean(x), w) for w, x in acc.items() if len(x) == len(SEEDS)), reverse=True)
    say(f"   {v:<12} medical : " + ", ".join(w for _, w in mean[:10]))
    say(f"   {'':<12} punitive: " + ", ".join(w for _, w in mean[-10:][::-1]))

say("=" * 90)
say("7. Stable nearest neighbours (top 15 in >= 8 of 10 runs), US, by variant")
for word in ("rehabilitation", "users", "abuse", "prevention"):
    for v in US_VARIANTS:
        cnt = defaultdict(int)
        if word not in US_WV[v][0]:
            say(f"   {word:<15} {v:<12} not in vocabulary")
            continue
        for wv in US_WV[v]:
            for x, _ in wv.most_similar(word, topn=15):
                cnt[x] += 1
        say(f"   {word:<15} {v:<12} " + ", ".join(sorted(x for x in cnt if cnt[x] >= 8)))

say("=" * 90)
say("8. Frequencies of target and anchor words in each US variant (training data)")
words = sorted(set(T["US"].values()) | set(FULL["US"][0]) | set(FULL["US"][1]))
freq = {v: {w: sum(s.count(w) for s in sents) for w in words} for v, sents in US_VARIANTS.items()}
say("   " + f"{'word':<15}" + "".join(f"{v:>13}" for v in US_VARIANTS))
for w in words:
    say("   " + f"{w:<15}" + "".join(f"{freq[v][w]:>13}" for v in US_VARIANTS))

with open(os.path.join(OUT, "h1_h2_by_variant.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["variant", "setup", "threshold", "CN lower", "CN higher", "undetermined",
                "CN rehab > 0 runs"])
    w.writerows(h_rows)
with open(os.path.join(OUT, "concepts_by_variant.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["variant", "setup", "concept", "US mean", "US side", "CN mean", "CN side",
                "diff CN-US", "runs CN lower", "direction"])
    w.writerows(c_rows)
with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
print(f"\nAll files saved in {OUT}")
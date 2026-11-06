"""
collocations.py

Read-only check of the contexts behind the scores, for the Results section.
Nothing is trained and no result file is changed. Uses exactly the same
tokenization as results_summary.py.

For each target word it reports
  1. frequency
  2. the 15 most common words immediately before and immediately after it
     (no words removed)
  3. the 20 most common words within five tokens on either side, with a
     short list of function words removed (the list is printed)
  4. how often a medical anchor and how often a punitive anchor (set-up C)
     occurs within five tokens on either side
  5. three random example lines (fixed seed)
and, for the Chinese corpus, how often 康复 appears in the fixed phrase
社区戒毒、社区康复 and how often 预防 appears next to 惩治.

Output is printed and also saved to ~/Desktop/results_v1/collocations.txt.

Run:
  conda activate thesis
  python ~/Desktop/collocations.py
"""

import os
import re
import random
import unicodedata
from collections import Counter
import jieba

jieba.add_word("吸毒人员")
DESK = os.path.expanduser("~/Desktop")
OUT = os.path.join(DESK, "results_v1", "collocations.txt")
WIN = 5
lines_out = []


def say(s=""):
    print(s)
    lines_out.append(s)


def noise(w):
    if not w.strip():
        return True
    if all(unicodedata.category(c)[0] in ("P", "S", "Z") for c in w):
        return True
    return re.fullmatch(r"[0-9.,%·]+", w) is not None


# ---------------------------------------------------------------- corpora
us_text = open(os.path.join(DESK, "usa_corpus_clean.txt"), encoding="utf-8").read().lower()
us_text = re.sub(r"[^a-z\s]", " ", us_text)
US = [s.split() for s in us_text.split("\n") if len(s.strip()) > 10]

CN_RAW = []
CN = []
for s in open(os.path.join(DESK, "china_corpus_clean_v3.txt"), encoding="utf-8").read().split("\n"):
    s = s.strip()
    if len(s) < 5:
        continue
    ws = [w for w in jieba.cut(s) if not noise(w)]
    if len(ws) > 3:
        CN.append(ws)
        CN_RAW.append(s)

ANCH = {
    "US": (["treatment", "patient", "disorder", "recovery", "therapy", "health", "care"],
           ["punishment", "offense", "offender", "criminal", "arrest", "sanction", "enforcement"]),
    "CN": (["治疗", "患者", "戒毒", "医疗", "健康", "药物"],
           ["处罚", "犯罪", "违法", "惩治", "刑罚", "逮捕", "罚款"]),
}
FUNC = {
    "US": {"the", "of", "and", "to", "in", "for", "a", "an", "on", "with", "by", "as",
           "is", "are", "that", "or", "this", "be", "will", "at", "from", "its", "their"},
    "CN": {"的", "和", "或者", "及", "等", "与", "对", "在", "了", "是", "并", "其",
           "以及", "或", "由", "为", "将", "应当", "可以"},
}
TARGETS = {
    "US": ["rehabilitation", "prevention", "abuse", "addiction"],
    "CN": ["康复", "预防", "滥用", "成瘾"],
}


def contexts(sents, word):
    for si, s in enumerate(sents):
        for i, w in enumerate(s):
            if w == word:
                yield si, s, i


say(f"Function words removed in part 3 only: US {sorted(FUNC['US'])}")
say(f"                                       CN {sorted(FUNC['CN'])}")
for c, sents in (("US", US), ("CN", CN)):
    med, pun = set(ANCH[c][0]), set(ANCH[c][1])
    for t in TARGETS[c]:
        left, right, win = Counter(), Counter(), Counter()
        n = n_med = n_pun = 0
        hits = []
        for si, s, i in contexts(sents, t):
            n += 1
            hits.append(si)
            if i > 0:
                left[s[i - 1]] += 1
            if i < len(s) - 1:
                right[s[i + 1]] += 1
            around = s[max(0, i - WIN):i] + s[i + 1:i + 1 + WIN]
            win.update(w for w in around if w not in FUNC[c])
            n_med += any(w in med for w in around)
            n_pun += any(w in pun for w in around)
        say("=" * 78)
        say(f"{c}  {t}   frequency {n}")
        if n == 0:
            continue
        say("  word before : " + ", ".join(f"{w} {k}" for w, k in left.most_common(15)))
        say("  word after  : " + ", ".join(f"{w} {k}" for w, k in right.most_common(15)))
        say(f"  within ±{WIN}   : " + ", ".join(f"{w} {k}" for w, k in win.most_common(20)))
        say(f"  medical anchor within ±{WIN} : {n_med}/{n} ({100 * n_med / n:.0f}%)")
        say(f"  punitive anchor within ±{WIN}: {n_pun}/{n} ({100 * n_pun / n:.0f}%)")
        random.seed(1)
        for k, si in enumerate(random.sample(sorted(set(hits)), min(3, len(set(hits)))), 1):
            line = CN_RAW[si] if c == "CN" else " ".join(sents[si])
            say(f"  example {k}: {line[:220]}")

# ------------------------------------------------ fixed Chinese phrases
say("=" * 78)
say("Chinese fixed phrases (counted in the segmented text)")
n_k = n_sq = n_phrase = n_jd = 0
for si, s, i in contexts(CN, "康复"):
    n_k += 1
    if i > 0 and s[i - 1] == "社区":
        n_sq += 1
        if i >= 3 and s[i - 3:i - 1] == ["社区", "戒毒"]:
            n_phrase += 1
    if "戒毒" in s[max(0, i - WIN):i] + s[i + 1:i + 1 + WIN]:
        n_jd += 1
say(f"  康复 total {n_k}; preceded by 社区 (社区康复) {n_sq} ({100 * n_sq / n_k:.0f}%); "
    f"in 社区戒毒 社区康复 {n_phrase} ({100 * n_phrase / n_k:.0f}%); "
    f"戒毒 within ±{WIN} {n_jd} ({100 * n_jd / n_k:.0f}%)")
n_y = n_yc = 0
for si, s, i in contexts(CN, "预防"):
    n_y += 1
    if "惩治" in s[max(0, i - WIN):i] + s[i + 1:i + 1 + WIN]:
        n_yc += 1
say(f"  预防 total {n_y}; 惩治 within ±{WIN} {n_yc} ({100 * n_yc / n_y:.0f}%)")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines_out))
print(f"\nSaved: {OUT}")
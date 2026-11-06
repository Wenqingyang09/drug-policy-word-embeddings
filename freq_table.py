"""
freq_table.py

Frequencies of every candidate concept and anchor word in the trained models
(usa_word2vec.model and china_word2vec_v3.model), for Table 2 and the anchor
word description in the Data and Methods section. Words below min_count are counted
directly in the cleaned text so that their exact frequency can be reported.

Run:
  conda activate thesis
  python ~/Desktop/freq_table.py
"""

import os
import re
import unicodedata
import jieba
from gensim.models import Word2Vec

jieba.add_word("吸毒人员")
DESK = os.path.expanduser("~/Desktop")

us_wv = Word2Vec.load(os.path.join(DESK, "usa_word2vec.model")).wv
cn_wv = Word2Vec.load(os.path.join(DESK, "china_word2vec_v3.model")).wv

us_text = open(os.path.join(DESK, "usa_corpus_clean.txt"), encoding="utf-8").read().lower()
us_tokens = re.sub(r"[^a-z\s]", " ", us_text).split()

cn_tokens = []
for line in open(os.path.join(DESK, "china_corpus_clean_v3.txt"), encoding="utf-8"):
    for w in jieba.cut(line.strip()):
        if w.strip() and not all(unicodedata.category(c)[0] in ("P", "S", "Z") for c in w):
            cn_tokens.append(w)


def freq(wv, tokens, w):
    if w in wv:
        return str(wv.get_vecattr(w, "count"))
    return f"{tokens.count(w)} (below min_count)"


concepts = [
    ("Addiction", "addiction", "成瘾"), ("Abuse", "abuse", "滥用"),
    ("Overdose", "overdose", "过量"), ("Prevention", "prevention", "预防"),
    ("Opioid", "opioid", "阿片类"), ("Rehabilitation", "rehabilitation", "康复"),
    ("Drug user", "users", "吸毒人员"), ("Prescription", "prescription", "处方"),
    ("Cocaine", "cocaine", "可卡因"), ("Heroin", "heroin", "海洛因"),
    ("Trafficking", "trafficking", "贩毒"), ("Drug (general)", "drug", "毒品"),
    ("Marijuana", "marijuana", "大麻"),
    ("-- excluded --", "", ""),
    ("Dealer", "dealer", "毒贩"), ("Addict", "addict", "成瘾者"),
    ("Fentanyl", "fentanyl", "芬太尼"), ("Smuggling", "smuggling", "走私"),
    ("Drug use (act)", "", "吸毒"), ("Drug user (old CN term)", "user", "吸毒者"),
]
print("=" * 70)
print(f"{'Concept':<26}{'US term':<16}{'US freq':<22}{'CN term':<10}{'CN freq'}")
for name, us, cn in concepts:
    if name.startswith("--"):
        print(name)
        continue
    uf = freq(us_wv, us_tokens, us) if us else "-"
    cf = freq(cn_wv, cn_tokens, cn) if cn else "-"
    print(f"{name:<26}{us or '-':<16}{uf:<22}{cn or '-':<10}{cf}")

print("=" * 70)
print("Anchor words")
anchors = {
    "US medical": ["treatment", "patient", "disorder", "recovery", "therapy", "health", "care"],
    "US punitive": ["punishment", "offense", "offender", "criminal", "arrest", "sanction", "enforcement"],
    "CN medical": ["治疗", "患者", "戒毒", "医疗", "健康", "药物"],
    "CN punitive": ["处罚", "犯罪", "违法", "惩治", "刑罚", "逮捕", "罚款"],
}
for k, ws in anchors.items():
    wv, tok = (us_wv, us_tokens) if k.startswith("US") else (cn_wv, cn_tokens)
    print(f"  {k}: " + ", ".join(f"{w} {freq(wv, tok, w)}" for w in ws))
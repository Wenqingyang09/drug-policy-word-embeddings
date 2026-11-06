"""
retrain_china_v3.py

Builds a corrected Chinese corpus and trains a new model. Existing files are
NOT overwritten: the new corpus and model are saved with a _v3 suffix.

Corrections, applied to ~/Desktop/china_corpus_clean.txt:
  1. Radical characters left by text extraction (e.g. ⼈ for 人) are converted
     to standard characters (NFKC plus a manual map for the radical supplement).
  2. Website boilerplate left in the white paper (欢迎访问..., 来源：, URLs)
     is removed.
  3. Criminal Law lines on smuggling of non-drug goods are removed: lines that
     contain 走私 but no drug keyword. All removed lines are printed for checking.
  4. Punctuation, symbols and number tokens are dropped after jieba
     segmentation, matching the US preprocessing, which keeps letters only.

Further steps:
  5. 吸毒人员 is added to the jieba dictionary so it is kept as one token
     (the default dictionary splits it into 吸毒 / 人员). No other word is added,
     so the counts of all anchor and other target words are unaffected.
  6. min_count is 5, the same as the US model.

Other training settings: skip-gram, 100 dims, window 5, 10 epochs.
workers=1 and a fixed seed make the run reproducible.

Run:
  conda activate thesis
  python ~/Desktop/retrain_china_v3.py
"""

import os
import re
import unicodedata
import jieba
jieba.add_word("吸毒人员")
from gensim.models import Word2Vec

DESK = os.path.expanduser("~/Desktop")
SRC = os.path.join(DESK, "china_corpus_clean.txt")
OUT_TXT = os.path.join(DESK, "china_corpus_clean_v3.txt")
OUT_MODEL = os.path.join(DESK, "china_word2vec_v3.model")

# Radical supplement characters that NFKC leaves unchanged
RADICAL_MAP = {
    "⺠": "民", "⻔": "门", "⻓": "长", "⻋": "车", "⻢": "马", "⻅": "见",
    "⻄": "西", "⻛": "风", "⻉": "贝", "⻜": "飞", "⻚": "页", "⻰": "龙",
    "⻥": "鱼", "⻦": "鸟", "⻬": "齐", "⻮": "齿", "⻘": "青", "⻩": "黄",
    "⻨": "麦", "⻝": "食", "⻤": "鬼", "⺟": "母", "⻆": "角", "⻁": "虎",
}

DRUG_KEYWORDS = ["毒品", "麻醉", "戒毒", "吸毒", "贩毒", "制毒", "罂粟", "海洛因",
                 "冰毒", "大麻", "精神药品", "易制毒"]
BOILERPLATE = ["欢迎访问", "来源：", "来源:", "www.", "WWW.", "http"]


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    return "".join(RADICAL_MAP.get(c, c) for c in text)


def is_noise_token(w):
    if not w.strip():
        return True
    if all(unicodedata.category(c)[0] in ("P", "S", "Z") for c in w):
        return True
    if re.fullmatch(r"[0-9.,%·]+", w):
        return True
    return False


with open(SRC, encoding="utf-8") as f:
    raw = f.read()

text = normalize(raw)
residual = sorted({c for c in text if 0x2E80 <= ord(c) <= 0x2FDF})
print("=" * 60)
print("Step 1: radical characters")
print(f"  remaining after normalization: {residual if residual else 'none'}")

lines = text.split("\n")
removed_boiler, removed_smuggling, kept = [], [], []
for line in lines:
    s = line.strip()
    if any(b in s for b in BOILERPLATE):
        removed_boiler.append(s)
        continue
    if "走私" in s and not any(k in s for k in DRUG_KEYWORDS):
        removed_smuggling.append(s)
        continue
    kept.append(s)

print("=" * 60)
print(f"Step 2: boilerplate lines removed: {len(removed_boiler)}")
for s in removed_boiler[:10]:
    print("  -", s[:70])
print("=" * 60)
print(f"Step 3: non-drug smuggling lines removed: {len(removed_smuggling)}")
for s in removed_smuggling:
    print("  -", s[:70])

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(kept))

sentences = []
for s in kept:
    if len(s) < 5:
        continue
    words = [w for w in jieba.cut(s) if not is_noise_token(w)]
    if len(words) > 3:
        sentences.append(words)

n_chars = sum(1 for c in "".join(kept) if "一" <= c <= "鿿")
print("=" * 60)
print("Step 4: corpus after corrections")
print(f"  Chinese characters : {n_chars:,}")
print(f"  training lines     : {len(sentences):,}")
print(f"  tokens             : {sum(len(x) for x in sentences):,}")

model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5,
                 workers=1, epochs=10, sg=1, seed=42)
model.save(OUT_MODEL)

print("=" * 60)
print("Model")
print(f"  tokens used in training    : {model.corpus_total_words:,}")
print(f"  vocabulary after min_count : {len(model.wv):,}")
targets = ["成瘾", "滥用", "预防", "阿片类", "阿片", "鸦片", "康复", "吸毒者", "吸毒人员", "吸毒",
           "处方", "可卡因", "海洛因", "贩毒", "走私", "毒品", "大麻",
           "治疗", "患者", "戒毒", "医疗", "健康", "药物",
           "处罚", "犯罪", "违法", "惩治", "刑罚", "逮捕", "罚款"]
print("  frequency of target and anchor words:")
for w in targets:
    freq = model.wv.get_vecattr(w, "count") if w in model.wv else "not in vocabulary"
    print(f"    {w}: {freq}")
print(f"\nSaved: {OUT_TXT}\nSaved: {OUT_MODEL}")
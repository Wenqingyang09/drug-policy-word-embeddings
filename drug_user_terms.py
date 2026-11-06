"""
drug_user_terms.py

Counts how each corpus refers to people who use drugs. It prints frequencies
and segmentation only, no projection scores.

Run:
  conda activate thesis
  python ~/Desktop/drug_user_terms.py
"""

import os
import re
import jieba

DESK = os.path.expanduser("~/Desktop")

cn = open(os.path.join(DESK, "china_corpus_clean_v3.txt"), encoding="utf-8").read()
us = open(os.path.join(DESK, "usa_corpus_clean.txt"), encoding="utf-8").read().lower()
us = re.sub(r"\s+", " ", us)

print("=" * 60)
print("CN: raw string counts in the corrected corpus")
for t in ["吸毒者", "吸毒人员", "吸毒成瘾人员", "吸毒成瘾者", "戒毒人员",
          "被戒毒人员", "涉毒人员", "成瘾者", "吸毒人"]:
    print(f"   {t:<8} {cn.count(t):>6}")

print("\nCN: how jieba segments these terms")
for t in ["吸毒人员", "吸毒成瘾人员", "戒毒人员", "吸毒者"]:
    print(f"   {t} -> {' / '.join(jieba.cut(t))}")

print("=" * 60)
print("US: counts in the cleaned corpus (lower case)")
for t in [r"\buser\b", r"\busers\b", r"\bdrug users?\b", r"\baddicts?\b",
          r"\bpeople who use drugs\b", r"\bpersons? who use drugs\b",
          r"\bindividuals who use drugs\b",
          r"\bpeople with (an? )?(opioid |substance )?use disorders?\b",
          r"\bindividuals with (an? )?(opioid |substance )?use disorders?\b"]:
    print(f"   {t:<62} {len(re.findall(t, us)):>6}")
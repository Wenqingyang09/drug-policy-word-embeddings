"""
users_context.py

Shows how "users" is used in the US corpus, to check whether it mostly refers
to people who use drugs.
Prints the words that most often come right before "users" and 25 random
contexts. No projection scores are computed.

Run:
  conda activate thesis
  python ~/Desktop/users_context.py
"""

import os
import re
import random
from collections import Counter

DESK = os.path.expanduser("~/Desktop")
text = open(os.path.join(DESK, "usa_corpus_clean.txt"), encoding="utf-8").read().lower()
text = re.sub(r"[^a-z\s]", " ", text)
words = text.split()

idx = [i for i, w in enumerate(words) if w == "users"]
print(f"'users' occurs {len(idx)} times")

left = Counter(words[i - 1] for i in idx if i > 0)
print("\nWord directly before 'users' (top 25):")
for w, n in left.most_common(25):
    print(f"   {w:<15}{n}")

random.seed(1)
print("\n25 random contexts:")
for i in sorted(random.sample(idx, min(25, len(idx)))):
    print("   ...", " ".join(words[max(0, i - 8):i]), "[USERS]",
          " ".join(words[i + 1:i + 6]), "...")
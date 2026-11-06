"""
check_names_units.py

Read-only diagnostics. Nothing is trained and nothing is changed.

Part 1 lists every capitalised multi-word phrase in the US corpus that contains
a target word or an anchor word (e.g. "Drug Abuse Warning Network"), with its
frequency, so that the list of institutional names can be checked
systematically. Phrases already covered by the current
list are marked with *.

Part 2 describes the sentence units: how many tokens fall in units longer than
100 tokens, and the beginning of the ten longest units.

Output: printed and saved to ~/Desktop/results_v2_check/check_names_units.txt

Run:
  conda activate thesis
  python ~/Desktop/check_names_units.py
"""

import os
import re
from collections import Counter

DESK = os.path.expanduser("~/Desktop")
OUT = os.path.join(DESK, "results_v2_check")
os.makedirs(OUT, exist_ok=True)
LOG = []


def say(s=""):
    print(s)
    LOG.append(s)


RAW = open(os.path.join(DESK, "usa_corpus_clean.txt"), encoding="utf-8").read()

WORDS = {"addiction", "abuse", "overdose", "prevention", "opioid", "opioids",
         "rehabilitation", "users", "prescription", "cocaine", "heroin", "trafficking",
         "drug", "drugs", "marijuana", "treatment", "patient", "patients", "disorder",
         "disorders", "recovery", "therapy", "health", "care", "punishment", "offense",
         "offender", "offenders", "criminal", "arrest", "sanction", "enforcement"}

CURRENT = [
    "substance abuse and mental health services administration",
    "center for substance abuse prevention", "center for substance abuse treatment",
    "substance abuse prevention and treatment block grant",
    "national institute on drug abuse", "national institute on alcohol abuse and alcoholism",
    "centers for disease control and prevention", "center for disease control and prevention",
    "office of national drug control policy", "national drug control strategy",
    "drug enforcement administration", "high intensity drug trafficking areas",
    "high intensity drug trafficking area",
    "california department of corrections and rehabilitation",
    "department of health and human services", "national survey on drug use and health",
    "immigration and customs enforcement",
]

# ------------------------------------------------------------ part 1
CAP = r"[A-Z][A-Za-z\-]+"
CONN = r"(?:of|and|for|on|the|in|to|&)"
PHRASE = re.compile(rf"{CAP}(?:[ \t]*\n?[ \t]*(?:{CONN}[ \t]*\n?[ \t]*)?{CAP})+")

cnt = Counter()
for m in PHRASE.finditer(RAW):
    p = re.sub(r"\s+", " ", m.group(0)).strip()
    p = re.sub(r"^(?:(?:The|A|An|In|Through|And|For|By|With|Under|This|These|Our|Its|At|As|From)\s+)+", "", p)
    toks = [t.lower().strip("-") for t in p.split()]
    if any(t in WORDS for t in toks) and len(toks) >= 2:
        cnt[p.lower()] += 1

say("=" * 90)
say("Part 1. Capitalised phrases containing a target or anchor word (frequency >= 5)")
say("        * = already covered by the current list of names")
for p, k in cnt.most_common():
    if k < 5:
        break
    mark = "*" if any(c in p for c in CURRENT) else " "
    say(f"   {mark} {k:>5}  {p}")

# ------------------------------------------------------------ part 2
ABBR = r"\b(Dr|Mr|Mrs|Ms|No|Nos|vs|etc|Inc|Jr|Sr|St|Fig|Vol|pp|Sec|Dept|Gov|approx)\."


def sentence_units(text):
    t = re.sub(r"-[ \t]*\n[ \t]*(?=[a-z])", "", text)
    t = re.sub(r"\b((?:[A-Za-z]\.){2,})", lambda m: m.group(1).replace(".", ""), t)
    t = re.sub(ABBR, r"\1", t, flags=re.I)
    out = []
    for block in re.split(r"\n\s*\n", t):
        block = re.sub(r"\s*\n\s*", " ", block)
        for s in re.split(r"(?<=[.!?])[\"'”’)]*\s+", block):
            if len(s.strip()) > 10:
                out.append(s)
    return out


units = sentence_units(RAW)
toks = [len(re.sub(r"[^a-z\s]", " ", u.lower()).split()) for u in units]
total = sum(toks)
long_ = [(n, u) for n, u in zip(toks, units) if n > 100]
say("=" * 90)
say("Part 2. Sentence units")
say(f"   units {len(units)}, tokens {total}")
for lim in (50, 100, 200):
    k = sum(n for n in toks if n > lim)
    say(f"   tokens in units longer than {lim}: {k} ({100 * k / total:.1f}%)")
say("   beginning of the ten longest units:")
for n, u in sorted(long_, reverse=True)[:10]:
    say(f"   [{n} tokens] {re.sub(chr(10), ' ', u)[:200]}")

with open(os.path.join(OUT, "check_names_units.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
print(f"\nSaved: {os.path.join(OUT, 'check_names_units.txt')}")
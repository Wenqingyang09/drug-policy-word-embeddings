import re
import os

input_file = os.path.expanduser("~/Desktop/usa_corpus.txt")
output_file = os.path.expanduser("~/Desktop/usa_corpus_clean.txt")

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned = []
for line in lines:
    if re.match(r"^\s*\d+\s*$", line):
        continue
    if re.match(r"^\s*[^a-zA-Z0-9\s]{1,3}\s*$", line):
        continue
    if len(line.strip()) < 4:
        continue
    line = re.sub(r" +", " ", line)
    cleaned.append(line)

text = "".join(cleaned)
text = re.sub(r"\n{3,}", "\n\n", text)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)

print("Cleaning complete. Output saved to:", output_file)
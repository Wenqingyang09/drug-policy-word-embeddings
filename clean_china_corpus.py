import re
import os

input_file = os.path.expanduser("~/Desktop/china_corpus.txt")
output_file = os.path.expanduser("~/Desktop/china_corpus_clean.txt")

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned = []
for line in lines:
    line = line.strip()
    # Remove URLs
    if re.search(r"www\.|http|\.gov\.cn|\.html", line):
        continue
    # Remove date lines
    if re.match(r"^\d{4}/\d{1,2}/\d{1,2}", line):
        continue
    # Remove very short lines
    if len(line) < 5:
        continue
    # Remove lines with mostly numbers or special characters
    if re.match(r"^[\d\s\W]+$", line):
        continue
    cleaned.append(line)

text = "\n".join(cleaned)
text = re.sub(r"\n{3,}", "\n\n", text)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)

print("Cleaning complete. Output saved to:", output_file)

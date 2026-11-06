import os
import subprocess

import pdfplumber
from docx import Document

# Input folder and output file
input_folder = os.path.expanduser("~/Desktop/Data Source CN")
output_file = os.path.expanduser("~/Desktop/china_corpus.txt")

# Drug-related keywords for filtering the Criminal Law provisions
drug_keywords = [
    "毒品", "麻醉", "戒毒", "吸毒", "贩毒", "制毒",
    "走私", "罂粟", "海洛因", "冰毒", "大麻",
]


def is_drug_related(text):
    return any(keyword in text for keyword in drug_keywords)


def extract_docx(filepath, filter_drug=False):
    doc = Document(filepath)
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if len(text) < 5:
            continue
        if filter_drug and not is_drug_related(text):
            continue
        paragraphs.append(text)
    return "\n".join(paragraphs)


def extract_doc(filepath):
    # .doc files are converted with textutil (macOS)
    result = subprocess.run(
        ["textutil", "-convert", "txt", "-encoding", "UTF-8", "-stdout", filepath],
        capture_output=True, text=True, encoding="utf-8", check=True
    )
    return result.stdout


def extract_pdf(filepath):
    pages_text = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return "\n".join(pages_text)


counts = []
with open(output_file, "w", encoding="utf-8") as out:
    for filename in sorted(os.listdir(input_folder)):
        if filename.startswith(".") or filename.startswith("~"):
            continue
        filepath = os.path.join(input_folder, filename)

        if filename.endswith(".docx"):
            print(f"Processing (docx): {filename}")
            text = extract_docx(filepath, filter_drug="criminal_law" in filename)
        elif filename.endswith(".doc"):
            print(f"Processing (doc): {filename}")
            text = extract_doc(filepath)
        elif filename.endswith(".pdf"):
            print(f"Processing (pdf): {filename}")
            text = extract_pdf(filepath)
        else:
            continue

        # Stop if a file yields no text
        if len(text.strip()) == 0:
            raise RuntimeError(f"No text extracted from {filename}")
        out.write(text + "\n\n")
        counts.append((filename, len(text.strip())))
        print(f"Completed: {filename}")

print(f"Documents extracted: {len(counts)}")
for name, n in counts:
    print(f"  {n:>7,} characters  {name}")
print("All files processed. Output saved to:", output_file)

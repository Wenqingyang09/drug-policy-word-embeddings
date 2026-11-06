import fitz
import pytesseract
from PIL import Image
import io
import os

# Input folder containing US drug policy PDFs
pdf_folder = os.path.expanduser("~/Desktop/Data Source USA")

# Output file for extracted text
output_file = os.path.expanduser("~/Desktop/usa_corpus.txt")

with open(output_file, "w", encoding="utf-8") as out:
    for filename in sorted(os.listdir(pdf_folder)):
        if filename.endswith(".pdf"):
            print(f"Processing: {filename}")
            pdf_path = os.path.join(pdf_folder, filename)
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render page as image at 300 DPI for OCR accuracy
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                # Extract text using Tesseract OCR
                text = pytesseract.image_to_string(img, lang="eng")
                out.write(text + "\n")
            doc.close()
            print(f"Completed: {filename}")

print("All files processed. Output saved to:", output_file)
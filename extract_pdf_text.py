import os
import sys
from pypdf import PdfReader

def extract_text(pdf_path, output_file):
    output_file.write(f"--- Extracting text from: {pdf_path} ---\n")
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                output_file.write(f"Page {i+1}:\n")
                output_file.write(text + "\n")
                output_file.write("-" * 20 + "\n")
            except Exception as e:
                output_file.write(f"Error reading page {i+1} of {pdf_path}: {e}\n")
    except Exception as e:
        output_file.write(f"Error reading {pdf_path}: {e}\n")

pdf_files = [
    "Copy of depin aders.pptx.pdf",
    "eve310_finalreport.pdf"
]

with open("output_utf8.txt", "w", encoding="utf-8") as f:
    for pdf in pdf_files:
        if os.path.exists(pdf):
            extract_text(pdf, f)
        else:
            f.write(f"File not found: {pdf}\n")

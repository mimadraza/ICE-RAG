import os
from pathlib import Path
from docling.document_converter import DocumentConverter


def get_text_from_pdf(source, converter,output_file):
    result = converter.convert(source)    
    doc = result.document
    output_file.write_text(doc.export_to_markdown(), encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent 
OUTPUT_DIR = BASE_DIR / "extracted_texts"

coverter = DocumentConverter()
counter = 1

for file in os.listdir(BASE_DIR / "corpus"):
    file_name = 'text' + str(counter) + '.md'
    output_path = OUTPUT_DIR / file_name
    get_text_from_pdf(BASE_DIR / "corpus" / file, coverter, output_path)
    counter += 1
    print(f"Extracted text from {file} and saved to {output_path}")

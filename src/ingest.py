from pypdf import PdfReader
import os

pdf_path = os.path.join("data", "sample.pdf")

reader = PdfReader(pdf_path)
documents = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    documents.append({
        "page_content": text,
        "metadata": {"source": pdf_path, "page": i}
    })

print(f"Number of pages loaded: {len(documents)}")
print("\n--- Content of Page 1 ---\n")
print(documents[0]["page_content"][:500])
print("\n--- Metadata of Page 1 ---\n")
print(documents[0]["metadata"])

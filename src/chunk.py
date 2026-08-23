from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# --- step 1 Load the PDF file and extract text from each page ---(ingestion)
pdf_path = os.path.join("data", "sample.pdf")

reader = PdfReader(pdf_path)
documents = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    documents.append({
        "page_content": text,
        "metadata": {"source": pdf_path, "page": i}
    })

# --- step 2: combine all page text(Optional but common) ---
full_text = "\n\n".join([doc["page_content"] for doc in documents])


# --- step 3: Setup the splitter ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

# --- step 4: Split into chunks ---
chunks = splitter.split_text(full_text)
 
# --- step 5: Inspect the results ---
print(f"Total chunks created: {len(chunks)}")
print("--- Chunk 1 ---")
print(chunks[0])
print("--- Chunk 2 ---")
print(chunks[1])
print(f"\nLength of chunk 1: {len(chunks[0])} characters")
print(f"Length of chunk 2: {len(chunks[1])} characters")




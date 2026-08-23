from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
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

full_text = "\n\n".join([doc["page_content"] for doc in documents])


# --- step 2: Chunk the text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

chunks = splitter.split_text(full_text)
 
# --- step 3: Load the embedding model ---
# This downloads the model the first time, then cached it locally

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- step 4: Generate embeddings for all chunks ---
embeddings = model.encode(chunks)

# --- step 5: Inspect the result ---
print(f"\nShape of embeddigs array: {embeddings.shape}")
print(f"Each chunk became a vector of {embeddings.shape[1]} numbers")
print("\n--- First 10 numbers of chunk 1's embedding ---")
print(embeddings[0][:10])





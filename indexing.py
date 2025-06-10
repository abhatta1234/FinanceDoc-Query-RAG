'''
This script indexes the documents in the docs folder
Creates the Faiss index and saves it to the index folder
for the retrieval later

'''


# 1. Read the documents
import os
import fitz  # PyMuPDF
import glob
from tqdm import tqdm
import re

def extract_pdf_with_metadata(pdf_path):
    """Extract text from PDF with page numbers and better structure."""
    doc_text = []
    with fitz.open(pdf_path) as doc:
        # Get document metadata
        filename = os.path.basename(pdf_path)
        total_pages = len(doc)
        
        # Process each page
        for page_num, page in enumerate(doc):
            # Extract text with more parameters for better extraction
            text = page.get_text("text", sort=True, flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
            
            # Clean the text
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
            text = text.strip()
            
            if text:  # Only add non-empty pages
                doc_text.append({
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1,
                        "total_pages": total_pages
                    }
                })
    return doc_text

# Process all PDFs in the docs directory
all_documents = []
pdf_files = glob.glob("docs/*.pdf")

print(f"Processing {len(pdf_files)} PDF files...")
for pdf_file in tqdm(pdf_files):
    all_documents.extend(extract_pdf_with_metadata(pdf_file))

print(f"Extracted {len(all_documents)} pages from PDFs")

# 2. Better chunking with semantic awareness
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,  # Increased overlap for better context preservation
    separators=["\n\n", "\n", ".", ";", ",", " ", ""]
)

all_chunks = []
all_metadatas = []

print("Chunking documents...")
for doc in tqdm(all_documents):
    chunks = text_splitter.split_text(doc["text"])
    # Add metadata to each chunk
    for chunk in chunks:
        if len(chunk.split()) >= 20:  # Only keep chunks with at least 20 words
            all_chunks.append(chunk)
            all_metadatas.append(doc["metadata"])

print(f"Created {len(all_chunks)} chunks")

# 3. Embed the chunks with progress
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Generating embeddings...")
embeddings = model.encode(all_chunks, show_progress_bar=True)

print(f"Created embeddings with shape: {embeddings.shape}")

# 4. Improved ChromaDB indexing
import chromadb
from chromadb.config import Settings

# Define the persist directory with a more portable path
persist_directory = "index/chroma_db"
os.makedirs(persist_directory, exist_ok=True)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=persist_directory)

# Create or get a collection
collection = client.get_or_create_collection(
    name="annual_report_chunks",
    metadata={"hnsw:space": "cosine"}
)

# Add documents with their metadata
print("Indexing documents in ChromaDB...")
collection.add(
    documents=all_chunks,
    embeddings=embeddings.tolist(),
    metadatas=all_metadatas,
    ids=[f"chunk_{i}" for i in range(len(all_chunks))]
)

print(f"Indexing complete! Collection count: {collection.count()}")

# # create faiss index
# dimension = embeddings[0].shape[0]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)

# # save the index
# faiss.write_index(index, "index.faiss")

# # load the index
# index = faiss.read_index("index.faiss")

# When you query later, it will use cosine similarity to find the most similar chunks
results = collection.query(
    query_embeddings=[your_query_embedding],
    n_results=3  # Get top 3 most similar chunks
)


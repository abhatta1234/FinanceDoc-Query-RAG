'''
This script indexes the documents in the docs folder
Creates the Faiss index and saves it to the index folder
for the retrieval later

'''


# 1. Read the documents

import fitz
document = ""
with fitz.open("docs/annualreport-2024.pdf") as doc:
    for page in doc:
        text = page.get_text()
        document += text

# 2. Split the documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)


chunks = text_splitter.split_text(document)

print("Number of chunks:")
print(len(chunks))

print("First 5 chunks:")
for i in range(5):
    print(chunks[i],len(chunks[i]))
    print("-"*100)


# 3. Embed the chunks
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks,show_progress_bar=True)

print("Shape of embeddings:")
print(embeddings.shape)


# create the ChromaDB index

import chromadb
from chromadb.config import Settings
import os

# Define the persist directory
persist_directory = "/scratch365/abhatta/Custom-RAG-exploration/index/chroma_db"
os.makedirs(persist_directory, exist_ok=True)

# Initialize ChromaDB client with proper settings
client = chromadb.PersistentClient(path=persist_directory)

# Create or get a collection
collection = client.get_or_create_collection(
    name="annual_report_chunks",
    metadata={"hnsw:space": "cosine"}  # Using cosine similarity for better results
)

# Add documents and embeddings
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),  # Convert numpy array to list
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

# Verify the collection was created and data was added
print(f"Collection count: {collection.count()}")
print(f"Persist directory: {persist_directory}")

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

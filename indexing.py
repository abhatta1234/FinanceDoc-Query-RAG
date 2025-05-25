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


# # create faiss index
# dimension = embeddings[0].shape[0]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)

# # save the index
# faiss.write_index(index, "index.faiss")

# # load the index
# index = faiss.read_index("index.faiss")

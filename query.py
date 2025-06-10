import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

# Near the top of your script
parser = argparse.ArgumentParser()
parser.add_argument("--query", type=str, default="What is total net profit for the year 2023, 2024 and 2025?")
parser.add_argument("--chunks", type=int, default=8, help="Number of chunks to retrieve")
args = parser.parse_args()

# 1. Load ChromaDB collection - update the collection name
persist_directory = "/scratch365/abhatta/Custom-RAG-exploration/index/chroma_db"
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("annual_report_chunks_e5_large")  # Updated collection name

# 2. Embed the user query - update the embedding model
model = SentenceTransformer("intfloat/e5-large-v2")  # Updated to match indexing
user_query = args.query
query_embedding = model.encode([user_query])

# 3. Query the collection
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=args.chunks
)

# 4. Print the retrieved chunks
print("\n=== RETRIEVED CHUNKS ===")
for i, (chunk, cid) in enumerate(zip(results['documents'][0], results['ids'][0])):
    print(f"Chunk {i+1} (ID: {cid}):\n{chunk}\n{'-'*80}")

# 5. Load Phi-2 model
print("\n=== LOADING PHI-2 MODEL ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
phi_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

# Function to generate response using Phi-2
def generate_phi2_response(prompt, max_new_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = phi_model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.92,
            repetition_penalty=1.3,
            do_sample=True,
            num_beams=3,
            no_repeat_ngram_size=3
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    response = response.replace(".", ". ").replace(",", ", ")
    response = ' '.join(response.split())
    
    return response

print("\n=== QUESTION ONLY RESPONSE ===")
# Generate answer with only the question
prompt_question_only = f"""You are a financial analyst assistant.
Question: {user_query}
Answer:"""

print("Generating answer without context...")
response_question_only = generate_phi2_response(prompt_question_only)
print("\nGenerated Answer (Question Only):")
print(response_question_only)

print("\n=== CONTEXT-BASED RESPONSE ===")
# Generate answer with context + question
context = "\n\n".join(results['documents'][0])
prompt_with_context = f"""You are a financial analyst assistant that answers questions based on annual reports.
Below is some context information from annual reports, followed by a question.

Instructions:
1. Answer the question using only the provided context.
2. Use proper formatting with spaces between words and after punctuation.
3. Provide a clear, well-structured response with complete sentences.
4. If the context doesn't contain the answer, say "I don't have enough information to answer this question."

Context:
{context}

Question: {user_query}

Answer (use proper spacing and formatting):"""

print("Generating answer with context...")
response_with_context = generate_phi2_response(prompt_with_context)
print("\nGenerated Answer (With Context):")
print(response_with_context)



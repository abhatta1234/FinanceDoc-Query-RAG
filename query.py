import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from typing import Literal, Optional
import textwrap

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--query", type=str, default="What is total net profit for the year 2023, 2024 and 2025?")
parser.add_argument("--model_size", type=str, choices=['small', 'medium', 'large'], default='medium',
                    help="Model size to use: small (0.5B), medium (1.1B), or large (2.7B)")
args = parser.parse_args()

# Model configurations
MODEL_CONFIGS = {
    'small': {
        'name': 'Qwen/Qwen-0.5B',
        'max_chunks': 4,
        'max_new_tokens': 200,
    },
    'medium': {
        'name': 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
        'max_chunks': 6,
        'max_new_tokens': 250,
    },
    'large': {
        'name': 'microsoft/phi-2',
        'max_chunks': 8,
        'max_new_tokens': 300,
    }
}

class ModelManager:
    def __init__(self, model_size: Literal['small', 'medium', 'large']):
        self.config = MODEL_CONFIGS[model_size]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nInitializing {model_size} model: {self.config['name']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['name'])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['name'],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens or self.config['max_new_tokens'],
                temperature=0.7,
                top_p=0.92,
                repetition_penalty=1.2,
                do_sample=True,
                num_beams=3,
                no_repeat_ngram_size=3
            )
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return ' '.join(response.split())  # Clean up spacing

def format_chunk(text: str, width: int = 100) -> str:
    """Format text chunk with proper wrapping"""
    return textwrap.fill(text, width=width)

# Initialize ChromaDB and embedding model
persist_directory = "index/chroma_db"
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("annual_report_chunks_e5_large")
embedding_model = SentenceTransformer("intfloat/e5-large-v2")

# Initialize the main model based on selected size
model_manager = ModelManager(args.model_size)

# Embed and retrieve relevant chunks
print("\n=== RETRIEVING RELEVANT CHUNKS ===")
query_embedding = embedding_model.encode([args.query])
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=MODEL_CONFIGS[args.model_size]['max_chunks']
)

# Display raw chunks
print("\n=== RAW RETRIEVED CHUNKS ===")
raw_chunks = results['documents'][0]
for i, chunk in enumerate(raw_chunks, 1):
    print(f"\nChunk {i}:")
    print(format_chunk(chunk))
    print("-" * 80)

# Summarize chunks
print("\n=== SUMMARIZING CHUNKS ===")
summarization_prompt = f"""Summarize the following text from an annual report, focusing on key financial information and important details:

{' '.join(raw_chunks)}

Summary:"""

summarized_context = model_manager.generate(summarization_prompt)
print("\nSummarized Context:")
print(format_chunk(summarized_context))
print("-" * 80)

# Generate answer without context
print("\n=== GENERATING ANSWER WITHOUT CONTEXT ===")
no_context_prompt = f"""You are a financial analyst assistant.
Question: {args.query}
Answer:"""

response_no_context = model_manager.generate(no_context_prompt)
print("\nAnswer (Without Context):")
print(format_chunk(response_no_context))

# Generate answer with summarized context
print("\n=== GENERATING ANSWER WITH CONTEXT ===")
context_prompt = f"""You are a financial analyst assistant. Answer the question based on the following context from annual reports.

Context:
{summarized_context}

Question: {args.query}

Answer:"""

response_with_context = model_manager.generate(context_prompt)
print("\nAnswer (With Context):")
print(format_chunk(response_with_context))

# Print model and retrieval settings used
print("\n=== SETTINGS USED ===")
print(f"Model: {MODEL_CONFIGS[args.model_size]['name']}")
print(f"Chunks Retrieved: {len(raw_chunks)}")
print(f"Model Max Tokens: {MODEL_CONFIGS[args.model_size]['max_new_tokens']}")



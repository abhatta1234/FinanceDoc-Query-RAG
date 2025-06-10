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
        'name': 'microsoft/phi-1_5',
        'max_tokens': 100,
        'num_chunks': 3,
    },
    'medium': {
        'name': 'microsoft/phi-2',
        'max_tokens': 150,
        'num_chunks': 4,
    },
    'large': {
        'name': 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF',
        'max_tokens': 200,
        'num_chunks': 5,
    }
}

class ModelManager:
    def __init__(self, model_size: Literal['small', 'medium', 'large']):
        self.config = MODEL_CONFIGS[model_size]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nInitializing {model_size} model: {self.config['name']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['name'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['name'],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, prompt: str) -> str:
        # First check if prompt is too long
        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        if len(tokens.input_ids[0]) > 1500:  # Leave room for generation
            # Truncate the context part of the prompt
            context_start = prompt.find("Financial Report:")
            if context_start != -1:
                # Keep the instruction and question, truncate the context
                before_context = prompt[:context_start]
                context_and_after = prompt[context_start:]
                context_end = context_and_after.find("\n\nQuestion:")
                if context_end != -1:
                    context = context_and_after[:context_end]
                    after_context = context_and_after[context_end:]
                    
                    # Tokenize just the context to see how much we need to truncate
                    context_tokens = self.tokenizer(context, truncation=False)
                    if len(context_tokens.input_ids) > 1000:  # Arbitrary threshold
                        # Take first and last part of the context
                        context_parts = context.split('\n\n')
                        selected_parts = context_parts[:2] + ['...'] + context_parts[-2:]
                        context = '\n\n'.join(selected_parts)
                    
                    prompt = before_context + context + after_context

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1800,  # Leave room for generation
            add_special_tokens=True,
            return_attention_mask=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs['attention_mask'],
                max_new_tokens=self.config['max_tokens'],
                do_sample=False,  # Use greedy decoding
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # Clean up response - take only the first sentence that makes sense
        sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 0]
        return sentences[0] + '.' if sentences else "No clear answer found."

def format_chunk(text: str, width: int = 100) -> str:
    """Format text chunk with proper wrapping"""
    return textwrap.fill(text, width=width)

def enhance_query(query: str) -> str:
    """Enhance the query to improve retrieval quality."""
    # Remove any question-specific words that might confuse embedding
    question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'is', 'are', 'was', 'were']
    words = query.lower().split()
    keywords = [w for w in words if w not in question_words]
    
    # Add context hints based on common financial terms in the query
    financial_terms = {
        'profit': 'profit revenue earnings income',
        'revenue': 'revenue income earnings sales',
        'cost': 'cost expense expenditure spending',
        'growth': 'growth increase improvement',
        'loss': 'loss deficit decrease decline',
        'ceo': 'ceo executive management leadership',
        'board': 'board directors governance management',
        'dividend': 'dividend payout distribution shareholders',
        'market': 'market industry sector business'
    }
    
    enhanced_terms = []
    for word in keywords:
        enhanced_terms.append(word)
        for term, context in financial_terms.items():
            if term in word:
                enhanced_terms.extend(context.split())
    
    return ' '.join(enhanced_terms)

def rerank_chunks(chunks: list, query: str, metadata: list) -> tuple:
    """Rerank chunks based on relevance and metadata."""
    scored_chunks = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
        score = 0
        # Prefer chunks with exact matches
        query_terms = query.lower().split()
        chunk_lower = chunk.lower()
        for term in query_terms:
            if term in chunk_lower:
                score += 1
        
        # Boost score based on metadata
        if meta.get('page', 0) <= 5:  # Prefer early pages (often contain summaries)
            score += 0.5
        
        scored_chunks.append((score, i, chunk, meta))
    
    # Sort by score and return original order for stable output
    scored_chunks.sort(reverse=True)
    return ([c[2] for c in scored_chunks], [c[3] for c in scored_chunks])

# Initialize ChromaDB and embedding model
persist_directory = "index/chroma_db"
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("annual_report_chunks_e5_large")

# Initialize embedding model with GPU support
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("intfloat/e5-large-v2")
embedding_model.to(device)

# Print device information
print(f"\n=== DEVICE INFORMATION ===")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# Initialize the main model based on selected size
model_manager = ModelManager(args.model_size)

# Enhance query and retrieve relevant chunks
print("\n=== RETRIEVING RELEVANT CHUNKS ===")
enhanced_query = enhance_query(args.query)
print(f"Enhanced query: {enhanced_query}")

query_embedding = embedding_model.encode([enhanced_query])
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=MODEL_CONFIGS[args.model_size]['num_chunks'] * 2  # Get more chunks for reranking
)

# Rerank and filter chunks
reranked_chunks, reranked_metadata = rerank_chunks(
    results['documents'][0],
    args.query,
    results['metadatas'][0]
)

# Take only the top N chunks based on model config
raw_chunks = reranked_chunks[:MODEL_CONFIGS[args.model_size]['num_chunks']]
raw_context = "\n\n".join(raw_chunks)

# Display chunks with relevance info
print("\n=== RETRIEVED CHUNKS ===")
for i, chunk in enumerate(raw_chunks, 1):
    print(f"\nChunk {i} (Page {reranked_metadata[i-1].get('page', 'N/A')}):")
    print(format_chunk(chunk))
    print("-" * 80)

# Generate answer without context
print("\n=== GENERATING ANSWER WITHOUT CONTEXT ===")
no_context_prompt = f"""Instruction: Answer this question in one clear sentence. If you don't know, say "I need to check the financial reports."

Question: {args.query}

Answer: """

response_no_context = model_manager.generate(no_context_prompt)
print("\nAnswer:")
print(format_chunk(response_no_context))

# Generate answer with raw context
print("\n=== GENERATING ANSWER WITH CONTEXT ===")
context_prompt = f"""Instruction: Using only the information in this financial report, answer the question in one clear sentence.

Financial Report:
{raw_context}

Question: {args.query}

If the information is not in the report, say "This information is not in the report."

Answer: """

response_with_context = model_manager.generate(context_prompt)
print("\nAnswer:")
print(format_chunk(response_with_context))

# Settings output
print("\n=== SETTINGS USED ===")
print(f"Model: {MODEL_CONFIGS[args.model_size]['name']}")
print(f"Chunks Retrieved: {len(raw_chunks)}")
print(f"Max Tokens: {MODEL_CONFIGS[args.model_size]['max_tokens']}")
print(f"Generation: Greedy decoding (no sampling)")



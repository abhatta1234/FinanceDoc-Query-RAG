import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from typing import Dict, Tuple
import re

# Set page configuration
st.set_page_config(
    page_title="📊 Annual Report RAG System",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state['processing'] = False
if 'debug_info' not in st.session_state:
    st.session_state['debug_info'] = ""
if 'error_message' not in st.session_state:
    st.session_state['error_message'] = ""
if 'model_info' not in st.session_state:
    st.session_state['model_info'] = ""

# Model configurations
MODELS: Dict[str, Dict] = {
    "Small": {
        "repo": "Qwen/Qwen1.5-0.5B-Chat",  # ~0.5 B params
        "chunks": 2,
        "max_tokens": 128,
    },
    "Medium": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # ~1.1 B params
        "chunks": 4,
        "max_tokens": 192,
    },
    "Large": {
        "repo": "microsoft/phi-2",  # 2.7 B params
        "chunks": 6,
        "max_tokens": 256,
    },
}

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("intfloat/e5-large-v2")

@st.cache_resource(show_spinner=False)
def load_chromadb():
    db_path = "/Users/amanbhatta/Custom-RAG-exploration/index/chroma_db"
    client = chromadb.PersistentClient(path=db_path)
    return client.get_collection("annual_report_chunks_e5_large")

@st.cache_resource(show_spinner=False)
def load_llm(repo: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    device = "cpu"  # Force CPU for low-resource environments
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, device

def generate(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    max_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs.input_ids.to(device)
    attn_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    if text == "":
        return "⚠️ Model could not produce an answer."
    return text

# ----------------------------
# Helper: deterministic extraction (moved up before use)
# ----------------------------

def extract_net_value(text: str, year: str, metric_keywords=("net income", "net profit")) -> str:
    """Extract a dollar value associated with a given year and metric keywords from text."""
    pattern_dollar = re.compile(rf"{year}[^\n$]*\$\s?([0-9,.]+)", re.IGNORECASE)
    pattern_metric = re.compile(rf"({'|'.join(metric_keywords)})[^\n$]*\$\s?([0-9,.]+)", re.IGNORECASE)

    for line in text.splitlines():
        m = pattern_dollar.search(line)
        if m:
            return m.group(1)

    for line in text.splitlines():
        m = pattern_metric.search(line)
        if m:
            return m.group(2)

    return ""

# App title and description
st.title("📊 Annual Report Analysis System")
st.caption("Ask a question about the annual reports — see how the model responds with and without retrieved context.")

# Model selector
model_size = st.radio(
    "Model size",
    list(MODELS.keys()),
    index=0,
    horizontal=True,
)
model_cfg = MODELS[model_size]

# Question input (locked while processing)
query = st.text_input(
    "What would you like to know?",
    placeholder="e.g. What was the net profit for 2023?",
    disabled=st.session_state.processing,
)

# Early exit
if query == "":
    st.info("Enter a question and press Enter ↩️ to run.")
    st.stop()

# ---------------
# Processing Flow
# ---------------

if not st.session_state.processing and query:
    st.session_state.processing = True

    # 1. Load resources
    with st.spinner("Loading models & database …"):
        embed_model = load_embedding_model()
        collection = load_chromadb()
        tokenizer, llm, device = load_llm(model_cfg["repo"])

    # 2. Retrieve relevant chunks
    with st.spinner("Retrieving relevant context …"):
        query_emb = embed_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_emb, n_results=model_cfg["chunks"])
        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
        context = "\n\n".join(chunks)

    # 3. REFINE: Use the model to condense/align context
    with st.spinner("Summarizing retrieved context …"):
        refine_prompt = (
            "You are an expert analyst assistant. Produce a concise, bullet-point briefing that contains ONLY the information from the context most relevant to answering the given question. "
            "Include important facts, numbers, definitions, or passages verbatim if necessary. Do NOT add information that is not in the context.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nBriefing (bullets):" 
        )
        briefing = generate(refine_prompt, tokenizer, llm, device, max_tokens=192)

    # 4. Generate FINAL answer using the summary
    # Limit raw context to keep within model limits
    raw_snippet = context[:1500]

    with st.spinner("Generating answer …"):
        prompt_ctx = (
            "You are a knowledgeable assistant. Use ONLY the information in the briefing and the raw context snippet to answer the question. "
            "If the answer cannot be found, reply 'Not available in provided context'. Do NOT hallucinate.\n\n"
            f"Briefing:\n{briefing}\n\nRaw context snippet:\n{raw_snippet}\n\nQuestion: {query}\nAnswer:"
        )
        answer_ctx = generate(prompt_ctx, tokenizer, llm, device, model_cfg["max_tokens"])

    # 5. Display results
    st.subheader("🟢 Answer")
    st.write(answer_ctx)

    # 6. Retrieved chunks and summary (collapsible)
    with st.expander("📄 Retrieved Context", expanded=False):
        st.markdown("#### 🔎 Summarized Context (Briefing)")
        st.write(briefing)
        st.markdown("---")
        st.markdown("#### 📄 Raw Chunks")
        for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
            source = meta.get("source", "?")
            page = meta.get("page", "?")
            st.markdown(f"**Chunk {i+1} — {source} (page {page})**")
            st.text(chunk)

    # 7. Reset processing flag
    st.session_state.processing = False

else:
    # Display placeholder when no query is entered
    st.info("Enter a question above to get started!")
    
    # Show model information
    st.markdown("### Model Information")
    for size, config in MODELS.items():
        st.markdown(f"""
        **{size.title()} Model ({config['repo']})**
        - Context chunks: {config['chunks']}
        - Max answer length: {config['max_tokens']} tokens
        """)
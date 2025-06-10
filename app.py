import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# Set page configuration
st.set_page_config(
    page_title="Annual Report RAG System",
    page_icon="📊",
    layout="wide"
)

# Cache model loading to avoid reloading on each interaction
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("intfloat/e5-large-v2")

@st.cache_resource
def load_llm():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    return tokenizer, model, device

@st.cache_resource
def load_chromadb():
    persist_directory = "/Users/amanbhatta/Custom-RAG-exploration/index/chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection("annual_report_chunks_e5_large")
    return collection

# Function to generate response using Phi-2
def generate_phi2_response(prompt, tokenizer, model, device, max_new_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
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
    
    # Post-process for better formatting
    response = response.replace(".", ". ").replace(",", ", ")
    response = ' '.join(response.split())
    
    return response

# App title and description
st.title("📊 Annual Report Analysis System")
st.markdown("""
This system uses state-of-the-art AI to analyze annual reports. 
Ask questions about financial information, and get answers extracted directly from the reports.
""")

# Load models
with st.spinner("Loading models... (this may take a minute on first run)"):
    embedding_model = load_embedding_model()
    tokenizer, phi_model, device = load_llm()
    collection = load_chromadb()

# Sidebar configuration
st.sidebar.header("Configuration")
num_chunks = st.sidebar.slider("Number of chunks to retrieve", 1, 15, 8)
show_retrieved = st.sidebar.checkbox("Show retrieved passages", False)

# Query input
query = st.text_input("What would you like to know about the annual reports?", 
                    placeholder="e.g., What was the net profit for 2023?")

# Process the query when entered
if query:
    start_time = time.time()
    
    # Embed the query
    with st.spinner("Retrieving relevant information..."):
        query_embedding = embedding_model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=num_chunks
        )
        retrieval_time = time.time() - start_time
    
    # Display retrieved chunks if option is selected
    if show_retrieved:
        with st.expander("Retrieved Passages", expanded=True):
            for i, (chunk, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                st.markdown(f"**Chunk {i+1}** (Source: {metadata['source']}, Page: {metadata['page']})")
                st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
    
    # Generate answers
    col1, col2 = st.columns(2)
    
    with col1:
        start_gen_time = time.time()
        with st.spinner("Generating answer based only on model knowledge..."):
            # Generate answer with only the question
            prompt_question_only = f"""You are a financial analyst assistant.
Question: {query}
Answer:"""
            response_question_only = generate_phi2_response(prompt_question_only, tokenizer, phi_model, device)
            no_context_time = time.time() - start_gen_time
        
        st.markdown("### Without Context 🔴")
        st.markdown(f"<div style='border-left: 4px solid red; padding-left: 10px;'>{response_question_only}</div>", unsafe_allow_html=True)
        st.caption(f"Generated in {no_context_time:.2f} seconds")
    
    with col2:
        start_gen_time = time.time()
        with st.spinner("Generating answer based on annual report data..."):
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

Question: {query}

Answer (use proper spacing and formatting):"""
            response_with_context = generate_phi2_response(prompt_with_context, tokenizer, phi_model, device)
            with_context_time = time.time() - start_gen_time
        
        st.markdown("### With Context 🟢")
        st.markdown(f"<div style='border-left: 4px solid green; padding-left: 10px;'>{response_with_context}</div>", unsafe_allow_html=True)
        st.caption(f"Generated in {with_context_time:.2f} seconds")

    # Add visual indicator of which is better
    st.markdown("### Analysis Comparison")
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <div style="display: inline-block; text-align: center;">
                <span style="font-size: 24px; color: green;">⬆️</span>
                <p><strong>With Context (Reliable)</strong><br>Grounded in actual document data</p>
            </div>
            <div style="display: inline-block; margin-left: 50px; margin-right: 50px; text-align: center;">
                <span style="font-size: 24px;">vs</span>
            </div>
            <div style="display: inline-block; text-align: center;">
                <span style="font-size: 24px; color: red;">⬇️</span>
                <p><strong>Without Context (Caution)</strong><br>May contain inaccuracies</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Performance metrics
    st.markdown("---")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("Retrieval Time", f"{retrieval_time:.2f}s")
    with metrics_col2:
        st.metric("Response Time (No Context)", f"{no_context_time:.2f}s")
    with metrics_col3:
        st.metric("Response Time (With Context)", f"{with_context_time:.2f}s")

else:
    # Display placeholder when no query is entered
    st.info("Enter a question above to get started!")
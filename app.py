import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb

# Load models and DB once
@st.cache_resource
def load_resources():
    persist_directory = "/scratch365/abhatta/Custom-RAG-exploration/index/chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection("annual_report_chunks")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return collection, embedder, generator

collection, embedder, generator = load_resources()

st.title("RAG Demo: Context vs No-Context")

user_query = st.text_input("Enter your question:")

if user_query:
    # Embed and retrieve context
    query_embedding = embedder.encode([user_query])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=3)
    retrieved_chunks = results['documents'][0]
    context = "\n\n".join(retrieved_chunks)

    # Prompt for no context
    prompt_question_only = f"Question: {user_query}\nAnswer:"
    response_question_only = generator(prompt_question_only, max_new_tokens=150)
    answer_no_context = response_question_only[0]['generated_text'][len(prompt_question_only):].strip()

    # Prompt with context
    prompt_with_context = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    response_with_context = generator(prompt_with_context, max_new_tokens=150)
    answer_with_context = response_with_context[0]['generated_text'][len(prompt_with_context):].strip()

    st.subheader("Answer (No Context):")
    st.write(answer_no_context)

    st.subheader("Answer (With Context):")
    st.write(answer_with_context)
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 1. Load ChromaDB collection
persist_directory = "/scratch365/abhatta/Custom-RAG-exploration/index/chroma_db"
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("annual_report_chunks")

# 2. Embed the user query
model = SentenceTransformer("all-MiniLM-L6-v2")
user_query = "What are the main achievements in the 2024 annual report?"
query_embedding = model.encode([user_query])

# 3. Query the collection
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=5
)

# 4. Print the results
for chunk, cid in zip(results['documents'][0], results['ids'][0]):
    print(f"ID: {cid}\nText: {chunk}\n{'-'*40}")

# Load the local LLM pipeline (do this once, at the top of your script)
generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")


print("###########################  Only Question   ##################################################")
# Generate answer with only the question
prompt_question_only = f"Question: {user_query}\nAnswer:"
print("Prompt for question only: ", prompt_question_only)
response_question_only = generator(prompt_question_only, max_new_tokens=150)
print("\nGenerated Answer (Question Only):")
print(response_question_only[0]['generated_text'][len(prompt_question_only):].strip())

print("###########################  With Context   ##################################################")
# Generate answer with context + question
context = "\n\n".join(results['documents'][0])
prompt_with_context = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
print("Prompt for context + question: ", prompt_with_context)
response_with_context = generator(prompt_with_context, max_new_tokens=150)
print("\nGenerated Answer (With Context):")
print(response_with_context[0]['generated_text'][len(prompt_with_context):].strip())


'''
N=3 outputs




Question: What are the main achievements in the 2024 annual report?
Answer:

Generated Answer (With Context):
Financial markets continued to perform exceptionally well, with our strong performance measured
 by stock performance. We have outperformed the S&P 500 Index and the S&P Financials 
 Index for 10 years or more. Our Progress in Measuring Performance is also exceptional. 
 Specifically, for 10 years or more, we have outperformed the S&P 500 Index and the S&P Financials Index 
 for 10 years or mo


 Prompt for question only:  Question: What are the main achievements in the 2024 annual report?
Answer:

Generated Answer (Question Only):
Some of the main achievements in the 2024 annual report are:
1. Strong sales growth, up 10.4%
2. Increased profitability, up 15.4%
3. Reduced operating expenses, down 6.1%
4. Increased market share, up 0.5%
5. Strong brand reputation, up 3.5%
6. Improved financial performance, up 15.4%
7. Continued focus on sustainability, up 2.4%
8. Investment in R&D and innovation, up 4.1%


'''
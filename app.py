from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import os
import requests
import pinecone

app = Flask(__name__)

# Hardcoded Pinecone API Key
PINECONE_API_KEY = "pcsk_3xBdCP_4pWeFB6PgW2tyUg9Amoxc4RwCYLa9h3cPiiyaXcsJzeCTceUuhg5Z6aJGjQkZ7M"  # <-- Replace with your actual key

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index setup
index_name = "medicalbot"

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os

# Hardcoded Pinecone API Key
PINECONE_API_KEY = "pcsk_3xBdCP_4pWeFB6PgW2tyUg9Amoxc4RwCYLa9h3cPiiyaXcsJzeCTceUuhg5Z6aJGjQkZ7M"
INDEX_NAME = "medicalbot"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Explicitly set the API key in the environment (Prevents issues)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)



retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Hardcoded Groq API Key
GROQ_API_KEY = "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"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def generate_response(query, retrieved_docs):
    """Generate a response using the Groq API."""
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant for question-answering tasks. "
                           "Use the following pieces of retrieved context to answer the question. "
                           "If you don't know the answer, say that you don't know. "
                           "Use three sentences maximum and keep the answer concise."
            },
            {"role": "user", "content": f"Query: {query}\n\nContext: {context}"}
        ]
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)

    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Error: Unexpected response from API.")


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()  # Avoid KeyError
    if not msg:
        return "Error: Empty input"

    print(f"User Input: {msg}")

    retrieved_docs = retriever.invoke(msg)  # Retrieve documents dynamically
    response = generate_response(msg, retrieved_docs)

    print(f"Response: {response}")
    return response


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render ka PORT use karo
    app.run(host='0.0.0.0', port=port)
    

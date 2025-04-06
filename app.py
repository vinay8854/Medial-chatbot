from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import os
import requests
from collections import deque
from pinecone import Pinecone

app = Flask(__name__)

# Hardcoded Pinecone API Key
PINECONE_API_KEY = "pcsk_3xBdCP_4pWeFB6PgW2tyUg9Amoxc4RwCYLa9h3cPiiyaXcsJzeCTceUuhg5Z6aJGjQkZ7M"
INDEX_NAME = "medicalbot"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Set API key explicitly in environment (Prevents issues)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Load Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Hardcoded Groq API Key
GROQ_API_KEY = "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"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Store last 5 exchanges for chat memory
chat_history = deque(maxlen=5)

def generate_response(query, retrieved_docs):
    """Generate a response using the Groq API while maintaining chat history in simple language."""
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Add user input to chat history
    chat_history.append({"role": "user", "content": query})

    # Create conversation history
    messages = [
        {
            "role": "system",
            "content":  "You are a highly intelligent and interactive medical assistant. Use the provided context to give smart, helpful, and accurate responses." 
                        "Always be clear, concise, and conversational—like ChatGPT. "

                        "If the question is clear and answerable, respond directly using the most relevant details from the context. Include brief suggestions if helpful, such as over-the-counter remedies or next steps."

                        "If the question is vague or incomplete, ask a simple follow-up question to gather key details (e.g., symptoms, duration, age)."

                        "Use a natural tone. Do not apologize or give long disclaimers. Do not mention you're an AI. Focus on helping the user efficiently."

                        "Use short, digestible paragraphs or bullet points when appropriate. If something is potentially serious, calmly suggest seeing a doctor."


        }
    ]

    # Append chat history to maintain memory
    messages.extend(chat_history)

    # Add latest query with retrieved context
    messages.append({"role": "user", "content": f"Query: {query}\n\nContext: {context}"})

    payload = {"model": "llama3-8b-8192", "messages": messages}

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)
    
    bot_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "Error: Unexpected response from API.")

    # Add bot response to chat history
    chat_history.append({"role": "assistant", "content": bot_response})

    return bot_response

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
    app.run(debug=True)

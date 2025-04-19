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
        "content": (
            "You are a highly intelligent and interactive medical assistant. Use the provided context to give smart, helpful, and accurate responses.\n\n"
            "- Always be clear, concise, and conversational—like ChatGPT.\n"
            "- Before answering, ask 2–3 brief and relevant follow-up questions to gather key details (e.g., symptoms, duration, age, severity, medical history).\n"
            "- Once the user responds, provide a direct and informative answer using the most relevant details from the context. Include helpful suggestions when possible, such as home remedies, over-the-counter options, or next steps.\n"
            "- Use a natural tone. Don’t apologize or give long disclaimers. Don’t mention you’re an AI. Focus on helping the user efficiently.\n"
            "- Use short, digestible paragraphs or bullet points. If something sounds potentially serious, calmly suggest seeing a doctor.\n"
            "- If the question is not related to the provided medical context, simply respond with “I don’t know.” Do not search online."
        )
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
    port = int(os.environ.get("PORT", 5000))  # Render ka PORT use karo
    app.run(host='0.0.0.0', port=port)
    

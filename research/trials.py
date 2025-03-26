#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("ok")


# In[2]:


from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# In[3]:


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents


# In[5]:


extracted_data=load_pdf_file(data="C:\\Users\\vinay\\OneDrive\\Desktop\\New folder (3)\\Data")


# In[6]:


#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks


# In[7]:


text_chunks=text_split(extracted_data)
print("Length of Text Chunks", len(text_chunks))


# In[8]:


from langchain.embeddings import HuggingFaceEmbeddings


# In[17]:


#Download the Embeddings from Hugging Face
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings
embeddings = download_hugging_face_embeddings()


# In[18]:


query_result = embeddings.embed_query("Hello world")
print("Length", len(query_result))


# In[19]:


from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os

pc = Pinecone(api_key="pcsk_3xBdCP_4pWeFB6PgW2tyUg9Amoxc4RwCYLa9h3cPiiyaXcsJzeCTceUuhg5Z6aJGjQkZ7M")

index_name = "medicalbot"


pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 


# In[21]:


import os
os.environ["PINECONE_API_KEY"] = "pcsk_3xBdCP_4pWeFB6PgW2tyUg9Amoxc4RwCYLa9h3cPiiyaXcsJzeCTceUuhg5Z6aJGjQkZ7M"


# In[22]:


# Embed each chunk and upsert the embeddings into your Pinecone index.
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)


# In[23]:


docsearch


# In[24]:


# Load Existing index 

from langchain_pinecone import PineconeVectorStore
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# In[25]:


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


# In[41]:


retrieved_docs = retriever.invoke("what is acne?")


# In[34]:


retrieved_docs


# In[51]:


import requests

GROQ_API_KEY = "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"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

def generate_response(query, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise."},
            {"role": "user", "content": f"Query: {query}\n\nContext: {context}"}
        ]
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)
    
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Error: Unexpected response from API.")

# Retrieve documents only once
question = "what is acne?"
retrieved_docs = retriever.invoke(question)

answer_1 = generate_response(question, retrieved_docs)


print("\nAI Response 1:\n", answer_1)



# In[ ]:





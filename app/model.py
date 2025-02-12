import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from app.db import get_vectordb  # Use the db.py function to load existing vectordb

# Function to initialize GoogleGenerativeAI for embeddings
def get_embeddings():
    # Initialize GoogleGenerativeAIEmbeddings with API key fetched from environment
    embedding_function = GoogleGenerativeAIEmbeddings(
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )
    return embedding_function

# Function to initialize ChatGoogleGenerativeAI model for LLM
def get_model():
    # Initialize GoogleGenerativeAI model (LLM) for chat
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=0.2
    )
    return model

# Function to retrieve an answer using vectordb and model
def retrieve_answer(question, vectordb, model):
    # Create a retriever from the vectordb with k=3 to get the top 3 relevant results
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Create a RetrievalQA chain using the provided LLM (ChatGoogleGenerativeAI)
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        return_source_documents=True  # Include source documents in the response
    )

    # Query the document with the provided question
    result = qa_chain({"query": question})

    # Process and return the result
    if result:
        answer = result.get("result", "No direct answer found")
        source_docs = result.get("source_documents", [])

        # Extract relevant content from source_documents to make it JSON serializable
        sources = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in source_docs
        ]

        # Return the answer and the source documents
        return answer, sources
    else:
        return "No results found for the query", []

# Function to set up ChromaDB by loading an existing vector database
def setup_vectordb(filename):
    # Use the db.py function to load the existing ChromaDB for the given filename
    vectordb = get_vectordb(filename)
    
    if vectordb is None:
        raise ValueError(f"Vector database for {filename} not found. Please create it first.")
    
    return vectordb

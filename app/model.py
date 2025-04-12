import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from app.db import get_vectordb  # Use the db.py function to load existing vectordb

# Function to initialize GoogleGenerativeAI for embeddings
def get_embeddings():
    embedding_function = GoogleGenerativeAIEmbeddings(
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )
    return embedding_function

# Function to initialize ChatGoogleGenerativeAI model for LLM
def get_model():
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=0.2
    )
    return model

# Function to set up conversation memory with a window size of 25 turns
def setup_memory():
    return ConversationBufferWindowMemory(k=25)

# Function to process the question and decide whether to retrieve information from vectordb
def process_question(question, model, memory):
    # Step 1: LLM processes the question and checks memory for any prior context
    template = """Answer the following question based on the conversation so far. 
    If more information is needed, respond with 'retrieve'. 
    Memory: {memory}
    Question: {question}"""
    
    # Note: memory needs to be passed as part of the input
    prompt = PromptTemplate(template=template, input_variables=["memory", "question"])

    # Use Pipe (|) operator for RunnableSequence
    sequence = prompt | model

    # Run the sequence to get the initial response
    initial_response = sequence.invoke({"memory": memory.buffer, "question": question})
    
    # Extract the content of the message
    return initial_response.content

# Function to retrieve an answer using vectordb and model with memory if needed
def retrieve_answer(question, vectordb, model, memory):
    # Step 1: Process the question to check if retrieval is needed
    initial_response = process_question(question, model, memory)
    
    # If the LLM indicates that more information is needed, retrieve from vectordb
    if "retrieve" in initial_response.lower():
        # Step 2: Create a retriever from the vectordb to get the top 3 relevant results
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Step 3: Create a RetrievalQA chain using the LLM and retriever
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            output_key="result"  # Ensure this key exists in the final output
        )

        # Query the document using `invoke()` method
        result = qa_chain.invoke({"query": question})

        # Process and return the result
        if result:
            answer = result.get("result", "No direct answer found")
            source_docs = result.get("source_documents", [])

            # Extract relevant content from source_documents to make it JSON serializable
            sources = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in source_docs
            ]

            return answer, sources
        else:
            return "No results found for the query", []
    else:
        # If retrieval is not needed, return the initial response as the answer
        return initial_response, []

# Function to set up ChromaDB by loading an existing vector database
def setup_vectordb(filename):
    vectordb = get_vectordb(filename)
    
    if vectordb is None:
        raise ValueError(f"Vector database for {filename} not found. Please create it first.")
    
    return vectordb

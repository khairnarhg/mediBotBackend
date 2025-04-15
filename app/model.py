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


def get_model():
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=0.2
    )
    return model


def setup_memory():
    return ConversationBufferWindowMemory(k=25)

# Function to process the question and decide whether to retrieve information from vectordb
def process_question(question, model, memory):
    
    template = """Answer the following question based on the conversation so far. 
    If more information is needed, respond with 'retrieve'. 
    Memory: {memory}
    Question: {question}"""
    
    prompt = PromptTemplate(template=template, input_variables=["memory", "question"])

    sequence = prompt | model

    initial_response = sequence.invoke({"memory": memory.buffer, "question": question})
    
    return initial_response.content


def retrieve_answer(question, vectordb, model, memory):
    
    initial_response = process_question(question, model, memory)
    
    
    if "retrieve" in initial_response.lower():
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            output_key="result"  
        )

        result = qa_chain.invoke({"query": question})

        if result:
            answer = result.get("result", "No direct answer found")
            source_docs = result.get("source_documents", [])

            sources = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in source_docs
            ]

            return answer, sources
        else:
            return "No results found for the query", []
    else:
        return initial_response, []

def setup_vectordb(filename):
    vectordb = get_vectordb(filename)
    
    if vectordb is None:
        raise ValueError(f"Vector database for {filename} not found. Please create it first.")
    
    return vectordb

import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Function to initialize the Google Generative AI Embeddings model
def get_embeddings():
    # Initialize GoogleGenerativeAIEmbeddings with API key fetched from environment
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )
    return embedding_function

# Create vector database from texts
def create_vectordb(texts, filename):
    persist_directory = f'app/static/chroma_db/{filename}'

    # Initialize the Google Generative AI Embeddings model
    embeddings = get_embeddings()

    # Create a new Chroma instance with the texts and embeddings
    vectordb = Chroma.from_texts(
        texts,
        embeddings,
        persist_directory=persist_directory
    )

    print(f"ChromaDB collection created for {filename}.")
    return vectordb

# Load an existing vector database
def get_vectordb(filename):
    persist_directory = f'app/static/chroma_db/{filename}'

    if os.path.exists(persist_directory):
        embeddings = get_embeddings()  # Provide the embeddings while loading
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings  # Pass the embedding function
        )
        print(f"Loaded ChromaDB for {filename}")
        return vectordb
    else:
        print(f"ChromaDB not found for {filename}")
        return None

# Remove the vector database
def remove_vectordb(filename):
    persist_directory = f'app/static/chroma_db/{filename}'
    
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Deleted ChromaDB for {filename}")
    else:
        print(f"No ChromaDB found for {filename}")

# Initialize vector databases for files in a folder
def initialize_vectordbs(upload_folder):
    from app.utils import load_pdf_text, split_text

    # Iterate over all files in the upload folder
    for filename in os.listdir(upload_folder):
        if filename.endswith('.pdf'):  # Adjust if other file types are allowed
            filepath = os.path.join(upload_folder, filename)
            
            # Load PDF text and split into chunks
            pdf_text = load_pdf_text(filepath)
            texts = split_text(pdf_text)

            # Check if the vector database already exists
            vectordb = get_vectordb(filename)
            if vectordb is None:
                # Create a new vector DB for the file
                create_vectordb(texts, filename)
                print(f"Vector database created for {filename}")
            else:
                print(f"Vector database already exists for {filename}")

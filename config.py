import os

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')
    CHROMADB_PERSIST_DIR = 'db/'  # Path where your ChromaDB will persist
    UPLOAD_FOLDER = 'app/static/uploads'  # For file uploads (optional)

from flask import Flask
from flask_cors import CORS
from app.routes import configure_routes
from app.db import initialize_vectordbs  # Import the initialization function

def create_app():
    app = Flask(__name__)

    # Enable CORS for all origins
    CORS(app)

    # Upload folder for PDFs
    app.config['UPLOAD_FOLDER'] = 'app/static/uploads'

    # Register routes
    configure_routes(app)

    # Initialize vector databases for files in the upload folder
    initialize_vectordbs(app.config['UPLOAD_FOLDER'])

    return app

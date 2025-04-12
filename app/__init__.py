from flask import Flask
from flask_cors import CORS
from app.routes import configure_routes
from app.db import initialize_vectordbs

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
    configure_routes(app)
    initialize_vectordbs(app.config['UPLOAD_FOLDER'])
    return app

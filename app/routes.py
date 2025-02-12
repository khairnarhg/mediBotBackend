from flask import request, jsonify, current_app, send_from_directory
import os
from app.utils import allowed_file, save_file, load_pdf_text, split_text, delete_file_and_vectordb
from app.db import create_vectordb, get_vectordb
from app.model import get_model, retrieve_answer  # Assuming get_model() and retrieve_answer() are defined in models.py

def configure_routes(app):
    # Set the upload folder during app initialization
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            # Use the upload folder from the app config
            file_path = save_file(file, app.config['UPLOAD_FOLDER'])
            
            # Load the PDF text and split it into chunks
            pdf_text = load_pdf_text(file_path)
            chunks = split_text(pdf_text)
            
            # Create a vector database for the uploaded file
            create_vectordb(chunks, file.filename)
            
            return jsonify({"message": f"File uploaded and ChromaDB created for {file.filename}"}), 200
        else:
            return jsonify({"error": "File not allowed"}), 400

    @app.route('/files', methods=['GET'])
    def list_files():
        """List all files in the upload directory."""
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        return jsonify({"files": files}), 200

    @app.route('/files/<filename>', methods=['GET'])
    def get_file(filename):
        """Serve a file for download or preview from static/uploads."""
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/files/<filename>', methods=['DELETE'])
    def delete_file(filename):
        """Delete the file and its associated vector database."""
        success = delete_file_and_vectordb(filename, app.config['UPLOAD_FOLDER'])
        if success:
            return jsonify({"message": f"File {filename} and its ChromaDB have been deleted."}), 200
        else:
            return jsonify({"error": f"File {filename} not found."}), 404

    @app.route('/ask', methods=['POST'])
    def ask():
        question = request.json.get('question')
        filename = request.json.get('filename')  # Expect the filename to be provided in the request body

        if not question:
            return jsonify({"error": "No question provided"}), 400

        if not filename:
            return jsonify({"error": "No filename provided"}), 400

        # Load the vector database dynamically based on the file uploaded
        vectordb = get_vectordb(filename)

        if vectordb is None:
            return jsonify({"error": f"Vector database not found for {filename}"}), 404

        # Get the answer and source documents from the vector database
        answer, source_docs = retrieve_answer(question, vectordb, get_model())
        
        return jsonify({
            "answer": answer,
            "sources": source_docs
        }), 200

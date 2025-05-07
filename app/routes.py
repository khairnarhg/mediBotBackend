from flask import request, jsonify
import os
from app.utils import allowed_file, save_file, load_pdf_text, split_text, delete_file_and_vectordb
from app.db import create_vectordb, get_vectordb
from app.model import get_model, retrieve_answer
from langchain.memory import ConversationBufferWindowMemory
from flask import Flask, request, jsonify
from firebase_admin import auth, firestore
from app.firebase_config import firebasedb

blacklisted_tokens = set()

def verify_firebase_token():
    id_token = request.headers.get('Authorization')

    if not id_token:
        return None, jsonify({'error': 'Authorization token missing'}), 401
    
    if id_token in blacklisted_tokens:
        return None, jsonify({'error': 'Token has been revoked'}), 401

    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        return uid, None, None
    except Exception as e:
        return None, jsonify({'error': 'Invalid token'}), 401


# Dictionary to hold the memory for each user/session
conversation_memories = {}

def configure_routes(app):
    # Set the upload folder during app initialization
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads')

    # routes.py or app.py


    @app.route('/register', methods=['POST'])
    def register_user():
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')
        age = data.get('age')
        gender = data.get('gender')
        blood_group = data.get('bloodGroup')
        medical_history = data.get('medicalHistory')

        try:
            # Create user in Firebase Auth
            user = auth.create_user(email=email, password=password)

            # Generate a custom token
            token = auth.create_custom_token(user.uid).decode()

            # Use the renamed Firestore DB client
            firebasedb.collection('users').document(email).set({
                'uid': user.uid,
                'name': name,
                'email': email,
                'age': age,
                'gender': gender,
                'bloodGroup': blood_group,
                'medicalHistory': medical_history,
                'createdAt': firestore.SERVER_TIMESTAMP,
            })

            return jsonify({'message': 'User registered successfully', 'token': token})

        except auth.EmailAlreadyExistsError:
            return jsonify({'error': 'Email already in use'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500


        

    @app.route('/login', methods=['POST'])
    def login_user():
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        # Firebase Admin SDK doesn't support verifying password directly.
        # So you have 2 options:
        # 1. Use Firebase Authentication REST API (preferred)
        # 2. Use a custom authentication system

        import requests

        api_key = "AIzaSyBYmzVNOVWlZMwIakwBbFu6MP-ghVEnk8s"
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"

        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }

        res = requests.post(url, json=payload)
        if res.status_code == 200:
            user_data = res.json()
            return jsonify({'message': 'Login successful', 'idToken': user_data['idToken']})
        else:
            return jsonify({'error': 'Invalid credentials'}), 401


    @app.route('/logout', methods=['POST'])
    def logout():
        uid, error_response, status = verify_firebase_token()
        if error_response:
            return error_response, status

        try:
            # Blacklist current token
            id_token = request.headers.get('Authorization')
            blacklisted_tokens.add(id_token)

            # Revoke refresh tokens
            auth.revoke_refresh_tokens(uid)
            return jsonify({'message': 'User logged out successfully'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500


    @app.route('/ask', methods=['POST'])
    def ask():
        question = request.json.get('question')
        filename = request.json.get('filename')  # Filename for the vectordb
        user_id = request.json.get('user_id')  # Unique user/session ID to track memory

        if not question:
            return jsonify({"error": "No question provided"}), 400

        if not filename:
            return jsonify({"error": "No filename provided"}), 400

        if not user_id:
            return jsonify({"error": "No user ID provided"}), 400

        # Load the vector database dynamically based on the file uploaded
        vectordb = get_vectordb(filename)

        if vectordb is None:
            return jsonify({"error": f"Vector database not found for {filename}"}), 404

        # Retrieve or create conversation memory for the user/session
        if user_id not in conversation_memories:
            # Create memory with window size of 25 for this user if it doesn't exist
            conversation_memories[user_id] = ConversationBufferWindowMemory(k=25, output_key="result")
        
        # Fetch the memory for this user
        memory = conversation_memories[user_id]

        # Get the answer and source documents from the vector database using memory
        answer, source_docs = retrieve_answer(question, vectordb, get_model(), memory)
        
        # Return the answer and sources to the user
        return jsonify({
            "answer": answer,
            "sources": source_docs
        }), 200
    
    @app.route('/profile/get', methods=['GET'])
    def get_profile():
        uid, error_response, status = verify_firebase_token()
        if error_response:
            return error_response, status

        try:
            # Fetch user by UID (or email, depending on how you store)
            users_ref = firebasedb.collection('users')
            user_docs = users_ref.where('uid', '==', uid).stream()
            user_data = None
            for doc in user_docs:
                user_data = doc.to_dict()
                break

            if not user_data:
                return jsonify({'success': False, 'message': 'User not found'}), 404

            return jsonify({'success': True, 'profile': user_data}), 200

        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500

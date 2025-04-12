import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase only once
cred = credentials.Certificate("./app/firebase_service_account.json")
firebase_admin.initialize_app(cred)

# Export Firestore DB
firebasedb = firestore.client()

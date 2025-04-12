# firebase_auth.py
import firebase_admin
from firebase_admin import auth, credentials

cred = credentials.Certificate("firebase_service_account.json")
firebase_admin.initialize_app(cred)

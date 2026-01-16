import os
import sys
import json
import socket
import pickle
import re
import requests
import dotenv
import numpy as np
import pandas as pd
from textblob import TextBlob
from scipy.sparse import hstack
from openai import OpenAI
from django.conf import settings
from django.core.management import execute_from_command_line
from django.http import JsonResponse
from django.urls import path
from django.views.decorators.csrf import csrf_exempt

# ---------------------------------------------------------
# SCAM SHIELD SERVER (FINAL: SMS + CALL + LIVE AUDIO)
# Uses Advanced Ensemble Model (LR + RF) for Offline Fallback
# ---------------------------------------------------------

# --- CONFIGURATION (SECURE) ---
dotenv.load_dotenv() 

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    print("❌ ERROR: 'GROQ_API_KEY' not found in environment variables!")
else:
    print("✅ Groq API Key loaded successfully.")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='django-insecure-hackathon-key',
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=['*'], 
    )

# --- ADVANCED MODEL LOADING & HELPERS ---

lr_model = None
rf_model = None
tfidf_vectorizer = None
simple_model = None # Fallback for simple model

def load_advanced_models():
    global lr_model, rf_model, tfidf_vectorizer, simple_model
    try:
        # Try loading advanced models first
        if os.path.exists('scam_detection_lr.pkl'):
            with open('scam_detection_lr.pkl', 'rb') as f: lr_model = pickle.load(f)
        if os.path.exists('scam_detection_rf.pkl'):
            with open('scam_detection_rf.pkl', 'rb') as f: rf_model = pickle.load(f)
        if os.path.exists('tfidf_vectorizer.pkl'):
            with open('tfidf_vectorizer.pkl', 'rb') as f: tfidf_vectorizer = pickle.load(f)
        
        if lr_model and rf_model and tfidf_vectorizer:
            print("✅ Advanced Offline Models Loaded Successfully!")
            return True
            
        # Fallback to simple model if advanced ones are missing
        if os.path.exists('scam_model.pkl'):
             with open('scam_model.pkl', 'rb') as f: simple_model = pickle.load(f)
             print("⚠️ Advanced models missing. Loaded simple 'scam_model.pkl' instead.")
             return True

    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
    return False

# Feature Extraction Helper (Must match training logic)
def get_text_metrics(text):
    blob = TextBlob(str(text))
    words = str(text).split()
    return [
        blob.sentiment.polarity,
        blob.sentiment.subjectivity,
        len(words),
        np.mean([len(w) for w in words]) if len(words) > 0 else 0
    ]

# Hybrid Prediction Logic (Offline)
def predict_offline(message):
    # Case 1: Advanced Models Available
    if lr_model and rf_model and tfidf_vectorizer:
        try:
            # 1. Extract Metrics
            metrics = get_text_metrics(message)
            # 2. Vectorize
            text_vector = tfidf_vectorizer.transform([message])
            # 3. Combine
            features = hstack([text_vector, [metrics]])
            
            # 4. Predict
            lr_prob = lr_model.predict_proba(features)[0][1]
            rf_prob = rf_model.predict_proba(features)[0][1]
            final_prob = (lr_prob + rf_prob) / 2
            
            # Threshold logic (FIX: Ensure standard Python boolean)
            is_scam = bool(final_prob > 0.45)
            reason = f"Offline Analysis Risk: {final_prob*100:.1f}%"
            return is_scam, reason
        except Exception as e:
            print(f"Advanced Prediction Error: {e}")
            # Fall through to simple model if advanced fails

    # Case 2: Simple Model Available (Backup)
    if simple_model:
        try:
            pred = simple_model.predict([message])[0]
            # Simple model might not have predict_proba if not trained with it
            # Assuming simple model returns 0 or 1
            return bool(pred == 1), "Offline Basic Check"
        except Exception as e:
             print(f"Simple Prediction Error: {e}")

    return None, "Models not loaded or prediction failed"

# Load models on startup
load_advanced_models()

# --- SERVER HELPERS ---
def cors_response(data, status=200):
    response = JsonResponse(data, status=status)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Content-Type"
    return response

def extract_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group(0))
        return None
    except: return None

def get_all_ips():
    ip_list = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_list.append(s.getsockname()[0])
        s.close()
    except: pass
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        if host_ip not in ip_list and not host_ip.startswith("127."):
            ip_list.append(host_ip)
    except: pass
    return ip_list

# --- VIEWS (ENDPOINTS) ---

# 1. CHECK NUMBER (Jab Call Aaye)
@csrf_exempt
def check_number(request):
    if request.method == 'OPTIONS': return cors_response({})
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            phone = data.get('phone', '')
            print(f"📞 Incoming Call Check: {phone}")
            
            spam_db = ['+919876543210', '1400', '+1234567890']
            is_spam = phone in spam_db
            
            return cors_response({
                'show_popup': True, 
                'is_known_spam': is_spam,
                'message': "⚠️ Suspected Spam" if is_spam else "Unknown Caller. Activate AI Shield?"
            })
        except: return cors_response({'error': 'Error'}, 400)
    return cors_response({})

# 2. ANALYZE CALL (Jab Baat Ho Rahi Ho)
@csrf_exempt
def analyze_call(request):
    if request.method == 'OPTIONS': return cors_response({})
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            transcript = data.get('transcript', '')
            mode = data.get('mode', 'online') # DEFAULT: Online Mode

            if not transcript: return cors_response({'action': 'none'})

            print(f"🗣️ Live Audio ({mode}): \"{transcript}\"")
            
            # A. Keyword Check (Runs in all modes as a fast pre-filter)
            danger_words = ['otp', 'cvv', 'card number', 'expiry', 'police', 'arrest', 'cbi', 'drugs']
            for w in danger_words:
                if w in transcript.lower():
                    print(f"   🚨 Trigger Word: {w}")
                    return cors_response({
                        'action': 'vibrate_strong', 
                        'risk_score': 99,
                        'reason': f"Scammer said '{w}'"
                    })
            
            # B. AI Context Check
            # --- OFFLINE MODE ---
            if mode == 'offline':
                 # Currently offline model is text-only trained, might not be best for transcript context
                 # but we can try it if user insists.
                 # For call analysis, keyword check is the main offline defense.
                 return cors_response({'action': 'none', 'risk_score': 0})

            # --- ONLINE / AUTO MODE ---
            try:
                if not API_KEY: raise Exception("Key Missing")
                
                prompt = f"Analyze call snippet: '{transcript}'. Is user being threatened or asked for money? JSON: {{'risk': 'high/low', 'reason': 'short text'}}"
                
                resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":prompt}], temperature=0.1)
                res = extract_json(resp.choices[0].message.content)
                
                if res and res.get('risk') == 'high':
                    return cors_response({'action': 'vibrate_strong', 'risk_score': 90, 'reason': res.get('reason')})
            except Exception as e:
                # If mode is STRICTLY online, return error
                if mode == 'online':
                    return cors_response({'error': 'Online AI Failed', 'details': str(e)}, 503)
                pass # In auto mode, just fail silently to safe
            
            return cors_response({'action': 'none', 'risk_score': 0})
        except: return cors_response({'error': 'Error'}, 400)
    return cors_response({})

# 3. PREDICT SCAM (SMS Check)
@csrf_exempt
def predict_scam(request):
    if request.method == 'OPTIONS': return cors_response({})
    
    if request.method == 'POST':
        try:
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                return cors_response({'error': 'Invalid JSON'}, 400)
            
            message = data.get('message', '')
            mode = data.get('mode', 'auto') # Default to 'auto'

            print(f"📩 Analyzing SMS (Mode: {mode})...")
            
            # --- OFFLINE MODE CHECK ---
            if mode == 'offline':
                print(f"   👉 Using Advanced Offline Model (Forced)...")
                is_scam, reason = predict_offline(message)
                if is_scam is not None:
                    return cors_response({'is_scam': is_scam, 'reason': reason})
                else:
                    return cors_response({'error': 'Offline Analysis Failed', 'details': reason}, 503)

            # --- ONLINE MODE (Default) ---
            try:
                # 1. Try Online AI
                if not API_KEY: raise Exception("Key Missing")
                prompt = f"Analyze SMS: '{message}'. Return JSON: {{'is_scam': true/false, 'reason': 'Hinglish reason'}}"
                resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":prompt}], temperature=0.1)
                res = extract_json(resp.choices[0].message.content)
                if res: return cors_response(res)
            except Exception as e:
                print(f"   ⚠️ Online AI Failed: {e}")
                
                # If mode is 'auto' (implicit fallback) or just failed, we CAN fallback 
                # BUT user requested 'online' or 'auto'. 
                # If strictly 'online', we should fail. 
                # But for UX, let's treat default 'online' as 'prefer online, fallback if fail'?
                # Actually, standard toggle logic usually implies strict mode.
                # Let's support 'auto' as a distinct mode for fallback.
                
                if mode == 'online':
                     # Strict Online failure
                     return cors_response({'error': 'Online AI Failed', 'details': str(e)}, 503)

                # If mode was 'auto' (or we decide to fallback anyway for resilience), proceed to offline
                print(f"   👉 Switching to Advanced Offline Model (Fallback)...")
                is_scam, reason = predict_offline(message)
                
                if is_scam is not None:
                    return cors_response({'is_scam': is_scam, 'reason': reason})
                else:
                    return cors_response({'error': 'Offline Analysis Failed', 'details': reason}, 503)

        except Exception as e:
            print(f"❌ Server Error: {e}")
            return cors_response({'error': 'Server Error', 'details': str(e)}, 400)
    return cors_response({})

def home(request): 
    return cors_response({
        'status': 'Online 🟢', 
        'endpoints': ['/predict', '/check_number', '/analyze_call']
    })

urlpatterns = [path('', home), path('predict', predict_scam), path('check_number', check_number), path('analyze_call', analyze_call)]

if __name__ == "__main__":
    ips = get_all_ips()
    port = '8000'
    print(f"\n🚀 SCAM SHIELD SERVER READY (Advanced Hybrid Mode)")
    print(f"👇 Server IPs:")
    if not ips: print(f"   1. 127.0.0.1:{port}")
    else:
        for i, ip in enumerate(ips): print(f"   {i+1}. {ip}:{port}")
    print("\n")
    
    if 'runserver' not in sys.argv: sys.argv += ['runserver', f'0.0.0.0:{port}']
    elif 'runserver' in sys.argv:
         found = False
         for arg in sys.argv:
             if '0.0.0.0' in arg: found = True
         if not found: sys.argv.append(f'0.0.0.0:{port}')
    
    execute_from_command_line(sys.argv)
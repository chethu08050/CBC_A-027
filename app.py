from flask import Flask, render_template, request, jsonify
import requests
from langdetect import detect
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import os
import re

app = Flask(__name__)

API_KEY = "AIzaSyAHcIvz7GCEb_lhY7mgRSPsSt3gD6hzg1M"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

# Store chat history
chat_history = {}

# Language code map
LANGUAGE_MAP = {
    "hi": "hi-IN", 
    "kn": "kn-IN",
    "en": "en-IN",
    "ml": "ml-IN",
    "ta": "ta-IN",
    "te": "te-IN"
}

INSTRUCTIONS = {
    "hi-IN": "Please respond in Hindi only. Keep answers short and direct.",
    "kn-IN": "Please respond in Kannada only. Keep answers short and direct.", 
    "en-IN": "Please respond in English only. Keep answers short and direct.",
    "ml-IN": "Please respond in Malayalam only. Keep answers short and direct.",
    "ta-IN": "Please respond in Tamil only. Keep answers short and direct.",
    "te-IN": "Please respond in Telugu only. Keep answers short and direct."
}

# Load data files
def load_data():
    try:
        price_data = None
        schemes_data = None
        govt_schemes_data = None
        
        if os.path.exists('karnataka_veg_prices_3days.csv'):
            price_data = pd.read_csv('karnataka_veg_prices_3days.csv')
            
        if os.path.exists('scheme.csv'):
            schemes_data = pd.read_csv('scheme.csv')
            
        if os.path.exists('digital_government_schemes_with_dates.csv'):
            govt_schemes_data = pd.read_csv('digital_government_schemes_with_dates.csv')
            
        return price_data, schemes_data, govt_schemes_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

# Global variables for data
price_data, schemes_data, govt_schemes_data = load_data()

# Function to check if query is about prices
def is_price_query(query):
    price_keywords = ['price', 'cost', 'rate', 'how much', 'rupees', 'rs', '₹', 'market rate', 'selling for']
    return any(keyword in query.lower() for keyword in price_keywords)

# Function to extract vegetable name from query  
def extract_vegetable(query):
    if price_data is None:
        return None
        
    # Get unique vegetable names from the dataset
    vegetable_list = price_data['vegetable_name'].unique()
    
    # Check for vegetable names in the query
    for vegetable in vegetable_list:
        if vegetable.lower() in query.lower():
            return vegetable
    
    return None

# Function to extract market name if present
def extract_market(query):
    if price_data is None:
        return None
        
    market_list = price_data['market_name'].unique()
    
    for market in market_list:
        if market.lower() in query.lower():
            return market
    
    return None

# Function to predict price using Random Forest
def predict_price(vegetable, market=None, target_date=None):
    if price_data is None:
        return None, None
    
    # Filter data for the specific vegetable
    veg_data = price_data[price_data['vegetable_name'] == vegetable].copy()
    
    if veg_data.empty:
        return None, None
    
    # If market is specified, filter by market as well
    if market:
        veg_data = veg_data[veg_data['market_name'] == market]
        if veg_data.empty:
            return None, None
    
    # Convert date strings to datetime objects
    veg_data['date'] = pd.to_datetime(veg_data['date'])
    
    # Convert dates to numerical features (days since some reference date)
    reference_date = veg_data['date'].min()
    veg_data['days_since_ref'] = (veg_data['date'] - reference_date).dt.days
    
    # Prepare features (X) and target variables (y_min, y_max)
    # Use one-hot encoding for market_name if no specific market
    if not market:
        X = pd.get_dummies(veg_data[['days_since_ref', 'market_name']])
    else:
        X = veg_data[['days_since_ref']]
    
    y_min = veg_data['min_price']
    y_max = veg_data['max_price']
    
    # Train Random Forest models
    rf_min = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_max = RandomForestRegressor(n_estimators=100, random_state=42)
    
    rf_min.fit(X, y_min)
    rf_max.fit(X, y_max)
    
    # Prepare prediction data
    if target_date is None:
        target_date = datetime.now()
    
    days_since_ref = (target_date - reference_date).days
    
    # Create prediction feature set
    if not market:
        # Create empty DataFrame with same columns as X
        pred_X = pd.DataFrame(columns=X.columns)
        pred_X.loc[0, 'days_since_ref'] = days_since_ref
        
        # Set all market columns to 0
        for col in pred_X.columns:
            if col != 'days_since_ref':
                pred_X.loc[0, col] = 0
        
        # Select the most common market for prediction
        most_common_market = veg_data['market_name'].mode()[0]
        market_col = f"market_name_{most_common_market}"
        if market_col in pred_X.columns:
            pred_X.loc[0, market_col] = 1
    else:
        pred_X = pd.DataFrame({'days_since_ref': [days_since_ref]})
    
    # Make predictions
    try:
        min_price_pred = max(0, round(rf_min.predict(pred_X)[0], 2))
        max_price_pred = max(min_price_pred, round(rf_max.predict(pred_X)[0], 2))
        
        return min_price_pred, max_price_pred
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

# Format price response based on the language
def format_price_response(vegetable, min_price, max_price, market, lang):
    market_info = f" in {market}" if market else ""
    
    responses = {
        "en-IN": f"The current price of {vegetable}{market_info} is between ₹{min_price} and ₹{max_price} per kg.",
        "hi-IN": f"{vegetable} का वर्तमान मूल्य{market_info} ₹{min_price} से ₹{max_price} प्रति किलो के बीच है।",
        "kn-IN": f"{vegetable} ಯ ಪ್ರಸ್ತುತ ಬೆಲೆ{market_info} ಕಿಲೋಗ್ರಾಮ್‌ಗೆ ₹{min_price} ಮತ್ತು ₹{max_price} ನಡುವೆ ಇದೆ.",
        "ml-IN": f"{vegetable} യുടെ നിലവിലെ വില{market_info} കിലോയ്ക്ക് ₹{min_price} മുതൽ ₹{max_price} വരെയാണ്.",
        "ta-IN": f"{vegetable} இன் தற்போதைய விலை{market_info} கிலோவுக்கு ₹{min_price} முதல் ₹{max_price} வரை உள்ளது.",
        "te-IN": f"{vegetable} యొక్క ప్రస్తుత ధర{market_info} కిలోకు ₹{min_price} మరియు ₹{max_price} మధ్య ఉంది."
    }
    
    return responses.get(lang, responses["en-IN"])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    lang = data.get("lang", "auto")
    session_id = data.get("session_id", "default")
    
    # Initialize chat history for new sessions
    if session_id not in chat_history:
        chat_history[session_id] = []
    
    # Auto language detection
    if lang == "auto" or lang not in INSTRUCTIONS:
        try:
            detected_lang = detect(user_input)
            lang = LANGUAGE_MAP.get(detected_lang, "en-IN")
        except Exception:
            lang = "en-IN"

    # Add user message to chat history
    chat_history[session_id].append({"role": "user", "content": user_input})
    
    # Create context from chat history
    context = "\n".join([f"{'Bot' if msg['role']=='assistant' else 'User'}: {msg['content']}" 
                        for msg in chat_history[session_id][-5:]])  # Keep last 5 messages for context

    # First try Gemini API
    instruction = INSTRUCTIONS.get(lang, INSTRUCTIONS["en-IN"])
    prompt = f'{instruction} Consider this chat history for context:\n{context}\n\nIf you don\'t know the answer, predict logically. If the user\'s message is unclear, ask for clarification: "{user_input}"'
    
    try:
        response = requests.post(GEMINI_ENDPOINT, json={
            "contents": [
{
  "parts": [{
    "text": f"If it is a question, provide a more specific and exact answer. If you don't know the exact value, then provide a random but reasonable value that makes sense remember that we are indian and we are talking about indian prices and indian currency. If the query is unclear, ask follow-up questions. else answer normaly.\n\n{prompt}"
  }]
}
            ]
        })
        result = response.json()
        reply = result["candidates"][0]["content"]["parts"][0]["text"]
        
        # Add bot response to chat history
        chat_history[session_id].append({"role": "assistant", "content": reply})
        
        return jsonify({"reply": reply, "detected_lang": lang})
    except Exception:
        # If Gemini fails, check if it's a price query
        if is_price_query(user_input):
            vegetable = extract_vegetable(user_input)
            market = extract_market(user_input)
            
            if vegetable:
                min_price, max_price = predict_price(vegetable, market)
                
                if min_price is not None and max_price is not None:
                    reply = format_price_response(vegetable, min_price, max_price, market, lang)
                    
                    # Add bot response to chat history
                    chat_history[session_id].append({"role": "assistant", "content": reply})
                    
                    return jsonify({"reply": reply, "detected_lang": lang})

        # If both fail, return error message
        error_msg = {
            "en-IN": "Error: Unable to get response.",
            "hi-IN": "त्रुटि: प्रतिक्रिया प्राप्त करने में असमर्थ।",
            "kn-IN": "ದೋಷ: ಪ್ರತಿಕ್ರಿಯೆಯನ್ನು ಪಡೆಯಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.",
            "ml-IN": "പിശക്: പ്രതികരണം ലഭിക്കാൻ കഴിഞ്ഞില്ല.",
            "ta-IN": "பிழை: பதிலைப் பெற முடியவில்லை.",
            "te-IN": "లోపం: ప్రతిస్పందనను పొందడం సాధ్యం కాలేదు."
        }
        error_response = error_msg.get(lang, error_msg["en-IN"])
        
        # Add error response to chat history
        chat_history[session_id].append({"role": "assistant", "content": error_response})
        
        return jsonify({"reply": error_response, "detected_lang": lang})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
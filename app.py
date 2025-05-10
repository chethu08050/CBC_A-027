from flask import Flask, render_template, request, jsonify
import requests
from langdetect import detect

app = Flask(__name__)

API_KEY = "AIzaSyAHcIvz7GCEb_lhY7mgRSPsSt3gD6hzg1M"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    lang = data.get("lang", "auto")

    # Auto language detection
    if lang == "auto" or lang not in INSTRUCTIONS:
        try:
            detected_lang = detect(user_input)
            lang = LANGUAGE_MAP.get(detected_lang, "en-IN")
        except Exception:
            lang = "en-IN"

    # Prepare prompt
    instruction = INSTRUCTIONS.get(lang, INSTRUCTIONS["en-IN"])
    prompt = f'{instruction} If you don’t know the answer, predict logically: "{user_input}"'

    try:
        response = requests.post(GEMINI_ENDPOINT, json={
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        })

        result = response.json()
        reply = result["candidates"][0]["content"]["parts"][0]["text"]
        return jsonify({"reply": reply, "detected_lang": lang})
    except Exception as e:
        return jsonify({"reply": "Error: Unable to get response from Gemini."})

if __name__ == "__main__":
    app.run(debug=True)
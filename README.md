# JanMitra AI Voice Assistant

A multilingual voice-enabled AI assistant built with Flask and Google's Gemini API, supporting multiple Indian languages and featuring a modern, animated UI.

## 🌟 Features

- **Multilingual Support**: Communicate in multiple Indian languages:
  - English (en-IN)
  - Hindi (hi-IN)
  - Kannada (kn-IN)
  - Malayalam (ml-IN)
  - Tamil (ta-IN)
  - Telugu (te-IN)
  - Auto language detection

- **Voice Interaction**:
  - Text-to-Speech output
  - Speech-to-Text input
  - Continuous voice recognition mode
  - Auto-read responses option

- **Smart Features**:
  - Vegetable price prediction
  - Government scheme information
  - Natural language processing
  - Context-aware responses

- **Modern UI/UX**:
  - Responsive design
  - Dark theme
  - Smooth animations
  - Real-time typing indicators
  - Offline mode detection
  - Ambient background effects

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Flask
- Google Cloud API key for Gemini
- Modern web browser with Web Speech API support

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gemini_voice_assistant_flask.git
cd gemini_voice_assistant_flask
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
export API_KEY="your_gemini_api_key"
```

4. Run the Flask application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## 💻 Usage

### Text Input
- Type your message in the input field
- Press Enter or click the Send button

### Voice Input
- Click the microphone button (🎤) to start voice input
- Hold Shift + Click for continuous voice recognition mode
- Click again to stop recording

### Language Selection
- Use the language buttons at the top to switch between languages
- Select "Auto Detect" for automatic language detection

### Voice Output
- Toggle "Auto" button to enable/disable automatic voice responses
- Click the speaker icon (🔊) on any message to hear it again
- Use the stop button (🔇) to stop current speech

## 🔧 Technical Details

### Backend
- Flask web framework
- Google Gemini API integration
- Random Forest model for price prediction
- Language detection using langdetect
- Speech synthesis and recognition

### Frontend
- Modern HTML5/CSS3
- Tailwind CSS for styling
- GSAP for animations
- Web Speech API for voice features
- Responsive design

## 📝 Data Files

The application uses several data files:
- `karnataka_veg_prices_3days.csv`: Vegetable price data
- `scheme.csv`: Government scheme information
- `digital_government_schemes_with_dates.csv`: Digital government schemes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API
- Flask Framework
- Web Speech API
- GSAP Animation Library
- Tailwind CSS 
# 🎧 AI Audio Pipeline

An automated end-to-end pipeline that **transcribes audio**, **redacts personally identifiable information (PII)**, **summarizes text**, and **generates an audio summary** — powered by **Google Cloud Speech-to-Text**, **spaCy**, and **Google Cloud Text-to-Speech**.

---

## 🧠 Overview

The **Audio AI Pipeline** processes raw audio files through multiple intelligent stages:

1. **Speech-to-Text (STT):** Converts audio to text using Google Cloud Speech API.  
2. **PII Redaction:** Detects and removes sensitive information (emails, phone numbers, credit cards, etc.) using:
   - Regular expressions (Regex)
   - Named Entity Recognition (NER) via spaCy  
3. **Summarization:** Compresses the transcribed text into a shorter summary.  
4. **Text-to-Speech (TTS):** Generates a spoken summary in natural-sounding audio.  

---

## 🗂️ Project Structure

AUDIO AI PIPELINE/\
│\
├── audio_samples/ # Input audio files\
├── google-cloud-sdk/ # Optional: Google SDK setup files\
├── scripts/ # Core Python scripts\
│ ├── 3_pii_redaction.py # Transcribe + PII redaction\
│ └── 4_tts_summary.py # Transcribe + summarize + TTS\
│ └── 1_basic_tt.py # Basic text-to-speech example\
│ └── 2_confidence_scoring.py # Confidence scoring \
├── .env # API credentials and environment variables\
├── requirements.txt # Dependencies\
├── output_summary.mp3 # Generated summary audio (output)\
├── output_transcript.txt # Transcribed and redacted text\
├── raw_transcript.txt # Raw transcription (before redaction)\
├── audit.log # Log file for debugging\
├── README.md # Project documentation \
└── REFLECTION.md # Development notes

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/xranano/AI-audio-pipeline.git 
cd AI-audio-pipeline
```
### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure environment
Create a .env file and include your Google Cloud credentials:
```bash
GOOGLE_APPLICATION_CREDENTIALS="path/to/your-service-account.json"
Make sure the Google Cloud SDK is installed and authenticated:
gcloud auth activate-service-account --key-file="path/to/your-service-account.json"
```

## 🧩 Usage
### ▶️ 1. Transcribe and Redact PII
```bash
python scripts/3_pii_redaction.py audio_samples/sample.mp3 
```
**Output:**

Console printout of original and redacted transcript\
Redacted text stored in output_transcript.txt

### ▶️ 2. Summarize and Generate Audio
```bash
python scripts/4_tts_summary.py audio_samples/sample.mp3
```
**Output:**\

Transcript and summary in console\
Summary audio file saved as output_summary.mp3\

## 🧰 Technologies Used
- Python 3.9+\
- Google Cloud Speech-to-Text\
- Google Cloud Text-to-Speech
- spaCy (en_core_web_sm)
- Regex for pattern-based redaction


## 🧪 Example
### Input audio:
**“Hi, I’m John Smith. My phone number is 555-123-4567. I live in Boston.”**

### Output text (after PII redaction):
“Hi, I’m [REDACTED_PERSON]. My phone number is [REDACTED_PHONE]. I live in Boston.”\

### Generated audio:
output_summary.mp3 — concise spoken summary of the conversation.\


## 📈 Future Improvements
- Advanced summarization (TextRank or Transformer models)\
- Multi-language support\
- Web UI or Streamlit dashboard\
- Database logging for analytics\

# 👩‍💻 Author
Anano Kutchava
📍 Kutaisi International University\
💻 GitHub: xranano\

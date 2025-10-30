import sys
import os
import json
import re
import spacy
import numpy as np
import librosa
from google.cloud import speech, texttospeech

# ------------------- Load spaCy NER model -------------------
nlp = spacy.load("en_core_web_sm")

# ------------------- Utilities -------------------
def guess_encoding(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".mp3":
        return speech.RecognitionConfig.AudioEncoding.MP3
    if ext in {".wav", ".pcm", ".raw"}:
        return speech.RecognitionConfig.AudioEncoding.LINEAR16
    return speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED

def transcribe_audio(audio_path):
    client = speech.SpeechClient()
    with open(audio_path, "rb") as f:
        content = f.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=guess_encoding(audio_path),
        language_code="en-US",
        enable_word_confidence=True,
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
    )
    response = client.recognize(config=config, audio=audio)
    transcript = response.results[0].alternatives[0].transcript
    words = response.results[0].alternatives[0].words
    return transcript, words

def calculate_snr(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    signal_power = np.mean(y ** 2)
    noise_power = np.var(y)
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")

def calculate_word_perplexity(words):
    confidences = [w.confidence for w in words]
    avg_conf = np.mean(confidences)
    return 1.0 / avg_conf if avg_conf > 0 else float("inf")

def multi_factor_confidence(audio_path, transcript, words):
    snr = calculate_snr(audio_path)
    perplexity = calculate_word_perplexity(words)
    api_conf = np.mean([w.confidence for w in words])
    snr_norm = min(max((snr - 10) / 20, 0), 1)
    perplexity_norm = max(1 - (perplexity - 1), 0)
    combined = 0.5*api_conf + 0.3*snr_norm + 0.2*perplexity_norm
    if combined > 0.85: level = "HIGH"
    elif combined > 0.7: level = "MEDIUM"
    else: level = "LOW"
    return combined, level

# ------------------- PII Redaction -------------------
def redact_pii_regex(text):
    patterns = {
        'CREDIT_CARD': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        'SSN': r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        'PHONE': r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    }
    redacted = text
    redactions = []
    for t, p in patterns.items():
        for m in re.finditer(p, text):
            original = m.group()
            redacted = redacted.replace(original, f"[REDACTED_{t}]")
            redactions.append({'type': t, 'original': original, 'position': m.span()})
    return redacted, redactions

def redact_pii_ner(text):
    doc = nlp(text)
    redacted = text
    redactions = []
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        if ent.label_ in ['PERSON', 'DATE']:
            redacted = redacted[:ent.start_char] + f"[REDACTED_{ent.label_}]" + redacted[ent.end_char:]
            redactions.append({'type': ent.label_, 'original': ent.text, 'position': (ent.start_char, ent.end_char)})
    return redacted, redactions

def redact_pii(text):
    redacted_regex, regex_list = redact_pii_regex(text)
    redacted_final, ner_list = redact_pii_ner(redacted_regex)
    return redacted_final, regex_list + ner_list

# ------------------- Summarization + TTS -------------------
def summarize_text(text, max_sentences=3):
    sentences = text.split('. ')
    summary = '. '.join(sentences[:max_sentences])
    if not summary.endswith('.'):
        summary += '.'
    return summary

def text_to_speech(text, output_file="output_summary.mp3"):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Neural2-A")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
    print(f"Audio summary saved as: {output_file}")
    return output_file

# ------------------- Full Pipeline -------------------
def run_pipeline(audio_path):
    print(f"\n=== Processing audio: {audio_path} ===")

    # 1️⃣ Transcription
    print("\nStep 1: Transcribing audio...")
    transcript, words = transcribe_audio(audio_path)
    with open("raw_transcript.txt", "w") as f:
        f.write(transcript)
    print(f"Transcript saved to raw_transcript.txt\n{transcript}")

    # 2️⃣ Confidence Scoring
    print("\nStep 2: Calculating multi-factor confidence...")
    score, level = multi_factor_confidence(audio_path, transcript, words)
    print(f"Combined Confidence Score: {score:.3f}, Level: {level}")

    # 3️⃣ PII Redaction
    print("\nStep 3: Redacting PII...")
    redacted_text, redactions = redact_pii(transcript)
    with open("output_transcript.txt", "w") as f:
        f.write(redacted_text)
    print(f"Redacted transcript saved to output_transcript.txt")
    print(f"Total items redacted: {len(redactions)}")

    # 4️⃣ Summarization
    print("\nStep 4: Summarizing text...")
    summary = summarize_text(redacted_text, max_sentences=2)
    print(f"Summary:\n{summary}")

    # 5️⃣ TTS
    print("\nStep 5: Generating audio summary...")
    summary_audio = text_to_speech(summary, "output_summary.mp3")

    # 6️⃣ Logging
    log_data = {
        "audio_file": audio_path,
        "transcript_file": "raw_transcript.txt",
        "redacted_transcript_file": "output_transcript.txt",
        "summary_audio_file": summary_audio,
        "confidence_score": score,
        "confidence_level": level,
        "redactions": redactions,
        "summary_text": summary
    }
    with open("audit.log", "w") as f:
        json.dump(log_data, f, indent=2)
    print("\nAudit log saved as audit.log")

    print("\n=== Pipeline Complete ===")
    return log_data

# ------------------- Main -------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_pipeline.py <audio_file>")
        exit(1)

    audio_file = sys.argv[1]
    run_pipeline(audio_file)
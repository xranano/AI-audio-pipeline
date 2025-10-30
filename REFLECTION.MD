I built an end-to-end audio pipeline that transcribes speech, scores confidence, redacts PII, summarizes, and generates an audio summary. I started by recording my test audio, which included my name, a short paragraph about AI, and a sample credit card number, with mild background noise to simulate realistic conditions.

The speech-to-text step highlighted the strengths and limitations of automated transcription. While most words were captured accurately, background noise and rapid speech reduced confidence in certain segments. Using a multi-factor confidence measure—combining API confidence, signal-to-noise ratio, and a perplexity-based metric—helped identify portions needing manual review, emphasizing the importance of human oversight in sensitive applications.

PII redaction was performed using both regex patterns and Named Entity Recognition, effectively masking credit card numbers, names, and dates. This hybrid approach provided better coverage than either method alone.

Finally, the summarization and Text-to-Speech steps produced a concise textual and audio summary of the transcript. This reinforced the need for clear phrasing and proper punctuation for natural-sounding speech.

Overall, this lab strengthened my understanding of speech processing, privacy protection, and automated summarization, while demonstrating how to integrate multiple AI components into a reliable workflow. Careful testing and analysis proved essential to ensure accuracy and privacy.
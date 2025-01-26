from langdetect import detect

def detect_language(text: str) -> str:
    lang_code = detect(text)
    if lang_code not in ["ar", "fr", "en"]:
        raise ValueError(f"Unsupported language: {lang_code}")
    return lang_code
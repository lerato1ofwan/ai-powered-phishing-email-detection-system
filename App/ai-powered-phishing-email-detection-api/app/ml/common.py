import re

# Text cleaning function, makes everything lowercase, removed non alpha-numeric characters and normalize white spaces
def simple_text_clean(text: str) -> str:
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text) # Keep spaces, remove other non-alphanumeric
        text = re.sub(r'\s+', ' ', text).strip()
    else:
        text = '' 
    return text

# Class names for predictions 
CLASS_NAMES = ['Legitimate', 'Phishing'] # 0: Legitimate, 1: Phishing

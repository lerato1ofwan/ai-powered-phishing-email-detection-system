import joblib
import pandas as pd
import re
from lime.lime_text import LimeTextExplainer
import numpy as np
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configure and setup model and preprocessor files
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
PREPROCESSOR_FILENAME = "email_preprocessor_20250506_203148.joblib" 
MODEL_FILENAME = "phishing_nb_model_20250506_203148.joblib"
PREPROCESSOR_PATH = os.path.join(ASSETS_DIR, PREPROCESSOR_FILENAME)
MODEL_PATH = os.path.join(ASSETS_DIR, MODEL_FILENAME)

# Load model and preprocessor
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    print("ML Model and Preprocessor loaded successfully from ml_logic.")
except FileNotFoundError:
    print(f"FATAL ERROR: Could not find model ('{MODEL_PATH}') or preprocessor ('{PREPROCESSOR_PATH}').")
    print("Ensure files are in 'app/assets/' and filenames are correct in ml_logic.py.")
    preprocessor = None
    model = None
except Exception as e:
    print(f"Error loading ML model/preprocessor: {e}")
    preprocessor = None
    model = None

# --- Load BERT-mini model and tokenizer from Hugging Face Hub ---
# Replace with your actual Hugging Face model ID
BERT_MODEL_ID = "lleratodev/720-bert-mini-phishing" # e.g., "LeratoLetsepe/phishing-bert-mini"
try:
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_ID)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_ID)
    bert_model.eval() # Set model to evaluation mode
    print(f"BERT-mini model ('{BERT_MODEL_ID}') and tokenizer loaded successfully from Hugging Face Hub.")
    # Determine device for BERT model (CPU by default, can be adapted for GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    print(f"BERT model moved to device: {device}")

except Exception as e:
    print(f"FATAL ERROR (BERT): Could not load model/tokenizer '{BERT_MODEL_ID}' from Hugging Face Hub: {e}")
    print("Ensure the model ID is correct, you have an internet connection, and the model files are correctly set up on the Hub.")
    bert_tokenizer = None
    bert_model = None
# --- End BERT Loading ---

# Text cleaning function, makes everything lowercase, removed non alpha-numeric characters and normalize white spaces
def simple_text_clean(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
    else:
        text = ''
    return text

# For explanability, LIME setup - # LIME probability function for MultinomialNB model
class_names = ['Legitimate', 'Phishing'] # 0: Legitimate, 1: Phishing
explainer = LimeTextExplainer(class_names=class_names)

def model_predict_probability_for_lime(combined_texts):
    if preprocessor is None or model is None:
        return np.array([[0.5, 0.5]] * len(combined_texts))
    
    subjects = []
    senders = []
    bodies = []

    for combined_text in combined_texts:
        s_marker = "subject: "
        d_marker = " sender: " 
        b_marker = " body: "    

        s_text, d_text, b_text = "", "", ""

        if d_marker in combined_text:
            s_text_part, rest = combined_text.split(d_marker, 1)
            if s_marker in s_text_part:
                s_text = s_text_part.replace(s_marker, "").strip()
            
            if b_marker in rest:
                d_text_part, b_text_part = rest.split(b_marker, 1)
                d_text = d_text_part.strip()
                b_text = b_text_part.strip()
            else: 
                d_text = rest.strip()
        else: 
             if s_marker in combined_text and b_marker in combined_text :
                  s_text_part, b_text_part = combined_text.split(b_marker, 1)
                  s_text = s_text_part.replace(s_marker, "").strip()
                  b_text = b_text_part.strip()
             elif s_marker in combined_text: 
                  s_text = combined_text.replace(s_marker,"").strip()
             else: 
                  b_text = combined_text.strip()


        subjects.append(simple_text_clean(s_text))
        senders.append(simple_text_clean(d_text))
        bodies.append(simple_text_clean(b_text))

    data_for_lime = pd.DataFrame({
        'subject': subjects,
        'sender': senders,
        'body': bodies
    })
    
    try:
        vectorized_input = preprocessor.transform(data_for_lime)
        probabilities = model.predict_proba(vectorized_input)
        return probabilities
    except Exception as e:
        print(f"Error in model_predict_probability_for_lime function during transform/predict: {e}")
        return np.array([[0.5, 0.5]] * len(combined_texts))

def get_prediction_and_explanation(subject: str, sender: str, body: str):
    if preprocessor is None or model is None:
        return {"error": "Model/Preprocessor not loaded. Check server logs.", "prediction": "Error", "label": -1, "confidence": 0.0, "explanation": []}
    
    cleaned_subject = simple_text_clean(subject)
    cleaned_sender = simple_text_clean(sender)
    cleaned_body = simple_text_clean(body)

    input_df_for_model = pd.DataFrame({
        'subject': [cleaned_subject],
        'sender': [cleaned_sender],
        'body': [cleaned_body]
        })

    try:
        vectorized_input = preprocessor.transform(input_df_for_model)
        prediction_label_int = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]
        
        predicted_class_name = class_names[prediction_label_int]
        confidence_score = probabilities[prediction_label_int]
    except Exception as e:
        return {"error": f"Prediction error: {e}", "prediction": "Error", 
                "label": -1, "confidence": 0.0, "explanation": []}

    text_for_lime = f"{cleaned_subject} : {cleaned_sender} : {cleaned_body}"

    explanation_data = []
    try:
        exp = explainer.explain_instance(
            text_instance=text_for_lime, 
            classifier_fn=model_predict_probability_for_lime, 
            num_features=15, 
            top_labels=1,  
            labels=(prediction_label_int,)
        )
        explanation_data = exp.as_list(label=prediction_label_int) 
        print(f"LIME Explanation (Top 3): {explanation_data[:3]}")
    except Exception as e:
        print(f"LIME explanation error: {e}")
        explanation_data = [("LIME explanation error or N/A", 0.0)]

    return {
        "prediction": predicted_class_name,
        "label": int(prediction_label_int),
        "confidence": float(confidence_score),
        "explanation": explanation_data
    }
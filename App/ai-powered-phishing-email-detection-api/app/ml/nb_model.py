import joblib
import pandas as pd
import numpy as np
import os
from lime.lime_text import LimeTextExplainer
from .common import simple_text_clean, CLASS_NAMES 

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets') 
PREPROCESSOR_FILENAME = "multinomial_nb_email_preprocessor.joblib"
MODEL_FILENAME = "trained_multinomial_nb_model.joblib"
PREPROCESSOR_PATH = os.path.join(ASSETS_DIR, PREPROCESSOR_FILENAME)
MODEL_PATH = os.path.join(ASSETS_DIR, MODEL_FILENAME)

nb_preprocessor = None
nb_model = None
lime_explainer_nb = None

try:
    nb_preprocessor = joblib.load(PREPROCESSOR_PATH)
    nb_model = joblib.load(MODEL_PATH)
    lime_explainer_nb = LimeTextExplainer(class_names=CLASS_NAMES)
    print("Multinomial NB model, Preprocessor, and LIME Explainer loaded successfully.")
except FileNotFoundError:
    print(f"FATAL ERROR (Naive Bayes): Could not find model ('{MODEL_PATH}') or nb_preprocessor ('{PREPROCESSOR_PATH}').")
    print("Ensure files are in 'app/assets/' and filenames are correct.")
except Exception as e:
    print(f"Error loading Multinomial NB model/preprocessor or initializing LIME: {e}")

def model_predict_probability_for_lime(combined_texts):
    if nb_preprocessor is None or nb_model is None:
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
        vectorized_input = nb_preprocessor.transform(data_for_lime)
        probabilities = nb_model.predict_proba(vectorized_input)
        return probabilities
    except Exception as e:
        print(f"Error in model_predict_probability_for_lime function during transform/predict: {e}")
        return np.array([[0.5, 0.5]] * len(combined_texts))

def get_prediction_and_explanation_nb(subject: str, sender: str, body: str):
    if nb_preprocessor is None or nb_model is None:
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
        vectorized_input = nb_preprocessor.transform(input_df_for_model)
        prediction_label_int = nb_model.predict(vectorized_input)[0]
        probabilities = nb_model.predict_proba(vectorized_input)[0]
        
        predicted_class_name = CLASS_NAMES[prediction_label_int]
        confidence_score = probabilities[prediction_label_int]
    except Exception as e:
        return {"error": f"Prediction error: {e}", "prediction": "Error", 
                "label": -1, "confidence": 0.0, "explanation": []}

    text_for_lime = f"{cleaned_subject} : {cleaned_sender} : {cleaned_body}"

    explanation_data = []
    try:
        exp = lime_explainer_nb.explain_instance(
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
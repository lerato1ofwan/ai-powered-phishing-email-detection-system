# app/ml/bert_mini_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from .common import simple_text_clean, CLASS_NAMES
import traceback
import os
from transformers_interpret import SequenceClassificationExplainer 

# # Load BERT-mini model and tokenizer from Hugging Face Hub 
BERT_MODEL_ID = "lleratodev/720-bert-mini-phishing-fine-tune" 
bert_mini_tokenizer = None
bert_mini_model = None
device = None
cls_explainer_bert_mini = None

try:
    bert_mini_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_ID)
    bert_mini_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_ID)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_mini_model.to(device)
    bert_mini_model.eval() 
    cls_explainer_bert_mini = SequenceClassificationExplainer(bert_mini_model, bert_mini_tokenizer)
    
    print(f"BERT-mini model ('{BERT_MODEL_ID}'), tokenizer, and Transformers-Interpret Explainer loaded successfully.")
    print(f"BERT-mini model running on device: {device}")

except Exception as e:
    print(f"FATAL ERROR (BERT): Could not load model/tokenizer '{BERT_MODEL_ID}' or initialize Transformers-Interpret Explainer: {e}")
    traceback.print_exc()

# # Using BERT-Mini model to make email classifications 
def bert_mini_predict_probability_for_lime(text_instances: list) -> np.ndarray:
    if bert_mini_tokenizer is None or bert_mini_model is None:
        # Return neutral probabilities if model isn't loaded (number of instances, number of classes)
        return np.array([[1.0/len(CLASS_NAMES)] * len(CLASS_NAMES)] * len(text_instances))

    all_probabilities = []
    try:
        for text_instance in text_instances:
            inputs = bert_mini_tokenizer(text_instance, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = bert_mini_model(**inputs)
                logits = outputs.logits
            
            probabilities_tensor = torch.softmax(logits, dim=-1)
        
            probabilities_for_instance = probabilities_tensor.cpu().numpy().squeeze() 
            all_probabilities.append(probabilities_for_instance)
        
        return np.array(all_probabilities)

    except Exception as e:
        print(f"Error in bert_mini_predict_probability_for_lime: {e}")
        traceback.print_exc()
        return np.array([[1.0/len(CLASS_NAMES)] * len(CLASS_NAMES)] * len(text_instances))


def get_prediction_and_explanation_bert_mini(subject: str, sender: str, body: str) -> dict:
    if bert_mini_tokenizer is None or bert_mini_model is None or cls_explainer_bert_mini is None:
        return {"error": "BERT-Mini Model/Tokenizer/Explainer not loaded correctly. Check server logs.",
                "prediction": "Error", "label": -1, "confidence": 0.0, "explanation": []}

    cleaned_sender = simple_text_clean(sender)
    cleaned_subject = simple_text_clean(subject)
    cleaned_body = simple_text_clean(body)
    
    combined_text_for_prediction = f"{cleaned_sender} {cleaned_subject} {cleaned_body}"


    try:
        inputs = bert_mini_tokenizer(combined_text_for_prediction, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = bert_mini_model(**inputs)
            logits = outputs.logits

        probabilities_tensor = torch.softmax(logits, dim=-1)
        probabilities = probabilities_tensor.cpu().numpy()[0]

        prediction_label_int = np.argmax(probabilities).item()
        confidence_score = probabilities[prediction_label_int].item()
        predicted_class_name = CLASS_NAMES[prediction_label_int]

        explanation_data = []
        try:
            word_attributions = cls_explainer_bert_mini(
                combined_text_for_prediction,
                index = prediction_label_int 
            )
            
            explanation_data = [(word, float(score)) for word, score in word_attributions]
            
            explanation_data.sort(key=lambda x: abs(x[1]), reverse=True)
            explanation_data = explanation_data[:15] 

        except Exception as e:
            print(f"Transformers-Interpret explanation error: {e}")
            traceback.print_exc()
            explanation_data = [("Explanation error with Transformers-Interpret", 0.0)]
        # --- End Explanation ---

        return {
            "prediction": predicted_class_name,
            "label": int(prediction_label_int),
            "confidence": float(confidence_score),
            "explanation": explanation_data,
            "error": None
        }
    except Exception as e:
        print(f"--- ORIGINAL ERROR in predict_with_bert_mini ---")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return {"error": f"BERT-Mini Prediction error: {str(e)}", "prediction": "Error",
                "label": -1, "confidence": 0.0, "explanation": []}
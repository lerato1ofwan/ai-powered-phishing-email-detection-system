# app/ml/__init__.py
from .nb_model import get_prediction_and_explanation_nb, nb_model, nb_preprocessor
from .bert_mini_model import get_prediction_and_explanation_bert_mini, bert_mini_model, bert_mini_tokenizer
from typing import Dict

def get_model_prediction(subject: str, sender: str, body: str, model_choice: str = "nb") -> Dict:
    """
    # Dispatcher function to get predictions from the chosen model.
    """
    if model_choice == "bert-mini":
        if bert_mini_model is None or bert_mini_tokenizer is None: 
             return {"error": "BERT-Mini Model/Tokenizer is not available. Check server logs.",
                    "prediction": "Error", "label": -1, "confidence": 0.0, "explanation": []}
        return get_prediction_and_explanation_bert_mini(subject, sender, body)
    elif model_choice == "nb":
        if nb_model is None or nb_preprocessor is None: # Check if NB loaded successfully
            return {"error": "Multinomial Naive Bayes Model/Preprocessor is not available. Check server logs.",
                    "prediction": "Error", "label": -1, "confidence": 0.0, "explanation": []}
        return get_prediction_and_explanation_nb(subject, sender, body)
    else:
        return {"error": f"Invalid model_choice: '{model_choice}'. Choose 'nb' or 'bert-mini'.",
                "prediction": "Error", "label": -1, "confidence": 0.0, "explanation": []}

def check_model_status():
    status = {
        "naive_bayes": {
            "model_loaded": nb_model is not None,
            "preprocessor_loaded": nb_preprocessor is not None
        },
        "bert-mini": {
            "model_loaded": bert_mini_model is not None,
            "tokenizer_loaded": bert_mini_tokenizer is not None
        }
    }
    return status
from fastapi import FastAPI
from pydantic import BaseModel
import os
import subprocess
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional 
from .ml import get_model_prediction, check_model_status

app = FastAPI(title="AI-Powered Phishing Email Detection System")

# Define allowed origins for CORS
origins = [
    "https://ai-powered-phishing-email-detection-system.vercel.app",
    "http://localhost:3000",
    "https://huggingface.co/spaces/lleratodev/multinomial-nb-phishing-email-detection-api",
    "https://*.hf.space",
    "https://huggingface.co/spaces/lleratodev/multinomial-nb-phishing-email-detection-api.hf.space",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True, # Allows cookies to be included in requests
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Input data model
class EmailInput(BaseModel):
    subject: Optional[str] = ""
    sender: Optional[str] = ""
    body: str
    model_choice: Optional[str] = "nb" # Default to Naive Bayes

# Define output data model
class PredictionResponse(BaseModel):
    prediction: str
    label: int
    confidence: float
    explanation: List[Tuple[str, float]] 
    error: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "AI-Powered Phishing Email Detection API. POST to /predict with 'subject', 'sender', 'body'."}

@app.get("/debug-info")
async def get_debug_info():
    try:
        cwd = os.getcwd()
        ls_output = subprocess.check_output(["ls", "-la", cwd], text=True)
        env_vars = dict(os.environ)
        # Add more commands or info as needed
        return {
            "cwd": cwd,
            "ls_output": ls_output,
            "environment_variables": env_vars
            # Be careful not to expose sensitive environment variables
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/status")
async def model_status():
    return check_model_status() 


@app.post("/predict", response_model=PredictionResponse)
async def predict_email(email_input: EmailInput):
 
    if email_input.model_choice not in ["nb", "bert-mini"]:
        return PredictionResponse(prediction="Error", label=-1, confidence=0.0, explanation=[],
                                  error="Invalid model_choice. Please use 'nb' or 'bert-mini'.")
    try:
        result = get_model_prediction(
            subject=email_input.subject or "", 
            sender=email_input.sender or "",
            body=email_input.body,
            model_choice=email_input.model_choice
        )
        return PredictionResponse(**result)

    except Exception as e:
        # Fallback for truly unexpected errors in the endpoint itself
        return PredictionResponse(prediction="Error", label=-1, confidence=0.0, explanation=[],
                                  error=f"Critical API endpoint error: {str(e)}")
# AI-Powered Phishing Email Detection System

**Author:** Lerato Letsepe (u25468023)
**Course:** COS720 Computer Information & Security I
**Project Submission Date:** May 17, 2025

## 1. Introduction

This project is an AI-Powered Phishing Email Detection System designed to classify emails as either legitimate or phishing attempts. It features a web-based interface allowing users to input email content (sender, subject, body) or upload an email file. Users can choose between two AI models for classification: a traditional Multinomial Naive Bayes (MultinomialNB) model and a fine-tuned BERT-mini transformer model. The system also provides explainability for the AI's decisions.

This `README.md` provides instructions on how to set up and run the project locally using Docker.

## 2. Project Structure Overview

The project codebase is organized as follows:

![image](https://github.com/user-attachments/assets/bc89cedd-78e4-4950-81fb-c086fd8058f3)


## 3. Technologies Used

* **Frontend:**
    * Next.js (v13+)
    * TypeScript
    * TailwindCSS
    * React
* **Backend:**
    * Python (v3.9+)
    * FastAPI
* **Machine Learning:**
    * Scikit-learn (for MultinomialNB, TF-IDF)
    * Hugging Face Transformers (for BERT-mini)
    * Pandas, Numpy
    * LIME (for Naive Bayes explanations)
    * transformers-interpret (for BERT-mini explanations)
* **Containerization:**
    * Docker
    * Docker Compose

## 4. Prerequisites

To run this project locally, you **must** have the following installed:

* **Docker Desktop** (or Docker Engine and Docker Compose CLI separately for Linux)

No other local Python or Node.js environment setup is strictly necessary if running via Docker, as the containers will manage dependencies.

## 5. Setup and Running the Application

1.  **Obtain the Code:**
    * Clone the repository (if applicable):
        ```bash
        git clone <repository_url>
        cd <repository_folder_name>
        ```
    * Or, extract the submitted ZIP file (`codebase.zip` or similar) into a directory on your local machine.

2.  **Navigate to the Root Directory:**
    Open a terminal or command prompt and navigate to the root directory of the project where the `docker-compose.yml` file is located.

3.  **Start the Application:**
    Run the following command to build the Docker images (if they don't exist) and start the frontend and backend services:
    ```bash
    docker compose up
    ```
    You may need to run `docker-compose up` (with a hyphen) if you are using an older version of Docker Compose.
    Wait for the services to build and start. You will see logs from both the frontend and backend containers.

4.  **Access the Application:**
    Once the containers are running, open your web browser and navigate to:
    [http://localhost:3000](http://localhost:3000)

    The AI-Powered Phishing Email Detection System interface should be accessible.

## 6. Stopping the Application

To stop the running Docker containers:

1.  Go to the terminal window where `docker compose up` is running and press `Ctrl+C`.
2.  To ensure the containers are stopped and removed (optional, but good for cleanup), you can run:
    ```bash
    docker compose down
    ```

## 7. Model Information

* **Multinomial Naive Bayes (MultinomialNB):**
    * The pre-trained scikit-learn MultinomialNB model and its TF-IDF preprocessor (`.joblib` files) are included within the backend API's `app/assets/` directory and are loaded directly by the application.
* **BERT-mini:**
    * The fine-tuned BERT-mini model (`prajjwall/bert-mini`) is loaded from the Hugging Face Model Hub at runtime by the backend API. The fine-tuning process and artifacts are detailed in the Jupyter Notebooks and the `Models/720-bert-mini-phishing-fine-tune/` directory provided in the submission.

The Jupyter Notebooks (`COS720_w_BERT_Trained_Refined.ipynb` and `MultinomialNB_Final_Submission_u25468023_COS_720_Project.ipynb`) detailing the model training and evaluation process can be found in the `Models/` directory submitted alongside the main codebase. This directory also contains the standalone trained model artifacts.

## 8. Email Samples for Testing

A collection of email samples (in `.txt` format) used for testing the prototype, as detailed in the project report (Section 3.5), can be found in the `Email Samples/` directory submitted alongside the main codebase.

## 9. Project Report

For a comprehensive understanding of the research, design decisions, model selection, evaluation, and detailed testing of this system, please refer to the main project report document:
`u25468023 Lerato Letsepe COS720 Project 2025.pdf`.

---

main-folder/
├── docker-compose.yml (Runs the api and backend)
├── README.md (Currently empty)
├── ai-powered-phishing-email-detection-frontend/
│   ├── Dockerfile
│   ├── next.config.ts
│   ├── package-lock.json
│   ├── package.json
│   ├── tsconfig.json
│   ├── app/ (Contains main frontend logic)
│       └── layout.tsx
│       └── page.tsx
│       └── globals.css
│
└── ai-powered-phishing-email-detection-api/
    ├── Dockerfile
    └── requirements.txt
    └── app/
        ├── main.py
        ├── __init__.py
        ├── ml_logic.py
        ├── assests/ (Contains MultinomialNM model .joblibs)
            └── multinomial_nb_email_preprocessor.joblib
            └── trained_multinomial_nb_model.joblib
        └── ml/ (Contains classification logic and ML implementations of predictions and explanations etc)
            └── __init__.py
            └── bert_mini_model.py
            └── common.py (Implements simple_text_clean function)
            └── nb_model.py
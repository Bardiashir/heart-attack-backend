# Heart Attack Risk Predictor – FastAPI Backend 🫀

This project is a complete backend system for predicting heart attack risk using a trained Machine Learning model (Random Forest) wrapped inside a FastAPI API. It includes a full preprocessing pipeline, input validation, and a clean prediction endpoint ready to be used by any frontend or mobile app.

---

## 🚀 Features

- **Machine Learning model** trained on the UCI Heart Disease dataset  
- **Full preprocessing pipeline** using scikit-learn (ColumnTransformer + Pipeline)  
- **RandomForestClassifier** with solid validation performance  
- **FastAPI backend** exposing:
  - `GET /health` – health check endpoint  
  - `POST /predict` – returns risk probability, risk label, and a short recommendation message  
- **Pydantic models** for request & response validation  
- **Automatic API docs** with Swagger UI at `/docs`  
- Designed to be easily connected to a web or mobile frontend

---

## 🧠 Tech Stack

- **Python 3**
- **FastAPI**
- **Uvicorn**
- **scikit-learn**
- **Pandas / NumPy**
- **Pydantic**

---

## 📁 Project Structure

> Folder names may differ slightly depending on setup, but the core idea is:


backend/
└── src/
    ├── api.py          # FastAPI app and endpoints
    ├── schema.py       # Request/response models (Pydantic)
    ├── train.py        # Model training + saving pipeline
    ├── models/
       └── heart_model.pkl   # Trained ML model
    ├── data/
       └── heart_disease_uci.csv   # Dataset used for model training

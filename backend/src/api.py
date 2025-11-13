from fastapi import FastAPI, HTTPException
from pathlib import Path
import joblib
import pandas as pd

from schema import HeartRiskRequest, HeartRiskResponse

app = FastAPI(
    title="Heart Risk API",
    version="0.0.2",
    description="Predicts heart disease risk using a trained ML model.",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heart_model.pkl"

# Load model artifact at startup
try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    model_name = artifact.get("model_name", "unknown_model")
    input_features = artifact["input_features"]
    feature_config = artifact.get("feature_config", {})
except Exception as e:
    print(f"⚠️ Failed to load model from {MODEL_PATH}: {e}")
    model = None
    model_name = "unavailable"
    input_features = []
    feature_config = {}


@app.get("/health")
def health():
    """
    Simple health check to verify that the API is running
    and the model is (or isn't) loaded.
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": model_name,
    }


@app.post("/predict", response_model=HeartRiskResponse)
def predict_heart_risk(data: HeartRiskRequest):
    """
    Take user input, map it to the model's expected features,
    let the pipeline handle preprocessing, and return risk.
    """
    if model is None:
        raise HTTPException(
            status_code=500, detail="Model is not loaded on the server.")

    # Convert incoming payload to dict
    payload = data.dict()

    # Map to the columns the model was trained on.
    # Any missing advanced features stay as None -> become NaN -> handled by imputers.
    row = {}

    # Core fields (we trained with: age, sex, cp, trestbps, chol, fbs, restecg,
    # thalch, exang, oldpeak, slope, ca, thal)
    # We only add keys that exist in input_features for safety.
    for feature in input_features:
        if feature in payload:
            row[feature] = payload[feature]
        else:
            # If it's not in user schema (e.g. legacy), set to None and let imputers handle it
            row[feature] = None

    # Special handling: cp, restecg, slope, thal might be text labels already;
    # that's fine because our training pipeline treated them as categoricals with OneHotEncoder.

    # Create DataFrame with a single row in correct column order
    X = pd.DataFrame([row], columns=input_features)

    try:
        # probability of class "1" (disease)
        proba = model.predict_proba(X)[0][1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Turn probability into label
    risk_prob = float(proba)
    risk_percent = round(risk_prob * 100, 2)

    if risk_prob < 0.3:
        label = "low"
        msg = (
            "Your predicted risk is low based on the information provided. "
            "Maintain healthy habits (diet, exercise, no smoking) ❤️"
        )
    elif risk_prob < 0.6:
        label = "medium"
        msg = (
            "Your predicted risk is moderate. Consider improving lifestyle, "
            "monitor blood pressure/cholesterol, and talk to your doctor if concerned."
        )
    else:
        label = "high"
        msg = (
            "Your predicted risk is high. This tool is NOT a diagnosis, but you "
            "should speak with a healthcare professional as soon as possible. 🩺"
        )

    return HeartRiskResponse(
        risk_probability=round(risk_prob, 4),
        risk_percent=risk_percent,
        risk_label=label,
        message=msg,
        model_name=model_name,
    )

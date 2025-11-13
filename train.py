import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib


# ===== Paths =====
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "heart_disease_uci.csv"  # adjust name if needed
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "heart_model.pkl"


def load_and_prepare_data():
    """
    Load dataset, create binary target, and select feature columns.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # ---- 1) Build binary target from 'num' (0=no disease, 1-4=disease) ----
    if "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)
    elif "target" in df.columns:
        # already binary
        pass
    else:
        raise ValueError("Dataset must contain 'num' or 'target' column.")

    # ---- 2) Drop rows with missing target ----
    df = df.dropna(subset=["target"])

    # ---- 3) Define feature set ----
    # We use all clinically relevant columns for stronger predictions.
    # 'id' and 'dataset' are identifiers/meta -> drop them.
    candidate_features = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalch",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    missing = [c for c in candidate_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in dataset: {missing}")

    X = df[candidate_features].copy()
    y = df["target"].astype(int)

    # ---- 4) Normalize / clean raw values (basic mapping) ----

    # sex: "Male"/"Female" -> 1/0
    if X["sex"].dtype == object:
        X["sex"] = X["sex"].map({"Male": 1, "Female": 0})

    # booleans -> int
    for col in ["fbs", "exang"]:
        if col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)

    # Ensure numeric types where expected (errors='ignore' so categoricals stay)
    for col in ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Done: leave cp, restecg, slope, thal as categoricals (strings)

    return X, y, candidate_features


def build_preprocessor(feature_names):
    """
    Create a ColumnTransformer that:
    - imputes + scales numeric
    - imputes + passes through binary
    - imputes + one-hot encodes categoricals
    """
    # Decide which columns are which
    numeric_features = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    binary_features = ["sex", "fbs", "exang"]
    categorical_features = ["cp", "restecg", "slope", "thal"]

    # Filter to only existing ones (safety)
    numeric_features = [c for c in numeric_features if c in feature_names]
    binary_features = [c for c in binary_features if c in feature_names]
    categorical_features = [c for c in categorical_features if c in feature_names]

    # Pipelines for each type
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # no scaling; already 0/1
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bin", binary_transformer, binary_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, {
        "numeric": numeric_features,
        "binary": binary_features,
        "categorical": categorical_features,
    }


def get_models():
    """
    Define candidate models to try.
    We'll pick the best by validation accuracy.
    """
    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }
    return models


def train_and_select_best_model():
    # ---- Load data ----
    X, y, feature_names = load_and_prepare_data()

    # ---- Train/Val split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # ---- Preprocessor ----
    preprocessor, feature_config = build_preprocessor(feature_names)

    # ---- Candidate models ----
    models = get_models()

    best_name = None
    best_pipeline = None
    best_acc = -1.0

    # ---- Train & evaluate each ----
    for name, clf in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", clf),
            ]
        )

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        acc = accuracy_score(y_val, preds)

        print(f"{name} validation accuracy: {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_pipeline = pipe

    if best_pipeline is None:
        raise RuntimeError("No model was successfully trained.")

    print(f"\nSelected best model: {best_name} with accuracy {best_acc:.3f}")

    # ---- Save best model + metadata ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_name": best_name,
        "model": best_pipeline,
        "input_features": feature_names,
        "feature_config": feature_config,
        "target_mean": float(y.mean()),  # can be used as a fallback baseline
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved best model to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_select_best_model()

from typing import Optional
from pydantic import BaseModel, Field


class HeartRiskRequest(BaseModel):
    # Core, simple questions (things users are likely to know)
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., description="1 = male, 0 = female")
    cp: Optional[str] = Field(
        default=None,
        description=(
            "Chest pain type: "
            "'typical angina', 'atypical angina', "
            "'non-anginal pain', or 'asymptomatic'"
        ),
    )
    trestbps: Optional[float] = Field(
        default=None,
        description="Resting blood pressure (systolic, mm Hg), if known",
    )
    chol: Optional[float] = Field(
        default=None,
        description="Serum cholesterol (mg/dL), if known",
    )
    fbs: Optional[int] = Field(
        default=None,
        description="Fasting blood sugar > 120 mg/dL? 1 = yes, 0 = no/unknown",
    )
    exang: Optional[int] = Field(
        default=None,
        description="Exercise-induced chest pain? 1 = yes, 0 = no",
    )

    # Advanced/clinical fields — optional.
    # If user doesn't provide them, our model pipeline imputes sensible values.
    restecg: Optional[str] = Field(
        default=None,
        description="Resting ECG result (optional)",
    )
    thalch: Optional[float] = Field(
        default=None,
        description="Max heart rate achieved, if known",
    )
    oldpeak: Optional[float] = Field(
        default=None,
        description="ST depression induced by exercise, if known",
    )
    slope: Optional[str] = Field(
        default=None,
        description="Slope of peak exercise ST segment (optional)",
    )
    ca: Optional[float] = Field(
        default=None,
        description="Number of major vessels (0-3), if known",
    )
    thal: Optional[str] = Field(
        default=None,
        description="Thalassemia status (optional)",
    )


class HeartRiskResponse(BaseModel):
    risk_probability: float  # between 0 and 1
    risk_percent: float      # 0-100
    risk_label: str          # "low" / "medium" / "high"
    message: str             # friendly explanation
    model_name: str          # which model was used (rf, gb, etc.)

"""FastAPI inference service for the NutriScore grade estimator.

Run from the project root:
    python -m uvicorn app:app --reload

This model estimates labels learned from the project dataset. It is not an
implementation of the official Nutri-Score algorithm.
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

import nutriscore_utils as nu


GRADES = ["A", "B", "C", "D", "E"]
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "outputs_nutriscore" / "nutriscore_pipeline.joblib"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        "Model artifact tidak ditemukan. Pastikan file tersedia di: "
        f"{MODEL_PATH}"
    )

# Load the complete preprocessing + estimator pipeline once when the API starts.
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="NutriScore Grade Estimator API",
    description=(
        "API untuk memperkirakan grade A-E berdasarkan model machine learning "
        "yang dilatih pada dataset proyek. Hasil bukan perhitungan resmi Nutri-Score."
    ),
    version="1.0.0",
)


class Product(BaseModel):
    """Product nutrition data; nutrient values use a per-100-gram basis."""

    energy_kcal: Optional[float] = Field(default=None, ge=0, le=900)
    fat_100g: Optional[float] = Field(default=None, ge=0, le=100)
    saturated_fat_100g: Optional[float] = Field(default=None, ge=0, le=100)
    carbs_100g: Optional[float] = Field(default=None, ge=0, le=100)
    sugars_100g: Optional[float] = Field(default=None, ge=0, le=100)
    fiber_100g: Optional[float] = Field(default=None, ge=0, le=100)
    proteins_100g: Optional[float] = Field(default=None, ge=0, le=100)
    salt_100g: Optional[float] = Field(default=None, ge=0, le=100)
    sodium_100g: Optional[float] = Field(default=None, ge=0, le=100)

    categories: str = ""

    # Send explicit booleans; unknown allergen information must not be treated
    # automatically as False.
    contains_gluten: bool
    contains_dairy: bool
    contains_nuts: bool
    contains_soy: bool
    contains_eggs: bool
    contains_fish: bool

    @model_validator(mode="after")
    def validate_nutrition_consistency(self):
        nutrient_values = [
            self.energy_kcal,
            self.fat_100g,
            self.saturated_fat_100g,
            self.carbs_100g,
            self.sugars_100g,
            self.fiber_100g,
            self.proteins_100g,
            self.salt_100g,
            self.sodium_100g,
        ]

        if all(value is None for value in nutrient_values):
            raise ValueError("Minimal satu informasi nutrisi harus diisi.")

        if (
            self.fat_100g is not None
            and self.saturated_fat_100g is not None
            and self.saturated_fat_100g > self.fat_100g
        ):
            raise ValueError("Lemak jenuh tidak boleh melebihi lemak total.")

        if (
            self.carbs_100g is not None
            and self.sugars_100g is not None
            and self.sugars_100g > self.carbs_100g
        ):
            raise ValueError("Gula tidak boleh melebihi karbohidrat total.")

        return self


@app.get("/health", tags=["Monitoring"])
def health():
    """Report whether the API process started and the model is loaded."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": "Ordinal-HGB (tuned)",
    }


@app.post("/predict", tags=["Prediction"])
def predict(product: Product):
    """Predict an estimated A-E grade from raw nutrition/product fields."""
    input_row = pd.DataFrame([product.model_dump()])
    input_row = input_row[nu.RAW_INPUT_COLS]

    scores = np.asarray(model.predict_proba(input_row)[0], dtype=float)
    if scores.shape[0] != len(GRADES):
        raise RuntimeError(
            "Jumlah output model tidak sesuai dengan lima grade A-E."
        )

    predicted_index = int(np.argmax(scores))

    return {
        "model": "Ordinal-HGB (tuned)",
        "predicted_grade": GRADES[predicted_index],
        "class_scores": {
            grade: round(float(score), 4)
            for grade, score in zip(GRADES, scores)
        },
        "note": (
            "Skor kelas adalah keluaran model dan belum dibuktikan terkalibrasi. "
            "Prediksi merupakan estimasi berdasarkan dataset proyek, bukan "
            "penilaian atau perhitungan resmi Nutri-Score."
        ),
    }

@app.get("/")
def root():
    return {
        "project": "Nutri-Score AI Engineering",
        "description": "Machine Learning API for Nutri-Score prediction",
        "status": "online",
        "documentation": "/docs",
        "health": "/health"
    }

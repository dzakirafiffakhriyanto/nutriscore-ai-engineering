# Contoh API untuk melayani model Nutri-Score.
# Jalankan dari folder project setelah model dan nutriscore_utils.py tersedia:
# uvicorn app:app --reload

from typing import Optional

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

import nutriscore_utils as nu

GRADES = ["A", "B", "C", "D", "E"]
MODEL_PATH = "outputs_nutriscore/nutriscore_pipeline.joblib"

model = joblib.load(MODEL_PATH)
app = FastAPI(
    title="Nutri-Score Grade Estimator",
    description=(
        "Prototype API untuk memperkirakan grade berdasarkan model "
        "yang dilatih pada dataset proyek. Bukan kalkulator resmi Nutri-Score."
    ),
    version="1.0.0",
)


class Product(BaseModel):
    # Semua nilai nutrisi menggunakan basis per 100 gram.
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

    # Harus dikirim secara eksplisit agar data yang tidak diketahui
    # tidak otomatis dianggap False.
    contains_gluten: bool
    contains_dairy: bool
    contains_nuts: bool
    contains_soy: bool
    contains_eggs: bool
    contains_fish: bool

    @model_validator(mode="after")
    def check_nutrition_consistency(self):
        nutrient_values = [
            self.energy_kcal, self.fat_100g, self.saturated_fat_100g,
            self.carbs_100g, self.sugars_100g, self.fiber_100g,
            self.proteins_100g, self.salt_100g, self.sodium_100g
        ]

        if all(value is None for value in nutrient_values):
            raise ValueError(
                "Minimal satu informasi nutrisi harus diisi."
            )

        if (
            self.fat_100g is not None
            and self.saturated_fat_100g is not None
            and self.saturated_fat_100g > self.fat_100g
        ):
            raise ValueError(
                "Lemak jenuh tidak boleh melebihi lemak total."
            )

        if (
            self.carbs_100g is not None
            and self.sugars_100g is not None
            and self.sugars_100g > self.carbs_100g
        ):
            raise ValueError(
                "Gula tidak boleh melebihi karbohidrat total."
            )

        return self


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(product: Product):
    row = pd.DataFrame([product.model_dump()])[nu.RAW_INPUT_COLS]
    scores = model.predict_proba(row)[0]
    predicted_index = int(np.argmax(scores))

    return {
        "predicted_grade": GRADES[predicted_index],
        "class_scores": {
            grade: round(float(score), 4)
            for grade, score in zip(GRADES, scores)
        },
        "note": (
            "Skor kelas berasal dari model klasifikasi proyek dan belum "
            "dibuktikan terkalibrasi. Hasil adalah estimasi dataset, "
            "bukan penilaian resmi Nutri-Score."
        ),
    }

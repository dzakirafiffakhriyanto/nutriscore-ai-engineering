"""API tests for the NutriScore Grade Estimator.

Run from the project root:
    python -m pytest -v

These tests load the real model through app.py. Keep the model artifact and
nutriscore_utils.py available before running them.
"""

import math

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

VALID_PRODUCT = {
    "energy_kcal": 320,
    "fat_100g": 8.0,
    "saturated_fat_100g": 2.5,
    "carbs_100g": 55.0,
    "sugars_100g": 12.0,
    "fiber_100g": 5.0,
    "proteins_100g": 7.0,
    "salt_100g": 0.8,
    "sodium_100g": 0.32,
    "categories": "en:snacks",
    "contains_gluten": False,
    "contains_dairy": False,
    "contains_nuts": False,
    "contains_soy": False,
    "contains_eggs": False,
    "contains_fish": False,
}


def test_health_endpoint_reports_loaded_model():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "Ordinal-HGB (tuned)"


def test_predict_endpoint_returns_grade_and_class_scores():
    response = client.post("/predict", json=VALID_PRODUCT)

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "Ordinal-HGB (tuned)"
    assert body["predicted_grade"] in ["A", "B", "C", "D", "E"]
    assert set(body["class_scores"].keys()) == {"A", "B", "C", "D", "E"}
    assert all(0.0 <= score <= 1.0 for score in body["class_scores"].values())
    assert math.isclose(sum(body["class_scores"].values()), 1.0, abs_tol=1e-3)
    assert "bukan" in body["note"].lower()


def test_predict_rejects_negative_nutrient_value():
    invalid_product = {**VALID_PRODUCT, "sugars_100g": -1}
    response = client.post("/predict", json=invalid_product)

    assert response.status_code == 422


def test_predict_rejects_saturated_fat_above_total_fat():
    invalid_product = {
        **VALID_PRODUCT,
        "fat_100g": 5.0,
        "saturated_fat_100g": 7.0,
    }
    response = client.post("/predict", json=invalid_product)

    assert response.status_code == 422


def test_predict_rejects_sugar_above_total_carbohydrates():
    invalid_product = {
        **VALID_PRODUCT,
        "carbs_100g": 10.0,
        "sugars_100g": 12.0,
    }
    response = client.post("/predict", json=invalid_product)

    assert response.status_code == 422


def test_predict_requires_explicit_allergen_flags():
    invalid_product = VALID_PRODUCT.copy()
    invalid_product.pop("contains_fish")
    response = client.post("/predict", json=invalid_product)

    assert response.status_code == 422


def test_predict_rejects_request_without_nutrition_values():
    invalid_product = {
        **VALID_PRODUCT,
        "energy_kcal": None,
        "fat_100g": None,
        "saturated_fat_100g": None,
        "carbs_100g": None,
        "sugars_100g": None,
        "fiber_100g": None,
        "proteins_100g": None,
        "salt_100g": None,
        "sodium_100g": None,
    }
    response = client.post("/predict", json=invalid_product)

    assert response.status_code == 422

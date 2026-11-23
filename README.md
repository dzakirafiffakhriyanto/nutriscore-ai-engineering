# Nutri-Score AI Engineering

![Nutri-Score CI](https://github.com/dzakirafiffakhriyanto/nutriscore-ai-engineering/actions/workflows/ci.yml/badge.svg?branch=main)

An end-to-end Machine Learning engineering project that transforms a supervised Nutri-Score classification model into a deployable REST API using **FastAPI, Docker, GitHub Actions, and Vercel**.

This project demonstrates the complete workflow of developing a Machine Learning application, from data exploration and model training to API development, automated testing, containerization, continuous integration, and cloud deployment.

**Live Application:** https://nutriscore-ai-engineering.vercel.app

**Interactive API Documentation:** https://nutriscore-ai-engineering.vercel.app/docs

---

# Dataset Sources

This project uses the following datasets from Kaggle as the raw data sources for the NutriScore classification pipeline.

## Source Dataset

**Global Food & Nutrition Database 2026**

- **Platform:** Kaggle
- **Publisher:** ahsanneural
- **Dataset URL:** https://www.kaggle.com/datasets/ahsanneural/global-food-and-nutrition-database-2026

### Files Used

The following CSV files were used:

1. `foods_health_scores_allergens.csv`
2. `foods_dietary_restrictions.csv`

### Usage in This Project

These datasets serve as the raw data sources for exploratory data analysis (EDA), data cleaning, preprocessing, and the development of the NutriScore classification model.

The datasets were processed and combined as part of the project workflow to produce the final dataset used for model development.

### Data Availability

The original CSV files are not included in this repository. Please download them directly from the original Kaggle source.

**Note:** Users must comply with the original dataset's license and terms of use when accessing or redistributing the data.

---

## Project Overview

This project extends a supervised Machine Learning workflow into a publicly accessible AI application.

The trained model predicts a Nutri-Score grade (A–E) based on product nutritional information. The model is integrated into a REST API, allowing users and external applications to submit product data and receive predictions through JSON requests.

The main objective is to demonstrate how a trained Machine Learning model can be transformed into a functional, tested, and deployed software application.

**Important:** This project uses a supervised Machine Learning classification model trained on dataset labels. It does not implement the official Nutri-Score calculation algorithm. Predictions should be interpreted as model-based estimates, not official Nutri-Score certifications.

## Key Features

- Exploratory Data Analysis (EDA) and data quality assessment.
- Data cleaning and preprocessing.
- Feature engineering and nutritional data transformation.
- Supervised Machine Learning classification with five output classes (A–E).
- Ordinal HistGradientBoosting model with hyperparameter tuning.
- Model evaluation and cross-validation.
- Model artifact persistence using Joblib.
- REST API development using FastAPI.
- Input validation using Pydantic.
- Prediction scores returned through `predict_proba`.
- Automated API testing using Pytest.
- Docker containerization.
- Continuous Integration using GitHub Actions.
- Public API deployment using Vercel.
- Interactive API documentation through Swagger UI.

## Machine Learning Model

The selected model is:

**Ordinal-HGB (Tuned)**

The model predicts one of five grade classes:

| Class | Predicted Grade |
|---|---|
| A | Grade A |
| B | Grade B |
| C | Grade C |
| D | Grade D |
| E | Grade E |

### Model Performance

The following metrics are based on the recorded final test-set evaluation:

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | 83.64% |
| Test Macro-F1 | 83.26% |
| Quadratic Weighted Kappa | 93.81% |
| Predictions Within ±1 Grade | 97.61% |

The final model was selected after comparing candidate models and evaluating their performance using cross-validation and held-out test data.

### Model Artifacts

The persisted model pipeline is stored at:

`outputs_nutriscore/nutriscore_pipeline.joblib`

Additional artifact files include:

- `metadata.json` — model configuration and evaluation metadata.
- `artifact_checksums.json` — artifact integrity information.
- `cv_all_models.csv` — cross-validation results from candidate models.
- `cv_final_models.csv` — cross-validation results for final model candidates.
- `decision_log.csv` — model selection and experimentation records.

The API loads the pre-trained model artifact for inference instead of retraining the model during application startup.

## Dataset & Reproducibility

The project uses a master dataset containing packaged food product information, nutritional values, product categories, allergen information, and Nutri-Score labels.

According to the Machine Learning notebook, the master dataset was prepared by integrating two source datasets:

- `foods_health_scores_allergens`
- `foods_dietary_restrictions`

The resulting integrated dataset is stored as `nutriscore_master`.

### Dataset Snapshot

The recorded notebook execution used a dataset containing:

- 5,052 product records.
- 22 columns.
- Five target classes (A–E).

These values represent the dataset snapshot used in the recorded experiment.

### Expected Input Files

The Machine Learning notebook accepts either of the following input formats:

- `datawrangling_fix_supervised.db`, containing a table named `nutriscore_master`.
- `nutriscore_master.csv`.

**Reproducibility note:** The current repository includes the Machine Learning notebook and exported model artifacts. The original raw datasets and complete upstream integration process are not included. Reproducing the full experiment requires obtaining the source data and preparing the expected master dataset.

Dataset sources, versions, redistribution permissions, and applicable licenses should be verified before distributing the data.

## API Architecture

The application follows a simple inference workflow:

1. A client sends product nutritional information through an HTTP request.
2. FastAPI receives the request.
3. Pydantic validates the request structure and input values.
4. The validated data is transformed into a Pandas DataFrame using the expected model input columns.
5. The persisted Machine Learning pipeline generates predictions.
6. The API returns the predicted grade and class scores as JSON.

### Input Features

The prediction endpoint accepts product information, including the following nutritional features:

| Feature | Description |
|---|---|
| `energy_kcal` | Energy per 100g |
| `fat_100g` | Total fat per 100g |
| `saturated_fat_100g` | Saturated fat per 100g |
| `carbs_100g` | Carbohydrates per 100g |
| `sugars_100g` | Sugars per 100g |
| `fiber_100g` | Fiber per 100g |
| `proteins_100g` | Protein per 100g |
| `salt_100g` | Salt per 100g |
| `sodium_100g` | Sodium per 100g |
| `categories` | Product category information |

The request also requires explicit Boolean allergen flags.

The API validates nutritional inputs to prevent invalid or inconsistent values, including:

- Negative nutritional values.
- Saturated fat exceeding total fat.
- Sugar exceeding total carbohydrates.
- Requests without any nutritional values.
- Missing required allergen flags.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Returns project information and API status |
| GET | `/health` | Checks API health and model loading status |
| POST | `/predict` | Generates a Nutri-Score grade prediction |

### Live API Endpoints

- **Root:** https://nutriscore-ai-engineering.vercel.app/
- **Health Check:** https://nutriscore-ai-engineering.vercel.app/health
- **Prediction:** https://nutriscore-ai-engineering.vercel.app/predict
- **Swagger UI:** https://nutriscore-ai-engineering.vercel.app/docs

### Interactive API Documentation

The application provides interactive API documentation through Swagger UI.

Users can explore available endpoints, submit product nutritional information, and inspect prediction responses directly from their browser without installing the application locally.

## Technology Stack

**Machine Learning**
- Python 3.13
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Joblib

**API Development**
- FastAPI
- Uvicorn
- Pydantic

**Testing**
- Pytest
- HTTPX
- FastAPI TestClient

**DevOps & Deployment**
- Docker
- Docker Desktop
- GitHub Actions
- Vercel

## Project Structure

```text
nutriscore-ai-engineering/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── notebook/
│   └── NutriScore_Supervised_Learning_Stage1_Final_Fixpynb.ipynb
│
├── outputs_nutriscore/
│   ├── nutriscore_pipeline.joblib
│   ├── metadata.json
│   ├── artifact_checksums.json
│   ├── cv_all_models.csv
│   ├── cv_final_models.csv
│   └── decision_log.csv
│
├── tests/
│   └── test_api.py
│
├── app.py
├── nutriscore_utils.py
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .python-version
├── vercel.json
└── README.md
```

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/dzakirafiffakhriyanto/nutriscore-ai-engineering.git

cd nutriscore-ai-engineering
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn app:app --reload
```

The API will be available at:

`http://localhost:8000`

Swagger documentation:

`http://localhost:8000/docs`

## Automated API Testing

The project includes automated tests to validate API behavior and input handling.

Install development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run the tests:

```bash
python -m pytest -v
```

### Test Coverage

The automated test suite includes seven test cases:

1. Health endpoint reports that the model is loaded.
2. Prediction endpoint returns a valid grade and class scores.
3. Negative nutritional values are rejected.
4. Saturated fat exceeding total fat is rejected.
5. Sugar exceeding total carbohydrates is rejected.
6. Explicit allergen flags are required.
7. Requests without nutritional values are rejected.

**Recorded local test result: 7 passed.**

## Docker Deployment

The application can be packaged and executed inside a Docker container.

### 1. Build the Docker image

From the project root directory:

```bash
docker build -t nutriscore-api .
```

### 2. Run the Docker container

```bash
docker run --rm --name nutriscore-api -p 8000:8000 nutriscore-api
```

The API will be accessible at:

`http://localhost:8000`

Swagger UI:

`http://localhost:8000/docs`

Health endpoint:

`http://localhost:8000/health`

### 3. Stop the container

Press `CTRL + C` in the terminal running the container.

### Docker Configuration

- `Dockerfile` defines the Python environment, dependency installation, model artifact copying, and API startup command.
- `.dockerignore` excludes unnecessary files and directories from the Docker build context.
- The Docker image uses `python:3.13-slim`.
- The application supports a configurable port through the `PORT` environment variable.

## Continuous Integration (CI)

This project uses **GitHub Actions** to automate API testing and Docker image validation.

The workflow is defined in:

`.github/workflows/ci.yml`

### Workflow Triggers

The CI pipeline runs automatically on:

- Push events to the `main` branch.
- Pull requests targeting the `main` branch.
- Manual workflow execution through GitHub Actions.

### CI Pipeline Steps

1. Checkout the repository.
2. Set up Python 3.13.
3. Install project and development dependencies.
4. Execute automated API tests using Pytest.
5. Build the Docker image to validate the container configuration.

The workflow verifies that the API tests pass and that the application, dependencies, and model artifacts can be packaged into a Docker image.

**Note:** This workflow performs Continuous Integration. It does not automatically publish the Docker image to a container registry.

## Cloud Deployment

The Nutri-Score API is publicly deployed using **Vercel**.

**Production URL:**

https://nutriscore-ai-engineering.vercel.app

The deployment uses Vercel's Python runtime to serve the FastAPI application.

The repository includes:

- `.python-version` — specifies the Python runtime version.
- `vercel.json` — configures the deployment and includes the model artifacts required for inference.

### Deployment Architecture

The project maintains two separate execution environments:

- **Docker:** Used for local containerized execution and automated Docker image validation through GitHub Actions.
- **Vercel:** Hosts the public FastAPI application using Vercel's Python runtime.

The Vercel deployment does not run the Docker image. Both configurations are maintained independently.

## Engineering Considerations

- The application loads a pre-trained model artifact instead of training during API startup.
- Model input columns are controlled through a shared utility module.
- Input validation helps prevent structurally invalid and inconsistent nutritional data.
- The Docker image packages the application and its runtime dependencies.
- Automated tests verify important API behaviors and validation rules.
- GitHub Actions automatically checks API tests and Docker build integrity.
- Vercel provides a publicly accessible API endpoint for demonstration and interaction.

## Limitations

- Predictions are based on a supervised model trained on dataset labels.
- The model does not implement the official Nutri-Score calculation rules.
- Returned class scores are model-generated estimates and are not necessarily calibrated probabilities.
- Prediction quality depends on training data quality and the distribution of products submitted to the API.
- The original upstream dataset integration process is not fully reproduced in the current notebook.
- The application is intended for Machine Learning demonstration and educational purposes, not as a substitute for official nutritional assessment or professional dietary advice.

## Future Improvements

Potential improvements for future development include:

- Develop a dedicated frontend for product nutrition analysis.
- Implement a nutritional insight and recommendation engine.
- Improve model probability calibration and evaluation.
- Add Explainable AI techniques to interpret model predictions.
- Build product comparison functionality.
- Introduce model performance monitoring and data drift detection.
- Improve dataset documentation and experiment reproducibility.

## Author

**Dzaki Rafif Fakhriyanto**

Machine Learning | Data Science | AI Engineering

GitHub: [dzakirafiffakhriyanto](https://github.com/dzakirafiffakhriyanto)

---

*This project demonstrates the end-to-end development of a supervised Machine Learning model into a tested, containerized, continuously validated, and publicly accessible API.*

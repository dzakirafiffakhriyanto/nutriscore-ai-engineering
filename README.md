# Nutri-Score AI Engineering

An end-to-end Machine Learning deployment project that transforms a supervised Nutri-Score classification model into a REST API using **FastAPI** and **Docker**.

This repository focuses on the engineering side of a Machine Learning project: model artifact management, API development, input validation, automated testing, and containerization.

## Project Overview

This project extends a supervised Machine Learning workflow into a deployable AI application.

The trained model predicts a Nutri-Score grade (A–E) based on product nutritional information. The model is served through a REST API, allowing users and other applications to request predictions using JSON input.

**Important:** This is a dataset-trained Machine Learning classification model. It is not an implementation of the official Nutri-Score calculation algorithm. Its predictions should be interpreted as model-based estimates, not official Nutri-Score certifications.

## Key Features

- Supervised Machine Learning classification with five output classes (A–E).
- Tuned Ordinal HistGradientBoosting model.
- Model artifact persistence using Joblib.
- REST API development using FastAPI.
- Input validation using Pydantic.
- Prediction scores returned through `predict_proba`.
- Automated API testing using Pytest.
- Docker image creation and containerized deployment.
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

The persisted model pipeline is stored in:

`outputs_nutriscore/nutriscore_pipeline.joblib`

Additional artifact files:

- `metadata.json` — model metadata.
- `artifact_checksums.json` — artifact integrity information.

The API uses the persisted model for inference rather than retraining the model when the application starts.

## API Architecture

The application follows a simple inference workflow:

1. A client sends product nutritional information through an HTTP request.
2. FastAPI receives the request.
3. Pydantic validates the request structure and input values.
4. The validated data is transformed into a DataFrame using the expected model input columns.
5. The persisted Machine Learning pipeline generates predictions.
6. The API returns the predicted grade and class scores as JSON.

## Input Features

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

The API performs validation to prevent invalid nutritional inputs, including negative nutrient values, saturated fat exceeding total fat, and sugar exceeding total carbohydrates.

At least one nutritional value must be provided.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks API health and model loading status |
| POST | `/predict` | Generates a Nutri-Score grade prediction |

### Interactive API Documentation

Once the API is running, access Swagger UI at:

`http://localhost:8000/docs`

Swagger UI allows users to explore endpoints, submit requests, and inspect API responses interactively.

## Technology Stack

**Machine Learning**
- Python
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

**Deployment**
- Docker
- Docker Desktop
- Docker Compose (available for future orchestration)

## Project Structure

```text
nutriscore-ai-engineering/
│
├── notebook/
│   └── Machine Learning notebooks
│
├── outputs_nutriscore/
│   ├── nutriscore_pipeline.joblib
│   ├── metadata.json
│   └── artifact_checksums.json
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
└── README.md
```

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/nutriscore-ai-engineering.git
cd nutriscore-ai-engineering
```

Replace `YOUR_USERNAME` with your GitHub username.

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

**Latest local test result: 7 passed.**

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

## Engineering Considerations

- The application loads a pre-trained model artifact instead of training during API startup.
- Model input columns are controlled through a shared utility module.
- Input validation helps prevent structurally invalid and inconsistent nutritional data.
- The Docker image packages the application and its runtime dependencies.
- Automated tests verify important API behaviors and validation rules.

## Limitations

- Predictions are based on a supervised model trained on dataset labels.
- The model does not implement the official Nutri-Score calculation rules.
- Returned class scores are model-generated estimates and are not necessarily calibrated probabilities.
- Prediction quality depends on the training data and the distribution of products submitted to the API.
- This application is intended for Machine Learning demonstration and educational purposes.

## Future Improvements

Potential improvements include:

- Add continuous integration using GitHub Actions.
- Add model and API performance monitoring.
- Introduce structured logging and improved error handling.
- Deploy the containerized API to a cloud platform.
- Build a user-friendly frontend for product predictions.

## Author

**YOUR NAME**

Machine Learning | Data Science | AI Engineering

GitHub: `https://github.com/YOUR_USERNAME`

---

*This project demonstrates the transition from a supervised Machine Learning model to a containerized, API-based application..*

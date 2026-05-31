import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_validation():
    # Тест на невалидные данные
    response = client.post("/predict", json=[{"invalid": "data"}])
    assert response.status_code == 422


def test_predict_without_model():
    # Тест на отсутствие загруженной модели
    response = client.post(
        "/predict",
        json=[
            {
                "person_age": 30,
                "person_gender": "M",
                "person_education": "Bachelor",
                "person_income": 50000,
                "person_emp_exp": 5,
                "person_home_ownership": "RENT",
                "loan_amnt": 10000,
                "loan_intent": "PERSONAL",
                "loan_int_rate": 10.0,
                "loan_percent_income": 0.2,
                "cb_person_cred_hist_length": 5,
                "credit_score": 700,
                "previous_loan_defaults_on_file": "No",
            }
        ],
    )
    assert response.status_code == 400
    assert "Модель не загружена" in response.json()["detail"]


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "total_requests" in response.json()

import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
import os
import time
from typing import List, Optional
from collections import defaultdict
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ValidationError
from pathlib import Path

# Настройка логгера
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("mortgage_api")
logger.setLevel(logging.INFO)

# Формат логов
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Вывод в консоль
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Вывод в файл с ротацией
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=3
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Импорт сервисов
from backend.services.csv_service import (
    process_csv_for_prediction,
    save_predictions,
)
from backend.services.model_service import (
    load_model_from_bytes,
    predict,
    predict_proba,
    is_model_loaded,
)
from backend.services.preprocess import FEATURE_COLUMNS, preprocess
from backend.services.validators import (
    validate_binary_file,
    validate_csv_file,
)


class PredictRequest(BaseModel):
    """Модель входных данных клиента."""
    person_age: Optional[float] = None
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: int
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: int
    previous_loan_defaults_on_file: str


app = FastAPI(
    title="Mortgage Prediction Service",
    description="Сервис предсказания одобрения ипотеки",
    version="1.0.0"
)

# Настройка путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = Path("/Users/shukalovichekaterina/Downloads/project/project-lr10/static")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Раздача статики
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_frontend():
    """Отдаёт frontend приложение"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        content={"error": "index.html not found"},
        status_code=404
    )


# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные метрики
metrics = {
    "total_requests": 0,
    "predictions_count": 0,
    "avg_response_time": 0.0,
    "errors": 0
}


@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """Middleware для сбора метрик запросов"""
    start_time = time.time()
    metrics["total_requests"] += 1

    try:
        response = await call_next(request)
        return response
    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"Request error: {str(e)}")
        raise
    finally:
        duration = time.time() - start_time
        if metrics["avg_response_time"] > 0:
            metrics["avg_response_time"] = (metrics["avg_response_time"] + duration) / 2
        else:
            metrics["avg_response_time"] = duration


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Обработчик ошибок валидации Pydantic"""
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Ошибка валидации данных", "errors": exc.errors()}
    )


@app.get("/metrics")
async def get_metrics():
    """Эндпоинт для получения метрик сервиса"""
    return metrics


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    logger.info("Health check requested")
    return {
        "status": "ok",
        "model_loaded": is_model_loaded()
    }


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """Загружает сериализованную ML-модель"""
    logger.info(f"Model upload requested: {file.filename}")

    try:
        validate_binary_file(file.content_type)

        if not file.filename.endswith('.pkl'):
            logger.warning(f"Invalid file extension: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Файл модели должен иметь расширение .pkl"
            )

        file_content = await file.read()
        load_model_from_bytes(file_content)

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "best_mortgage_model.pkl")
        with open(model_path, "wb") as f:
            f.write(file_content)

        logger.info(f"Model '{file.filename}' uploaded successfully")
        return {
            "message": f"Model '{file.filename}' uploaded successfully.",
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при загрузке модели: {str(e)}"
        )


@app.post("/predict/")
async def predict_endpoint(
        request: list[PredictRequest],
):
    """
    Выполняет предсказание для списка клиентов.
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=400,
            detail="Модель не загружена. Загрузите модель через POST /upload-model"
        )

    df = pd.DataFrame([item.model_dump() for item in request])
    processed_df = preprocess(df)

    missing_cols = set(FEATURE_COLUMNS) - set(processed_df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Отсутствуют необходимые признаки: {missing_cols}"
        )

    X = processed_df[FEATURE_COLUMNS]
    predictions = predict(X)
    probabilities = predict_proba(X)  # <-- Получаем вероятности

    result = df.copy()
    result["loan_status"] = predictions
    result["loan_status_label"] = result["loan_status"].map({1: "approved", 0: "rejected"})

    # Добавляем вероятность одобрения (второй элемент в списке вероятностей)
    result["probability_approved"] = [p[1] for p in probabilities]

    return {
        "predictions": result.to_dict(orient="records"),
    }


@app.post("/predict-from-csv/")
async def predict_from_csv(file: UploadFile = File(...)):
    """Выполняет предсказание для CSV-файла"""
    logger.info("CSV prediction request received")

    try:
        if not is_model_loaded():
            logger.warning("CSV prediction requested without loaded model")
            raise HTTPException(
                status_code=400,
                detail="Модель не загружена. Загрузите модель через POST /upload-model"
            )

        validate_csv_file(file.content_type)

        df = pd.read_csv(file.file)
        X, processed_df = process_csv_for_prediction(df)

        predictions = predict(X)
        result = save_predictions(processed_df, predictions)

        metrics["predictions_count"] += len(predictions)
        logger.info(f"CSV prediction completed: {len(predictions)} results")

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during CSV prediction: {str(e)}")
        metrics["errors"] += 1
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке CSV: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
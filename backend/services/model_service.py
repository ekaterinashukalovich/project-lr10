import joblib
import io
import pandas as pd
from fastapi import HTTPException
from backend.settings import settings


def load_model_from_bytes(file_content: bytes) -> None:
    """
    Загружает сериализованную ML-модель из бинарного файла.
    """
    try:
        # Используем joblib для загрузки моделей scikit-learn
        # io.BytesIO создает файловый объект из байтов
        settings.model = joblib.load(io.BytesIO(file_content))
    except Exception as exception:
        raise HTTPException(
            status_code=400,
            detail=f"Error loading model: {str(exception)}",
        )


def predict(data: pd.DataFrame) -> list[int]:
    """
    Выполняет предсказание для переданных данных.
    """
    if settings.model is None:
        raise HTTPException(
            status_code=400,
            detail="Model not uploaded yet.",
        )

    predictions = settings.model.predict(data)
    return [int(prediction) for prediction in predictions]


def predict_proba(data: pd.DataFrame) -> list[list[float]]:
    """
    Возвращает вероятности классов.
    """
    if settings.model is None:
        raise HTTPException(
            status_code=400,
            detail="Model not uploaded yet.",
        )

    if not hasattr(settings.model, "predict_proba"):
        raise HTTPException(
            status_code=400,
            detail="Model does not support probability predictions.",
        )

    probas = settings.model.predict_proba(data)
    return probas.tolist()


def is_model_loaded() -> bool:
    return settings.model is not None

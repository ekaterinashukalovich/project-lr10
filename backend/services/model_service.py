import pickle

import pandas as pd
from fastapi import HTTPException

from settings import settings


def load_model_from_bytes(file_content: bytes) -> None:
    """
    Загружает сериализованную ML-модель из бинарного файла.

    Args:
        file_content (bytes):
            Бинарное содержимое файла модели.

    Raises:
        HTTPException:
            Возникает при ошибке загрузки модели.
    """

    try:
        settings.model = pickle.loads(file_content)
    except Exception as exception:
        raise HTTPException(
            status_code=400,
            detail=f"Error loading model: {str(exception)}",
        )


def predict(data: pd.DataFrame) -> list[int]:
    """
    Выполняет предсказание для переданных данных.

    Args:
        data (pd.DataFrame):
            Датафрейм с признаками клиентов.

    Returns:
        list[int]:
            Список предсказаний модели.

    Raises:
        HTTPException:
            Возникает, если модель не была загружена.
    """

    if settings.model is None:
        raise HTTPException(
            status_code=400,
            detail="Model not uploaded yet.",
        )

    predictions = settings.model.predict(data)

    return [int(prediction) for prediction in predictions]
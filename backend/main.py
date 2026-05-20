import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.csv_service import (
    process_csv_for_prediction,
    save_predictions,
)
from services.model_service import (
    load_model_from_bytes,
    predict,
)
from services.preprocess import FEATURE_COLUMNS, preprocess
from services.validators import (
    validate_binary_file,
    validate_csv_file,
)


class PredictRequest(BaseModel):
    """
    Модель входных данных клиента.
    """

    person_age: float | None = None
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


app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.post("/upload-model/")
async def upload_model(
    file: UploadFile = File(...),
):
    """
    Загружает сериализованную ML-модель.

    Args:
        file (UploadFile):
            Файл модели в формате .pkl.

    Returns:
        dict:
            Сообщение об успешной загрузке модели.
    """

    validate_binary_file(file.content_type)

    file_content = await file.read()

    load_model_from_bytes(file_content)

    return {
        "message": f"Model '{file.filename}' uploaded successfully.",
    }


@app.post("/predict/")
async def predict_endpoint(
    request: list[PredictRequest],
):
    """
    Выполняет предсказание для списка клиентов.

    Args:
        request (list[PredictRequest]):
            Список клиентов.

    Returns:
        dict:
            Результаты предсказания.
    """

    df = pd.DataFrame(
        [
            item.model_dump()
            for item in request
        ]
    )

    processed_df = preprocess(df)

    X = processed_df[FEATURE_COLUMNS]

    predictions = predict(X)

    result = df.copy()

    result["loan_status"] = predictions

    result["loan_status_label"] = result[
        "loan_status"
    ].map(
        {
            1: "approved",
            0: "rejected",
        }
    )

    return {
        "predictions": result.to_dict(
            orient="records",
        ),
    }


@app.post("/predict-from-csv/")
async def predict_from_csv(
    file: UploadFile = File(...),
):
    """
    Выполняет предсказание для CSV-файла.

    Args:
        file (UploadFile):
            CSV-файл с данными клиентов.

    Returns:
        dict:
            Результаты предсказания
            и ROC-AUC при наличии target.
    """

    validate_csv_file(file.content_type)

    try:
        df = pd.read_csv(file.file)

        X, processed_df = process_csv_for_prediction(df)

    except Exception as exception:
        raise HTTPException(
            status_code=400,
            detail=f"Error reading CSV file: {str(exception)}",
        )

    predictions = predict(X)

    result = save_predictions(
        processed_df,
        predictions,
    )

    return result
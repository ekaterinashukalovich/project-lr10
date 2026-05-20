import pandas as pd
from sklearn.metrics import roc_auc_score

from services.preprocess import FEATURE_COLUMNS, TARGET_COLUMN, preprocess


def process_csv_for_prediction(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Подготавливает CSV-данные для предсказания.

    Args:
        df (pd.DataFrame):
            Исходный датафрейм.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            X — признаки для модели,
            df — обработанный датафрейм.
    """

    df = preprocess(df)

    X = df[FEATURE_COLUMNS]

    return X, df


def save_predictions(
    df: pd.DataFrame,
    predictions: list[int],
) -> dict:
    """
    Добавляет предсказания в датафрейм
    и вычисляет ROC-AUC при наличии target-переменной.

    Args:
        df (pd.DataFrame):
            Исходный датафрейм.

        predictions (list[int]):
            Предсказания модели.

    Returns:
        dict:
            Словарь с результатами предсказания.
    """

    df["predicted_loan_status"] = predictions

    df["predicted_loan_status_label"] = df[
        "predicted_loan_status"
    ].map(
        {
            1: "approved",
            0: "rejected",
        }
    )

    rows = [df.columns.tolist()] + df.values.tolist()

    result = {
        "rows": rows,
    }

    if TARGET_COLUMN in df.columns:
        roc_auc = roc_auc_score(
            df[TARGET_COLUMN],
            df["predicted_loan_status"],
        )

        result["roc_auc"] = float(roc_auc)

    return result
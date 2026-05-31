import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "person_emp_exp",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "person_income_log",
    "loan_amnt_log",
]

CATEGORICAL_COLUMNS = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

TARGET_COLUMN = "loan_status"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет предобработку входного датафрейма.

    Этапы обработки:
    - удаление дубликатов;
    - создание логарифмированных признаков;
    - удаление неиспользуемых признаков;
    - преобразование числовых колонок;
    - удаление строк с пропущенными значениями.

    Args:
        df (pd.DataFrame):
            Исходный датафрейм с данными клиентов.

    Returns:
        pd.DataFrame:
            Предобработанный датафрейм.
    """

    df = df.copy()

    df = df.drop_duplicates()

    if "person_income" in df.columns:
        df["person_income_log"] = np.log1p(df["person_income"])

    if "loan_amnt" in df.columns:
        df["loan_amnt_log"] = np.log1p(df["loan_amnt"])

    if "person_age" in df.columns:
        df = df.drop(columns=["person_age"])

    if "person_income" in df.columns:
        df = df.drop(columns=["person_income"])

    if "loan_amnt" in df.columns:
        df = df.drop(columns=["loan_amnt"])

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna()

    return df

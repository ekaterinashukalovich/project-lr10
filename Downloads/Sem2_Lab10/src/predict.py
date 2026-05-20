import joblib
import pandas as pd
import numpy as np
from src.config import MODEL_PATH


class MortgagePredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        self.pipeline = joblib.load(model_path)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Принимает сырые данные, возвращает датафрейм с предсказанием и вероятностью"""
        data = df.copy()
        if "person_income" in data.columns:
            data["person_income_log"] = np.log1p(data["person_income"])
        if "loan_amnt" in data.columns:
            data["loan_amnt_log"] = np.log1p(data["loan_amnt"])

        cols_to_drop = [c for c in ["person_income", "loan_amnt"] if c in data.columns]
        data = data.drop(columns=cols_to_drop, errors="ignore")

        preds = self.pipeline.predict(data)
        probs = self.pipeline.predict_proba(data)[:, 1]

        result = pd.DataFrame({
            "predicted_loan_status": preds,
            "approval_probability": probs
        })
        return pd.concat([df.reset_index(drop=True), result], axis=1)
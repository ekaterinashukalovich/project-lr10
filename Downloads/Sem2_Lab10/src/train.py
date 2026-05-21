import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
git status
from src.config import DATA_PATH, MODEL_PATH, FEATURES_PATH

warnings.filterwarnings("ignore")


def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df[df["person_age"] <= 80]
    df = df[df["person_age"] - df["person_emp_exp"] >= 16]

    df["person_income_log"] = np.log1p(df["person_income"])
    df["loan_amnt_log"] = np.log1p(df["loan_amnt"])
    return df.drop(columns=["person_age", "person_income", "loan_amnt"])


def build_pipeline() -> Pipeline:
    num_cols = [
        "person_emp_exp", "loan_int_rate", "loan_percent_income",
        "cb_person_cred_hist_length", "credit_score",
        "person_income_log", "loan_amnt_log"
    ]
    cat_cols = [
        "person_gender", "person_education", "person_home_ownership",
        "loan_intent", "previous_loan_defaults_on_file"
    ]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def main():
    df = load_and_clean(DATA_PATH)

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(" Сборка пайплайна...")
    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nReport:")
    print(classification_report(y_test, pipeline.predict(X_test), target_names=["Refused", "Approved"]))

    print("Сохранение...")
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)



if __name__ == "__main__":
    main()
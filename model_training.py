from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "credit risk dataset.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "credit_risk_model.joblib"

NUMERIC_FEATURES = [
    "age", "annual_income", "credit_amount", "duration_months",
    "debt_to_income_ratio", "credit_history_length", "credit_to_income_ratio",
    "monthly_payment",
]
CATEGORICAL_FEATURES = [
    "loan_purpose", "employment_status", "housing_status",
    "savings_account", "checking_account", "age_group", "income_group",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DECISION_THRESHOLD = 0.40


def generate_dataset(sample_count: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 81, sample_count)
    annual_income = np.clip(rng.lognormal(np.log(52000), 0.55, sample_count), 15000, 250000)
    credit_amount = np.clip(rng.lognormal(np.log(12000), 0.65, sample_count), 1000, 80000)
    duration = rng.choice([6, 12, 18, 24, 36, 48, 60], sample_count, p=[.04, .12, .12, .27, .25, .14, .06])
    employment = rng.choice(["employed", "self-employed", "unemployed", "retired"], sample_count, p=[.68, .17, .08, .07])
    housing = rng.choice(["own", "rent", "free"], sample_count, p=[.42, .48, .10])
    purpose = rng.choice(["car", "furniture", "education", "business", "vacation", "medical"], sample_count)
    savings = rng.choice(["little", "moderate", "rich", "quite_rich"], sample_count, p=[.42, .35, .16, .07])
    checking = rng.choice(["little", "moderate", "rich"], sample_count, p=[.45, .4, .15])
    history = np.clip((age - 18) * rng.uniform(.25, .55, sample_count) + rng.normal(2, 2, sample_count), 0, 35)
    dti = np.clip((credit_amount / annual_income) * 12 + rng.normal(.16, .09, sample_count), .02, 1.2)
    risk_score = (
        1.8 * dti + .65 * (credit_amount / annual_income) + .45 * (duration >= 48)
        + .7 * np.isin(employment, ["unemployed", "retired"]) + .35 * (housing == "rent")
        + .45 * (savings == "little") + .35 * (checking == "little") - .04 * history
        - .01 * (age - 40) + rng.normal(0, .28, sample_count)
    )
    probability = 1 / (1 + np.exp(-(risk_score - 1.1)))
    defaulted = rng.binomial(1, probability)
    return pd.DataFrame({
        "age": age, "annual_income": annual_income.round(2), "credit_amount": credit_amount.round(2),
        "duration_months": duration, "debt_to_income_ratio": dti.round(4),
        "credit_history_length": history.round(2), "loan_purpose": purpose,
        "employment_status": employment, "housing_status": housing, "savings_account": savings,
        "checking_account": checking, "defaulted": defaulted,
    })


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.rename(columns={
        "income": "annual_income", "duration": "duration_months", "purpose": "loan_purpose",
        "housing": "housing_status", "credit_risk": "defaulted",
    }).copy()
    if "credit_to_income_ratio" not in data:
        data["credit_to_income_ratio"] = data["credit_amount"] / data["annual_income"]
    if "monthly_payment" not in data:
        data["monthly_payment"] = data["credit_amount"] / data["duration_months"]
    if "age_group" not in data:
        data["age_group"] = pd.cut(data["age"], [17, 30, 45, 60, 100], labels=["Young", "Adult", "Middle", "Senior"])
    if "income_group" not in data:
        data["income_group"] = pd.cut(data["annual_income"], [-np.inf, 15000, 30000, 50000, 80000, np.inf], labels=["Low", "Below_Avg", "Average", "Above_Avg", "High"])
    return data


def load_dataset() -> tuple[pd.DataFrame, str]:
    if DATA_PATH.exists():
        data = prepare_features(pd.read_csv(DATA_PATH))
        missing = [column for column in FEATURES + ["defaulted"] if column not in data.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
        return data[FEATURES + ["defaulted"]].copy(), "data/credit risk dataset.csv"
    data = generate_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH, index=False)
    return data, "generated demo data"


def train() -> dict:
    data, source = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data["defaulted"], test_size=.2, random_state=42, stratify=data["defaulted"]
    )
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=400, max_depth=2, learning_rate=0.02, min_child_weight=3,
            subsample=0.9, colsample_bytree=1.0, reg_lambda=5,
            objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=-1,
        )),
    ])
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_high_risk": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_high_risk": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_high_risk": float(f1_score(y_test, predictions, zero_division=0)),
        "decision_threshold": DECISION_THRESHOLD,
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }
    transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    importance = pipeline.named_steps["model"].feature_importances_
    feature_importance = sorted(
        [{"feature": name.replace("numeric__", "").replace("categorical__", ""), "importance": float(value)} for name, value in zip(transformed_names, importance)],
        key=lambda item: item["importance"], reverse=True,
    )[:15]
    artifact = {
        "pipeline": pipeline, "features": FEATURES, "metrics": metrics,
        "feature_importance": feature_importance, "dataset_source": source,
        "dataset_rows": len(data), "positive_rate": float(data["defaulted"].mean()),
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    return artifact


if __name__ == "__main__":
    result = train()
    print(f"Trained on {result['dataset_rows']:,} rows from {result['dataset_source']}.")
    print(f"ROC-AUC: {result['metrics']['roc_auc']:.3f}")
    print(f"Saved model to {ARTIFACT_PATH}")

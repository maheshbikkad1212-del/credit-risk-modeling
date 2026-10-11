# Credit Risk Studio

An end-to-end credit risk modeling project with a scikit-learn training pipeline and Streamlit frontend.

## Quick start (Windows)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py model_training.py
streamlit run streamlit_app.py
```

The browser app opens at `http://localhost:8501`.

## Dataset handoff

The project uses the cleaned `data/credit risk dataset.csv` file. It contains normalized feature names and the training pipeline also supports the original names automatically. If the file is absent, the first training run creates reproducible demo data as a fallback.

`age`, `income`, `credit_amount`, `duration`, `debt_to_income_ratio`, `credit_history_length`, `purpose`, `employment_status`, `housing`, `savings_account`, `checking_account`, `credit_risk`

Use `0` for an approved/non-defaulted outcome and `1` for defaulted. Retrain after replacing the file with `py model_training.py`.

## Project structure


This is an educational demonstration. Credit decisions require legal, compliance, fairness, security, and human-review controls appropriate to the jurisdiction and product.

The cleaned file has a 74% majority-class baseline. Validated models reach approximately 75% held-out accuracy and ROC-AUC around 0.64 on this version, so an honest 80% score cannot be supported with the available predictors. Additional predictive fields or a more consistently defined target are required. Using `defaulted` itself as an input would be target leakage and would not represent real prediction performance.

The production decision threshold is `0.40` rather than `0.50` because high-risk recall is the priority. On the fixed holdout this gives approximately 99.7% recall, 74.4% precision, and 85.2% F1 for the default class. This means the system flags nearly all risky applications, while accepting more false positives.

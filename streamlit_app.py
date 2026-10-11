from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from model_training import ARTIFACT_PATH, FEATURES, prepare_features, train


st.set_page_config(page_title="Credit Risk Studio", page_icon="C", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
:root { --ink:#17212b; --muted:#64727f; --accent:#e4572e; --teal:#147d7e; --paper:#f6f1e9; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stAppViewContainer"] { background: radial-gradient(circle at 85% 5%, #d9eee8 0, transparent 30%), var(--paper); color:var(--ink); }
[data-testid="stSidebar"] { background:#17212b; }
[data-testid="stSidebar"] * { color:#f6f1e9; }
h1, h2, h3 { font-family:'Space Grotesk', sans-serif; letter-spacing:0; }
.hero { padding: 1.2rem 0 1.8rem; border-bottom:1px solid #d7d0c6; margin-bottom:1.5rem; }
.eyebrow { color:var(--accent); font-weight:700; text-transform:uppercase; letter-spacing:.12em; font-size:.75rem; }
.risk-low { color:#147d7e; } .risk-high { color:#c13b26; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_artifact():
    if not ARTIFACT_PATH.exists():
        return train()
    return joblib.load(ARTIFACT_PATH)


artifact = get_artifact()
pipeline = artifact["pipeline"]
st.sidebar.markdown("## Credit Risk Studio")
page = st.sidebar.radio("Workspace", ["Overview", "Risk prediction", "Batch scoring", "Model details"])
st.sidebar.caption(f"Data source: {artifact['dataset_source']}")

if page == "Overview":
    st.markdown('<div class="hero"><div class="eyebrow">Decision intelligence</div><h1>Credit risk, made legible.</h1><p>Explore portfolio signals and turn an application into a transparent risk estimate.</p></div>', unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].metric("Applications", f"{artifact['dataset_rows']:,}")
    cols[1].metric("ROC-AUC", f"{artifact['metrics']['roc_auc']:.3f}")
    cols[2].metric("Observed default rate", f"{artifact['positive_rate']:.1%}")
    st.subheader("What drives the model")
    importance = pd.DataFrame(artifact["feature_importance"]).sort_values("importance")
    st.plotly_chart(px.bar(importance, x="importance", y="feature", orientation="h", color_discrete_sequence=["#e4572e"]), use_container_width=True)
elif page == "Risk prediction":
    st.markdown('<div class="hero"><div class="eyebrow">Single application</div><h1>Assess an applicant</h1><p>Enter the applicant profile to estimate probability of default.</p></div>', unsafe_allow_html=True)
    with st.form("risk_form"):
        left, right = st.columns(2)
        with left:
            age = st.slider("Age", 18, 80, 35)
            annual_income = st.number_input("Annual income", min_value=1000.0, value=55000.0, step=1000.0)
            credit_amount = st.number_input("Credit amount", min_value=500.0, value=12000.0, step=500.0)
            duration_months = st.selectbox("Duration (months)", [6, 12, 18, 24, 36, 48, 60], index=3)
            debt_to_income_ratio = st.slider("Debt-to-income ratio", .01, 1.2, .28, .01)
            credit_history_length = st.slider("Credit history (years)", 0.0, 35.0, 8.0, .5)
        with right:
            loan_purpose = st.selectbox("Loan purpose", ["car", "furniture", "education", "business", "vacation", "medical"])
            employment_status = st.selectbox("Employment status", ["employed", "self-employed", "unemployed", "retired"])
            housing_status = st.selectbox("Housing status", ["own", "rent", "free"])
            savings_account = st.selectbox("Savings account", ["little", "moderate", "rich", "quite_rich"])
            checking_account = st.selectbox("Checking account", ["little", "moderate", "rich"])
        submitted = st.form_submit_button("Evaluate application", type="primary")
    if submitted:
        row = prepare_features(pd.DataFrame([locals()]))[FEATURES]
        probability = float(pipeline.predict_proba(row)[0, 1])
        threshold = artifact["metrics"].get("decision_threshold", .5)
        label = "High risk" if probability >= threshold else "Moderate risk" if probability >= .25 else "Low risk"
        st.divider()
        result_left, result_right = st.columns([1, 2])
        result_left.metric("Estimated default probability", f"{probability:.1%}")
        result_left.markdown(f"### {label}")
        result_right.plotly_chart(px.bar(pd.DataFrame({"Outcome": ["Default risk", "Remaining"], "Probability": [probability, 1-probability]}), x="Probability", y="Outcome", orientation="h", range_x=[0, 1], color="Outcome", color_discrete_map={"Default risk":"#e4572e", "Remaining":"#d9eee8"}), use_container_width=True)
elif page == "Batch scoring":
    st.markdown('<div class="hero"><div class="eyebrow">Portfolio workflow</div><h1>Score a file</h1><p>Upload a CSV containing the model inputs and download scored applications.</p></div>', unsafe_allow_html=True)
    st.caption("Required columns: " + ", ".join(FEATURES))
    uploaded = st.file_uploader("Upload applications CSV", type="csv")
    if uploaded:
        batch = prepare_features(pd.read_csv(uploaded))
        missing = [column for column in FEATURES if column not in batch.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            scored = batch.copy()
            scored["default_probability"] = pipeline.predict_proba(batch[FEATURES])[:, 1]
            threshold = artifact["metrics"].get("decision_threshold", .5)
            scored["risk_band"] = pd.cut(scored["default_probability"], [-.01, .25, threshold, 1], labels=["Low", "Moderate", "High"])
            st.dataframe(scored, use_container_width=True)
            st.download_button("Download scored CSV", scored.to_csv(index=False), "credit_risk_predictions.csv", "text/csv")
else:
    st.markdown('<div class="hero"><div class="eyebrow">Model card</div><h1>How the model performs</h1><p>A compact view of the current training run and its operating characteristics.</p></div>', unsafe_allow_html=True)
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{artifact['metrics']['accuracy']:.1%}")
    cols[1].metric("Precision", f"{artifact['metrics']['precision_high_risk']:.1%}")
    cols[2].metric("High-risk recall", f"{artifact['metrics']['recall_high_risk']:.1%}")
    cols[3].metric("High-risk F1", f"{artifact['metrics']['f1_high_risk']:.1%}")
    cols[4].metric("ROC-AUC", f"{artifact['metrics']['roc_auc']:.3f}")
    report = pd.DataFrame(artifact["metrics"]["classification_report"]).T.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
    st.dataframe(report.style.format("{:.3f}"), use_container_width=True)
    st.caption(f"High-risk decisions use a {artifact['metrics']['decision_threshold']:.0%} probability threshold to prioritize detecting defaults.")
    st.info("This demonstration model is trained on synthetic data unless a compatible CSV is placed at data/credit_risk_dataset.csv. It must not be used as the sole basis for real lending decisions.")

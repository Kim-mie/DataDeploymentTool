"""Streamlit churn predictor.  Run:  streamlit run app.py"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")

HERE = os.path.dirname(os.path.abspath(__file__))


LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "decision_tree": "Decision Tree",
    "deep_neural_net": "Neural Network (ANN)",
}
MODELS_DIR = os.path.join(HERE, "models")


@st.cache_resource
def load_all():
    meta = json.load(open(os.path.join(MODELS_DIR, "meta.json")))
    predictors = {}
    for key in LABELS:
        if key == "deep_neural_net":
            try:  # optional: needs tensorflow installed
                import tensorflow as tf
                pre = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
                net = tf.keras.models.load_model(
                    os.path.join(MODELS_DIR, "deep_neural_net.keras"))
                predictors[key] = lambda X, pre=pre, net=net: float(
                    net.predict(pre.transform(X), verbose=0).ravel()[0])
            except Exception:
                continue
        else:
            path = os.path.join(MODELS_DIR, f"{key}.joblib")
            if os.path.exists(path):
                pipe = joblib.load(path)
                predictors[key] = lambda X, pipe=pipe: float(
                    pipe.predict_proba(X)[0, 1])
    return predictors, meta["columns"], meta["metrics"]


predictors, columns, metrics = load_all()

st.title("📉 Customer Churn Predictor")
st.caption("Enter a customer's details to estimate their risk of leaving.")

choice = st.selectbox(
    "Model", list(predictors), format_func=lambda k: LABELS[k],
    help="Pick which trained model makes the prediction.")

YN = ["Yes", "No"]
NO_NET = ["Yes", "No", "No internet service"]

with st.form("customer"):
    st.subheader("Account")
    c1, c2 = st.columns(2)
    tenure = c1.slider("Tenure (months)", 0, 72, 12)
    monthly = c2.number_input("Monthly charges", 18.0, 120.0, 70.0, step=0.5)
    contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = c2.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = c1.selectbox("Paperless billing", YN)

    st.subheader("Services")
    c3, c4 = st.columns(2)
    phone = c3.selectbox("Phone service", YN)
    multi = c4.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet = c3.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    sec = c4.selectbox("Online security", NO_NET)
    backup = c3.selectbox("Online backup", NO_NET)
    device = c4.selectbox("Device protection", NO_NET)
    tech = c3.selectbox("Tech support", NO_NET)
    tv = c4.selectbox("Streaming TV", NO_NET)
    movies = c3.selectbox("Streaming movies", NO_NET)

    st.subheader("Customer")
    c5, c6 = st.columns(2)
    gender = c5.selectbox("Gender", ["Female", "Male"])
    senior = c6.selectbox("Senior citizen", ["No", "Yes"])
    partner = c5.selectbox("Partner", YN)
    dependents = c6.selectbox("Dependents", YN)

    go = st.form_submit_button("Predict churn risk", type="primary")

if go:
    row = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": multi,
        "InternetService": internet, "OnlineSecurity": sec,
        "OnlineBackup": backup, "DeviceProtection": device,
        "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        # estimated: we don't ask for lifetime billing
        "TotalCharges": tenure * monthly,
    }
    X_new = pd.DataFrame([row])[columns]
    p = predictors[choice](X_new)

    if p < 0.35:
        tier, color = "LOW", "green"
        action = "Healthy. No action needed."
    elif p < 0.65:
        tier, color = "MEDIUM", "orange"
        action = "Watch closely. Consider a check-in call or small loyalty offer."
    else:
        tier, color = "HIGH", "red"
        action = ("Likely to leave. Prioritise retention: offer a longer "
                  "contract discount or a service review.")

    st.divider()
    st.caption(f"Model: {LABELS[choice]}")
    st.metric("Churn probability", f"{p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))
    st.markdown(f"### Risk level: :{color}[{tier}]")
    st.info(action)

    with st.expander("What do all the models say about this customer?"):
        st.dataframe(pd.DataFrame({
            "Model": [LABELS[k] for k in predictors],
            "Churn probability": [f"{f(X_new):.0%}" for f in predictors.values()],
        }), hide_index=True, width="stretch")

with st.expander("Model performance (held-out test set)"):
    st.dataframe(
        pd.DataFrame(metrics).T.rename(index=LABELS)
        .sort_values("roc_auc", ascending=False),
        width="stretch")
    st.caption("All models are class-balanced. Trained on the IBM Telco "
               "churn dataset; 70/30 stratified split.")

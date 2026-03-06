import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model
from sklearn.datasets import load_breast_cancer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Hospital AI Cancer Dashboard",
    page_icon="🧬",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "login" not in st.session_state:
    st.session_state.login = False

# ---------------- LOGIN PAGE ----------------
if not st.session_state.login:

    st.title("🏥 Personalized AI Cancer Risk System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "1234":
            st.session_state.login = True
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# ---------------- LOAD MODELS ----------------
try:
    model = joblib.load("model/trained_model.pkl")
    encoder = load_model("model/encoder.h5")
    scaler = joblib.load("model/scaler.pkl")
except:
    st.error("Model files not found. Run train_model.py first.")
    st.stop()

# ---------------- LOAD DATASET ----------------
data = load_breast_cancer()
features = data.feature_names

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧬 Patient AI Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Patient Prediction",
        "Patient History",
        "Feature Importance",
        "Dataset Prediction"
    ]
)

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":

    st.title("📊 Patient AI Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Accuracy", "94%")
    col2.metric("ROC-AUC", "0.98")

    # Patient count from CSV
    if os.path.exists("patient_history.csv"):
        history_df = pd.read_csv("patient_history.csv")
        patient_count = len(history_df)
    else:
        patient_count = 0

    col3.metric("Patients Analysed", patient_count)

    if os.path.exists("patient_history.csv"):

        df = pd.read_csv("patient_history.csv")

        fig = px.line(df, y="risk", title="Cancer Risk Timeline")
        st.plotly_chart(fig)

# ---------------- PATIENT PREDICTION ----------------
elif menu == "Patient Prediction":

    st.title("🔬 Patient Cancer Risk Assessment")

    patient_id = st.text_input("Patient Name")

    col1, col2 = st.columns(2)

    input_values = []

    sample = data.data[0]

    for i, f in enumerate(features):

        if i < 15:
            val = col1.number_input(
                f,
                value=float(sample[i]),
                step=0.1
            )
        else:
            val = col2.number_input(
                f,
                value=float(sample[i]),
                step=0.1
            )

        input_values.append(val)

    if st.button("Predict Risk"):

        arr = np.array(input_values).reshape(1, -1)

        arr_scaled = scaler.transform(arr)

        emb = encoder.predict(arr_scaled)

        prob = model.predict_proba(emb)[0][1]

        risk_percent = prob * 100

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percent,
            title={'text': "Cancer Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 45], 'color': "green"},
                    {'range': [45, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "red"}
                ]
            }
        ))

        st.plotly_chart(fig)

        st.subheader("Prediction Result")

        if prob >= 0.75:

            st.markdown("""
            <div style="background-color:#ff4b4b;padding:20px;border-radius:10px;
            text-align:center;color:white;font-size:24px;">
            🔴 HIGH RISK
            </div>
            """, unsafe_allow_html=True)

        elif prob >= 0.45:

            st.markdown("""
            <div style="background-color:#ffa500;padding:20px;border-radius:10px;
            text-align:center;color:white;font-size:24px;">
            🟡 MEDIUM RISK
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div style="background-color:#00c853;padding:20px;border-radius:10px;
            text-align:center;color:white;font-size:24px;">
            🟢 LOW RISK
            </div>
            """, unsafe_allow_html=True)

        st.write("Prediction Probability:", round(prob, 3))

        # -------- SAVE DATA PERMANENTLY --------
        data_to_save = pd.DataFrame([{
            "patient": patient_id,
            "risk": prob
        }])

        file_name = "patient_history.csv"

        if os.path.exists(file_name):
            data_to_save.to_csv(file_name, mode="a", header=False, index=False)
        else:
            data_to_save.to_csv(file_name, index=False)

# ---------------- PATIENT HISTORY ----------------
elif menu == "Patient History":

    st.title("📋 Patient Prediction History")

    if os.path.exists("patient_history.csv"):

        df = pd.read_csv("patient_history.csv")

        st.dataframe(df)

        fig = px.bar(df, x="patient", y="risk", title="Patient Risk Levels")
        st.plotly_chart(fig)

    else:
        st.info("No predictions yet")

# ---------------- FEATURE IMPORTANCE ----------------
elif menu == "Feature Importance":

    st.title("🧬 Model Feature Importance")

    importances = model.feature_importances_

    df = pd.DataFrame({
        "Feature": [f"Embedding_{i}" for i in range(len(importances))],
        "Importance": importances
    })

    fig = px.bar(
        df.sort_values("Importance", ascending=False).head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Influential Features"
    )

    st.plotly_chart(fig)

# ---------------- DATASET PREDICTION ----------------
elif menu == "Dataset Prediction":

    st.title("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV")

    if file:

        df = pd.read_csv(file, encoding="latin1")

        st.write("Dataset Preview", df.head())

        try:

            X = scaler.transform(df)

            emb = encoder.predict(X)

            preds = model.predict_proba(emb)[:,1]

            df["risk"] = preds

            st.dataframe(df)

            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                "prediction_results.csv"
            )

        except:
            st.error("Dataset format incorrect. Please match training features.")

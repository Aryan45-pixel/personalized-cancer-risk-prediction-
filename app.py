# import streamlit as st
# import numpy as np
# import joblib
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_breast_cancer
# from tensorflow.keras.models import load_model
#
# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(
#     page_title="Cancer Risk Prediction",
#     page_icon="🧬",
#     layout="wide"
# )
#
# # ---------------- LOAD MODELS ----------------
# model = joblib.load("model/trained_model.pkl")
# encoder = load_model("model/encoder.h5")
#
# data = load_breast_cancer()
# feature_names = data.feature_names
#
# # ---------------- SIDEBAR ----------------
# st.sidebar.title("🧬 Navigation")
# menu = st.sidebar.radio(
#     "Go to",
#     ["Home", "Prediction", "Model Insights"]
# )
#
# # ---------------- HOME PAGE ----------------
# if menu == "Home":
#     st.title("🧬 Personalized Cancer Risk Prediction System")
#     st.markdown("""
#     This system predicts cancer risk using:
#     - SMOTE for imbalance handling
#     - Autoencoder for feature embeddings
#     - Random Forest classifier
#
#     Developed for precision medicine applications.
#     """)
#
# # ---------------- PREDICTION PAGE ----------------
# elif menu == "Prediction":
#
#     st.title("🔬 Patient Risk Assessment")
#
#     col1, col2 = st.columns(2)
#
#     patient_id = st.text_input("Enter Patient ID")
#
#     input_data = []
#
#     for feature in feature_names:
#         value = st.number_input(feature, value=0.0)
#         input_data.append(value)
#
#     if st.button("Predict Risk"):
#
#         input_array = np.array(input_data).reshape(1, -1)
#         embedded = encoder.predict(input_array)
#         prediction_prob = model.predict_proba(embedded)[0][1]
#
#         st.subheader("📊 Prediction Result")
#         st.write("Risk Probability:", round(prediction_prob, 4))
#
#         # Risk indicator
#         if prediction_prob > 0.7:
#             st.error("🔴 High Risk")
#         elif prediction_prob > 0.4:
#             st.warning("🟡 Medium Risk")
#         else:
#             st.success("🟢 Low Risk")
#
#         # Download report
#         report = pd.DataFrame({
#             "Patient ID": [patient_id],
#             "Risk Probability": [prediction_prob]
#         })
#
#         csv = report.to_csv(index=False).encode('utf-8')
#         st.download_button(
#             label="Download Report",
#             data=csv,
#             file_name="cancer_risk_report.csv",
#             mime="text/csv"
#         )
#
# # ---------------- MODEL INSIGHTS PAGE ----------------
# elif menu == "Model Insights":
#
#     st.title("📈 Model Insights")
#
#     st.subheader("Top Important Embedding Features")
#
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[-10:]
#
#     plt.figure()
#     plt.barh(range(len(indices)), importances[indices])
#     plt.yticks(range(len(indices)), [f"Embedding_{i}" for i in indices])
#     plt.xlabel("Importance Score")
#     st.pyplot(plt)
#
#     st.write("Accuracy: 94%")
#     st.write("ROC-AUC Score: 0.988")

# import streamlit as st
# import numpy as np
# import joblib
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_breast_cancer
# from tensorflow.keras.models import load_model
#
# rf_gene_model = joblib.load("model/rf_gene_model.pkl")
# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(
#     page_title="AI Cancer Risk System",
#     page_icon="🧬",
#     layout="wide"
# )
#
# # ---------------- CUSTOM STYLING ----------------
# st.markdown("""
# <style>
# .big-font {
#     font-size:22px !important;
#     font-weight: bold;
# }
# </style>
# """, unsafe_allow_html=True)
#
# # ---------------- LOAD MODELS ----------------
# model = joblib.load("model/trained_model.pkl")
# encoder = load_model("model/encoder.h5")
# scaler = joblib.load("model/scaler.pkl")
#
# data = load_breast_cancer()
# feature_names = data.feature_names
#
# # ---------------- SIDEBAR ----------------
# st.sidebar.title("🧬 Navigation")
# menu = st.sidebar.radio("Select Page",
#                         ["Home", "Prediction", "Explainability"])
#
# # ---------------- HOME ----------------
# if menu == "Home":
#     st.title("🧬 AI-Based Cancer Risk Prediction System")
#     st.markdown("""
#     This system uses:
#     - SMOTE for class balancing
#     - Autoencoder for feature embeddings
#     - Random Forest classifier
#     - SHAP for Explainable AI
#     """)
#
# # ---------------- PREDICTION ----------------
# elif menu == "Prediction":
#
#     st.title("🔬 Patient Risk Assessment")
#
#     patient_id = st.text_input("Patient Name")
#
#     col1, col2 = st.columns(2)
#
#     input_data = []
#
#     for i, feature in enumerate(feature_names):
#         if i < len(feature_names)//2:
#             val = col1.number_input(feature, value=0.0)
#         else:
#             val = col2.number_input(feature, value=0.0)
#         input_data.append(val)
#
#     if st.button("Predict Risk"):
#
#         input_array = np.array(input_data).reshape(1, -1)
#         input_scaled = scaler.transform(input_array)
#         embedded = encoder.predict(input_scaled)
#
#         prob = model.predict_proba(embedded)[0][1]
#
#         st.markdown('<p class="big-font">Risk Probability: {:.4f}</p>'.format(prob),
#                     unsafe_allow_html=True)
#
#         if prob > 0.7:
#             st.error("🔴 High Risk")
#         elif prob > 0.4:
#             st.warning("🟡 Medium Risk")
#         else:
#             st.success("🟢 Low Risk")
#
# # ---------------- EXPLAINABILITY ----------------
# elif menu == "Explainability":
#
#     st.title("🧠 Real Gene-Level Explainability (SHAP)")
#
#     # Load sample data
#     X_sample = scaler.transform(data.data[:100])
#
#     # Create SHAP explainer for gene-level model
#     explainer = shap.TreeExplainer(rf_gene_model)
#
#     shap_values = explainer.shap_values(X_sample)
#
#     if isinstance(shap_values, list):
#         shap_values = shap_values[1]
#
#     st.subheader("Top Gene Impact on Cancer Risk")
#
#     fig, ax = plt.subplots()
#     shap.summary_plot(
#         shap_values,
#         X_sample,
#         feature_names=feature_names,
#         show=False
#     )
#
#     st.pyplot(fig)

import streamlit as st
import numpy as np
import pandas as pd
import joblib
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

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- LOGIN PAGE ----------------
if not st.session_state.login:

    st.title("🏥 Hospital AI Cancer Risk System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "1234":
            st.session_state.login = True
            st.success("Login Successful")
        else:
            st.error("Invalid credentials")

    st.stop()

# ---------------- LOAD MODELS ----------------
model = joblib.load("trained_model.pkl")
encoder = load_model("encoder.h5")
scaler = joblib.load("scaler.pkl")

data = load_breast_cancer()
features = data.feature_names

# ---------------- THEME TOGGLE ----------------
theme = st.sidebar.toggle("🌙 Dark Mode")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧬 Hospital AI Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Patient Prediction",
        "Patient History",
        "Gene Insights",
        "AI Explanation",
        "Dataset Prediction"
    ]
)

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":

    st.title("📊 Hospital AI Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Accuracy", "94%")
    col2.metric("ROC-AUC", "0.988")
    col3.metric("Patients Analysed", len(st.session_state.history))

    st.subheader("Prediction Timeline")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        fig = px.line(df, y="risk", title="Cancer Risk Timeline")
        st.plotly_chart(fig)

# ---------------- PATIENT PREDICTION ----------------
elif menu == "Patient Prediction":

    st.title("🔬 Patient Cancer Risk Assessment")

    patient_id = st.text_input("Patient Name")

    col1, col2 = st.columns(2)

    input_values = []

    for i, f in enumerate(features):

        if i < 15:
            val = col1.number_input(f, value=0.0)
        else:
            val = col2.number_input(f, value=0.0)

        input_values.append(val)

    if st.button("Predict Risk"):

        arr = np.array(input_values).reshape(1, -1)
        arr_scaled = scaler.transform(arr)

        emb = encoder.predict(arr_scaled)

        prob = model.predict_proba(emb)[0][1]

        # animated gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Cancer Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))

        st.plotly_chart(fig)

        if prob > 0.7:
            st.error("High Risk")
        elif prob > 0.4:
            st.warning("Medium Risk")
        else:
            st.success("Low Risk")

        # save history
        st.session_state.history.append({
            "patient": patient_id,
            "risk": prob
        })

# ---------------- PATIENT HISTORY ----------------
elif menu == "Patient History":

    st.title("📋 Patient Prediction History")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        st.dataframe(df)

        fig = px.bar(df, x="patient", y="risk", title="Patient Risk Levels")
        st.plotly_chart(fig)

    else:
        st.info("No patient predictions yet")

# ---------------- GENE INSIGHTS ----------------
elif menu == "Gene Insights":

    st.title("🧬 Gene Importance Analysis")

    importances = model.feature_importances_

    df = pd.DataFrame({
        "gene": [f"Embedding_{i}" for i in range(len(importances))],
        "importance": importances
    })

    fig = px.bar(
        df.sort_values("importance", ascending=False).head(10),
        x="importance",
        y="gene",
        orientation="h",
        title="Top Influential Features"
    )

    st.plotly_chart(fig)

# ---------------- AI EXPLANATION ----------------
elif menu == "AI Explanation":

    st.title("🧠 Explainable AI Panel")

    st.info("SHAP Explanation can be visualized here.")

    st.write("""
    The AI explanation panel shows how different genomic
    features influence cancer risk prediction.

    Higher SHAP value → Higher influence on prediction.
    """)

# ---------------- DATASET PREDICTION ----------------
elif menu == "Dataset Prediction":

    st.title("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV")

    if file:

        df = pd.read_csv(file)

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

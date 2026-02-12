import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="COVID Vaccine Cardiac Risk Dashboard",
    layout="wide"
)

st.title("AI-Based Post-COVID Vaccination Cardiac Risk Dashboard")

# ---------------- LOAD MODEL ----------------
model = joblib.load("../model/cardiac_model.pkl")
encoders = joblib.load("../model/encoders.pkl")

# ---------------- DATA SOURCE ----------------
st.sidebar.subheader("Dataset Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload New Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom Dataset Loaded ")
else:
    df = pd.read_csv("../data/covid_vaccine_cardiac_dataset.csv")


page = st.sidebar.radio(
    "Navigation",
    ["Patient Risk Prediction", "Model Dashboard", "Dataset Overview"]
)

if page == "Patient Risk Prediction":

    st.header("Patient Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 100)
        sex = st.selectbox("Sex", encoders["sex"].classes_)
        covid_positive = st.selectbox(
            "Prior COVID Infection",
            encoders["covid_positive"].classes_
        )
        vaccination_status = st.selectbox(
            "Vaccination Status",
            encoders["vaccination_status"].classes_
        )
        vaccine_name = st.selectbox(
            "Vaccine Name",
            encoders["vaccine_name"].classes_
        )

    with col2:
        number_of_doses = st.slider("Number of Doses", 0, 5)
        diabetes = st.selectbox("Diabetes", [0,1])
        hypertension = st.selectbox("Hypertension", [0,1])
        prior_cvd = st.selectbox("Prior Cardiovascular Disease", [0,1])

    if st.button("Predict Cardiac Risk"):

        input_data = pd.DataFrame([{
            "age": age,
            "sex": encoders["sex"].transform([sex])[0],
            "covid_positive": encoders["covid_positive"].transform([covid_positive])[0],
            "vaccination_status": encoders["vaccination_status"].transform([vaccination_status])[0],
            "vaccine_name": encoders["vaccine_name"].transform([vaccine_name])[0],
            "number_of_doses": number_of_doses,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "prior_cvd": prior_cvd
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        st.subheader("Risk Report")

        if prediction == 1:
            st.error(f"⚠ High Cardiac Arrest Risk ({probability:.2f}%)")
        else:
            st.success(f"Low Cardiac Arrest Risk ({probability:.2f}%)")
   
    st.subheader("Vaccine Distribution Overview")

    if "vaccine_name" in df.columns:

        vaccine_counts = df["vaccine_name"].value_counts()

        fig_vaccine, ax_vaccine = plt.subplots()

        ax_vaccine.pie(
            vaccine_counts,
            labels=vaccine_counts.index,
            autopct="%1.1f%%"
        )

        ax_vaccine.set_title("Distribution of Vaccine Types")

        st.pyplot(fig_vaccine)

    else:
        st.warning("vaccine_name column not found in dataset")

elif page == "Model Dashboard":

    st.header("AI Model Performance Dashboard (Random Forest)")


    X = df.drop(columns=["cardiac_arrest","state","time_to_event","event"])
    y = df["cardiac_arrest"]

    X = X.fillna(method="ffill")


    for col in encoders:

        known = set(encoders[col].classes_)

        X[col] = X[col].fillna(encoders[col].classes_[0])
        X[col] = X[col].apply(
            lambda x: x if x in known else encoders[col].classes_[0]
        )

        X[col] = encoders[col].transform(X[col])


    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]


    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y, y_pred)

    st.subheader("Model Accuracy")
    st.metric("Random Forest Accuracy", f"{acc:.3f}")


    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)


    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    st.subheader("ROC Curve")

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve (AUC = {roc_auc:.2f})")

    st.pyplot(fig_roc)

   
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y, y_prob)

    st.subheader("Precision-Recall Curve")

    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")

    st.pyplot(fig_pr)


    st.subheader("Feature Importance")

    importances = model.feature_importances_

    fig_imp, ax_imp = plt.subplots()
    ax_imp.bar(X.columns, importances)
    plt.xticks(rotation=90)

    st.pyplot(fig_imp)


elif page == "Dataset Overview":

    st.header("Dataset Insights")

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Cardiac Arrest Distribution")

    fig3, ax3 = plt.subplots()
    df["cardiac_arrest"].value_counts().plot(kind="bar", ax=ax3)

    st.pyplot(fig3)

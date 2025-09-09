# app.py
# ===============================
# Fraud Detection App (Streamlit)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from imblearn.over_sampling import SMOTE
import joblib

# ===============================
# Sidebar
# ===============================
st.sidebar.title("Fraud Detection App")
st.sidebar.markdown("Demo project for Genpact (Credit Card Fraud Dataset)")

menu = st.sidebar.radio("Navigation", ["Upload Data", "EDA", "Model Training", "Predict Transaction"])

# ===============================
# Upload dataset
# ===============================
if menu == "Upload Data":
    st.title("Upload Dataset")
    uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.success("✅ Dataset uploaded successfully!")
        st.write(df.head())
    else:
        st.info("Upload Kaggle's Credit Card Fraud dataset (creditcard.csv)")

# ===============================
# EDA
# ===============================
elif menu == "EDA":
    st.title("Exploratory Data Analysis")
    if 'df' not in st.session_state:
        st.warning("Please upload dataset first!")
    else:
        df = st.session_state['df']

        # Class distribution
        st.subheader("Class Distribution")
        st.write(df['Class'].value_counts())
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Class', data=df, ax=ax1)
        st.pyplot(fig1)

        # Amount distribution
        st.subheader("Transaction Amount Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['Amount'], bins=100, log_scale=(False, True), ax=ax2)
        st.pyplot(fig2)

        # Log transform Amount
        df['amount_log'] = np.log1p(df['Amount'])
        st.subheader("Log-transformed Amount Distribution")
        fig3, ax3 = plt.subplots()
        sns.histplot(df['amount_log'], bins=100, ax=ax3)
        st.pyplot(fig3)

        # Time -> Hour of Day
        df['hour'] = (df['Time'] // 3600) % 24
        st.subheader("Fraud by Hour of Day")
        fig4, ax4 = plt.subplots()
        sns.countplot(x='hour', data=df, hue='Class', ax=ax4)
        st.pyplot(fig4)

        st.session_state['df'] = df

# ===============================
# Model Training
# ===============================
elif menu == "Model Training":
    st.title("Model Training & Evaluation")
    if 'df' not in st.session_state:
        st.warning("Please upload dataset first!")
    else:
        df = st.session_state['df']

        # Feature engineering
        df['amount_log'] = np.log1p(df['Amount'])
        df['hour'] = (df['Time'] // 3600) % 24
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)

        features = [col for col in df.columns if col not in ['Time','Amount','Class','hour']]
        X = df[features]
        y = df['Class']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )

        # SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Train model
        st.info("Training XGBoost model...")
        model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)
        model.fit(X_train_res, y_train_res)

        # Save model & scaler
        joblib.dump(model, "fraud_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        st.success("✅ Model trained and saved!")

        # Evaluation
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, digits=4))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig5, ax5 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax5)
        st.pyplot(fig5)
        auc_score = roc_auc_score(y_test, y_prob)
        st.write(f"ROC AUC Score: {auc_score:.4f}")

        # Precision-Recall Curve
        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        fig6, ax6 = plt.subplots()
        ax6.plot(recall, precision, label=f"PR AUC={pr_auc:.2f}")
        ax6.set_xlabel("Recall")
        ax6.set_ylabel("Precision")
        ax6.legend()
        st.pyplot(fig6)

        # Feature importance
        st.subheader("Feature Importance (XGBoost)")
        fig7, ax7 = plt.subplots()
        xgb.plot_importance(model, max_num_features=10, ax=ax7)
        st.pyplot(fig7)

# ===============================
# Predict Transaction
# ===============================
elif menu == "Predict Transaction":
    st.title("Predict Fraud for New Transaction")

    try:
        model = joblib.load("fraud_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except:
        st.error("⚠️ Train the model first under 'Model Training' tab.")
        st.stop()

    st.subheader("Enter Transaction Details")

    amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0)
    time_sec = st.number_input("Time since first txn (seconds)", min_value=0, step=1000)

    # Derived features
    amount_log = np.log1p(amount)
    hour = (time_sec // 3600) % 24
    hour_sin = np.sin(2 * np.pi * hour/24)
    hour_cos = np.cos(2 * np.pi * hour/24)

    # Fill in random V1...V28 (demo only)
    v_features = [st.slider(f"V{i}", -5.0, 5.0, 0.0, 0.1) for i in range(1,29)]

    if st.button("Predict"):
        input_data = pd.DataFrame([[amount_log, hour_sin, hour_cos] + v_features],
                                  columns=['amount_log','hour_sin','hour_cos'] + [f'V{i}' for i in range(1,29)])
        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]

        if prob > 0.5:
            st.error(f"🚨 Fraud Probability: {prob:.2%}")
        else:
            st.success(f"✅ Legitimate Transaction. Fraud Probability: {prob:.2%}")

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
# Upload dataset (from GitHub instead of frontend upload)
# ===============================
if menu == "Upload Data":
    st.title("Load Dataset from GitHub")

    # Raw GitHub URLs of your 6 dataset parts
    urls = [
        "https://raw.githubusercontent.com/tavishee/Fraud-Detection-Tool/main/your_dataset_part1.csv",
        "https://raw.githubusercontent.com/tavishee/Fraud-Detection-Tool/main/your_dataset_part2.csv",
        "https://raw.githubusercontent.com/tavishee/Fraud-Detection-Tool/main/your_dataset_part3.csv",
        "https://raw.githubusercontent.com/tavishee/Fraud-Detection-Tool/main/your_dataset_part4.csv",
        "https://raw.githubusercontent.com/tavishee/Fraud-Detection-Tool/main/your_dataset_part5.csv",
        "https://raw.githubusercontent.com/tavishee/Fraud-Detection-Tool/main/your_dataset_part6.csv",
    ]

    try:
        df_parts = [pd.read_csv(url) fo_]()_

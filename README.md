# 🛡️ Credit Card Fraud Detection System

An End-to-End Machine Learning project that detects fraudulent credit card transactions using multiple ML and Deep Learning models.

---

## 🚀 Features

- Data Preprocessing & Feature Engineering
- Exploratory Data Analysis (EDA)
- Fraud Detection using:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Neural Network
- SMOTE for Imbalanced Data
- Model Evaluation (F1, ROC-AUC, Recall)
- Real-Time Prediction System
- Interactive Streamlit Dashboard
- Batch Transaction Scanner

---

## 🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow
- Streamlit
- Matplotlib
- Seaborn

---

## 📊 Dataset

Kaggle Credit Card Fraud Detection Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## ▶️ Run Project

### Install dependencies

```bash
pip install -r requirements.txt

Run preprocessing
python 1_data_preprocessing.py

Run EDA
python 2_eda.py

Train models
python 3_train_models.py

Evaluate models
python 4_evaluate_models.py

Launch Streamlit App
streamlit run app.py

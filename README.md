# ML-AI
predictive and deep learning for business use case

Customer Churn Prediction (Telco Dataset)
This project predicts whether a customer will churn (leave the service) based on demographic and usage data.

🔧 Tools & Libraries:

Python (pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn)
Jupyter Notebook

📂 Dataset: Telco Customer Churn Dataset – Kaggle

7,043 customers

Features: demographics, contract type, monthly charges, tenure, services

Target: Churn (Yes/No)

📊 Workflow:

Data Cleaning & Preprocessing (missing values, encoding, scaling)

Exploratory Data Analysis (EDA) – churn distribution, correlations

Feature Engineering – categorical encoding, normalization

Model Training – Logistic Regression, Random Forest, XGBoost

Model Evaluation – Accuracy, Precision, Recall, F1-score, ROC-AUC

Business Insights – contract type, tenure, monthly charges impact churn

📈 Results (XGBoost):
Accuracy: 78%

ROC-AUC: 0.82

Key Drivers: contract type, monthly charges, tenure

✅ Business Impact:

This model enables customer success teams to identify at-risk customers and reduce churn by targeting retention strategies such as discounts or contract upgrades.

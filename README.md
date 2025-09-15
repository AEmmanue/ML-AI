# ML-AI
Portfolio of Machine Learning & AI projects (Classification, NLP, Time-Series)


1️⃣ Customer Churn Prediction – Classification (Telco Dataset)

Objective: Predict whether a customer will leave (churn) based on demographics and usage behavior.
Tools & Libraries: Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn

🔹 Workflow:

Data cleaning & feature engineering (contract type, tenure, monthly charges)

Baseline model (Logistic Regression) → advanced models (Random Forest, XGBoost)

Performance evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)

Feature importance analysis

🔹 Results:

Logistic Regression: Accuracy = 79%, ROC-AUC = 0.84

Random Forest: Accuracy = 79%, ROC-AUC = 0.82

XGBoost: Accuracy = 78%, ROC-AUC = 0.82

Key drivers of churn: short contract length, high monthly charges, low tenure

🔹 Business Value:
Helps identify at-risk customers for targeted retention campaigns (e.g., discounts, loyalty programs).

2️⃣ Airbnb Review Sentiment Analysis – NLP (Text Classification)

Objective: Classify Airbnb guest reviews into positive, neutral, or negative sentiment to uncover customer experience insights.
Tools & Libraries: Python, pandas, nltk, scikit-learn, TensorFlow/Keras, seaborn, matplotlib

🔹 Workflow:

Text preprocessing (tokenization, stopword removal, lemmatization)

Weak label generation with TextBlob (positive/negative/neutral)

ML Models: Logistic Regression, Naive Bayes with TF-IDF features

DL Model: LSTM with word embeddings

Model evaluation with classification reports & confusion matrices

🔹 Results:

Logistic Regression: ~78% accuracy

Naive Bayes: ~77% accuracy

LSTM: ~82% accuracy (best performance on longer reviews)

Sentiment distribution: majority reviews positive; negative reviews often mention cleanliness and noise

🔹 Business Value:
Provides hosts and platforms with insights into guest satisfaction drivers and areas for improvement.

3️⃣ Sales Forecasting – Time-Series Regression (Retail Data)

Objective: Forecast future sales performance for a retail dataset to support inventory and pricing decisions.
Tools & Libraries: Python, pandas, statsmodels, Prophet, matplotlib, seaborn

🔹 Workflow:

Time-series decomposition (trend, seasonality, residuals)

ARIMA and Prophet forecasting models

Evaluation using RMSE and MAPE

Visualization of actual vs forecasted sales

🔹 Results:

Prophet model captured seasonality and holiday effects better than ARIMA

Achieved low forecast error (<10% MAPE) on validation data

Forecast plots highlight sales spikes during holiday seasons

🔹 Business Value:
Supports demand planning, inventory management, and pricing strategy, reducing stockouts and overstock risks.

⚙️ Skills Demonstrated

Data Preprocessing: Cleaning, transformation, feature engineering

Machine Learning: Logistic Regression, Naive Bayes, Random Forest, XGBoost

Deep Learning (NLP): LSTM for text classification

Time-Series Forecasting: ARIMA, Prophet

Evaluation Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, RMSE, MAPE

Visualization & Insights: Seaborn, matplotlib

📌 How to Use

Clone the repo:

git clone https://github.com/your-username/ml-ai-portfolio.git
cd ml-ai-portfolio


Open individual project folders (churn_prediction/, airbnb_sentiment/, sales_forecasting/).

Run Jupyter notebooks:

jupyter notebook

🚀 Future Work

Deploy churn prediction and sentiment analysis models with Streamlit dashboards

Explore transformer-based NLP models (BERT, DistilBERT) for sentiment analysis

Integrate external economic/holiday data into sales forecasting

This model enables customer success teams to identify at-risk customers and reduce churn by targeting retention strategies such as discounts or contract upgrades.

# 🛡️ InsureIQ — Vehicle Insurance Claim Likelihood Predictor

A machine learning web app that predicts the likelihood of a vehicle insurance claim based on policyholder risk attributes.

## 🚀 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## 📌 Features
- **3 ML Models**: Gradient Boosting, Random Forest, Logistic Regression
- **Real-time Prediction** with a risk gauge and factor breakdown
- **Model Analytics**: ROC Curves, Confusion Matrix, Feature Importance
- **Data Insights**: Interactive charts on the synthetic dataset
- **Insurer Recommendations** based on risk tier

## 🧠 Risk Attributes Used
| Feature | Description |
|---|---|
| Age | Policyholder age |
| Gender | Male / Female |
| Vehicle Age | Age of vehicle in years |
| Annual KM | Kilometres driven per year |
| Previous Claims | Number of past claims |
| Credit Score | Credit score (300–850) |
| Vehicle Type | Sedan / SUV / Truck / Hatchback / Sports |
| Region | Urban / Suburban / Rural |
| Coverage Type | Basic / Standard / Premium |
| Safe Driver Discount | Eligible (Yes/No) |
| Annual Premium | Premium amount in ₹ |

## 🛠️ Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Tech Stack
- **Frontend**: Streamlit + Plotly
- **ML**: scikit-learn (GBM, Random Forest, Logistic Regression)
- **Data**: Synthetic dataset (5,000 records, probabilistic labelling)

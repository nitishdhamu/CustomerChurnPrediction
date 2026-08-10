# Streaming Platform Subscription Predictor 🍿🎧

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Enabled-orange)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📌 Overview
The biggest challenge for streaming platforms like Netflix, Spotify, or Amazon Prime isn't getting new users—it's stopping existing users from canceling (churn). This project provides an end-to-end Machine Learning pipeline to identify subscribers who are at a high risk of canceling their subscriptions. 

By accurately predicting subscription churn based on viewing/listening habits and account metrics, businesses can proactively offer incentives or targeted support to retain at-risk customers.

This repository includes:
1. **Synthetic Data Generation**: A robust script simulating realistic streaming platform data (watch time, active days, subscription tiers, payment failures).
2. **Data Preprocessing**: Handles categorical encoding, feature scaling, and addresses class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).
3. **Machine Learning Models**: Trains and evaluates Logistic Regression, Decision Trees, and Neural Networks (Multi-Layer Perceptron).
4. **Evaluation & Visualization**: Automatically generates Confusion Matrices, ROC curves, and Feature Importance plots.

## ⚙️ Features & Architecture
- **`generate_data.py`**: Simulates 10,000 user profiles with streaming-specific features such as `monthly_active_days`, `avg_watch_time_hours`, `subscription_tier`, `device_type`, `auto_renew_enabled`, and `payment_failures`. It injects logical correlations (e.g., users who turn off auto-renew and don't watch much content are highly likely to cancel).
- **`churn_prediction.py`**: The core ML pipeline.
  - **Preprocessing**: `StandardScaler` for numerical data, `pd.get_dummies()` for categorical variables.
  - **Oversampling**: Applies `SMOTE` only to the training set to prevent data leakage while balancing the non-churn to churn ratio.
  - **Modeling**: Uses `sklearn` to train models and evaluate them based on Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## 🛠️ Installation & Setup

1. **Clone the repository** (if hosted on GitHub):
   ```bash
   git clone <your-repo-url>
   cd Customer_Churn_Prediction
   ```

2. **Create a virtual environment** (Recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run

1. **Generate the Dataset**:
   This will create a `data/` directory and populate it with a `streaming_churn_data.csv` file.
   ```bash
   python src/generate_data.py
   ```

2. **Run the Prediction Pipeline**:
   This script loads the data, trains the models, and outputs all visualizations and metrics to a newly created `results/` folder.
   ```bash
   python src/churn_prediction.py
   ```

## 📊 Results & Performance
The pipeline compares three algorithms. Because customer churn heavily relies on identifying at-risk users, **Recall** is a critical metric (identifying as many true cancellations as possible). 

**Key Insights:**
- **Auto-renew status**, **Watch Time**, and **Active Days** are historically the strongest predictors of a user's likelihood to cancel.
- The pipeline provides detailed feature importance and ROC-AUC curves in the `results/` folder after every run.

## 📁 Directory Structure
```text
Customer_Churn_Prediction/
│
├── src/
│   ├── generate_data.py      # Data synthesis script
│   └── churn_prediction.py   # ML pipeline & evaluation
│
├── requirements.txt          # Project dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```
*(Note: `data/` and `results/` folders are automatically generated at runtime and ignored by git).*

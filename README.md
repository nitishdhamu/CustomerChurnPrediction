# 🍿 Streaming Platform Subscription Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Enabled-orange)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

Welcome to the **Streaming Platform Subscription Predictor**! This repository houses an end-to-end Machine Learning pipeline designed to solve one of the biggest problems in the streaming industry (Netflix, Spotify, Hulu, etc.): **Customer Churn**.

By analyzing user engagement metrics, account settings, and friction points, this tool predicts which users are at a high risk of canceling their subscription, allowing businesses to proactively intervene.

---

## 🎯 What Does This Project Do?

1. **Simulates Real-World Data**: It generates a highly realistic synthetic dataset of 10,000 users, including features like `monthly_active_days`, `avg_watch_time_hours`, `subscription_tier`, and `auto_renew_enabled`.
2. **Handles Imbalanced Data**: Because most users *don't* cancel, churn data is naturally imbalanced. This project uses **SMOTE** (Synthetic Minority Over-sampling Technique) to artificially balance the training data, ensuring the algorithms learn effectively.
3. **Trains Predictive Models**: It trains and evaluates three distinct algorithms:
   - **Logistic Regression**: A powerful statistical baseline.
   - **Decision Tree**: Highly interpretable, allowing us to see *why* users leave.
   - **Neural Network (MLP)**: A deep learning model that captures complex behavioral patterns.
4. **Generates Insights**: Automatically produces Confusion Matrices, ROC-AUC curves, and Feature Importance charts to explain its findings.

---

## 🛠️ Quick Start Guide

Want to run this project on your own machine? Follow these simple steps!

### Prerequisites
Make sure you have **Python 3.8+** installed on your system. You can download it from [python.org](https://www.python.org/).

### 1. Clone the Repository
Open your terminal (Command Prompt, PowerShell, or bash) and run:
```bash
git clone <your-repo-url>
cd Customer_Churn_Prediction
```

### 2. Set Up a Virtual Environment (Recommended)
It's best practice to install dependencies in an isolated environment so it doesn't conflict with other projects on your computer.

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries (like `pandas`, `scikit-learn`, and `seaborn`):
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline!
First, generate the synthetic streaming data. This creates a `data/` folder containing the CSV dataset.
```bash
python src/generate_data.py
```

Next, run the machine learning pipeline. This will process the data, train the models, and evaluate them.
```bash
python src/churn_prediction.py
```

### 5. Predict Real-World Cancellations (Inference)
Now that the AI is trained, you can run the inference script. This script simulates taking a brand new list of 100 *currently active* customers and feeds them through the trained Artificial Intelligence.
```bash
python src/predict_active_customers.py
```
This will output three separate spreadsheets in your `results/` folder so the marketing team can prioritize their efforts:
- `high_risk_customers.csv` (76% - 100% chance of churning)
- `medium_risk_customers.csv` (41% - 75% chance of churning)
- `low_risk_customers.csv` (0% - 40% chance of churning)

### 6. View the Full Results
Once the script finishes, a `results/` folder will appear! Open it to find:
- `model_comparison.csv`: A spreadsheet summarizing Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- `feature_importance.csv`: A spreadsheet showing the top reasons users cancel (e.g., turning off auto-renew).
- `at_risk_customers_list.csv`: The final list of currently active users who are flagged as highly likely to cancel.

---

## 🧠 Key Insights
If you look at the generated `feature_importance.png`, you'll notice a few trends:
- **Auto-Renew Disabled**: Users who manually turn off auto-renew are the absolute highest flight risk.
- **Watch Time & Active Days**: A sudden drop in engagement is a strong indicator of churn.
- **Support Tickets & Payment Failures**: Friction points highly correlate with user frustration and eventual cancellation.

By using this pipeline, a streaming company could identify these users *before* they cancel and automatically send them a targeted retention offer (e.g., "Get your next month 50% off!").

---

## 📁 Repository Structure
```text
Customer_Churn_Prediction/
│
├── src/
│   ├── generate_data.py      # Simulates user profiles and streaming behavior
│   └── churn_prediction.py   # Cleans data, applies SMOTE, trains models, and plots results
│
├── requirements.txt          # Python package dependencies
├── .gitignore                # Tells Git to ignore temporary files (like data/ and results/)
└── README.md                 # This guide!
```

---
*Created as part of a Data Science Internship Project.*

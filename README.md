# 🍿 Streaming Platform Subscription Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Enabled-orange)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

Welcome to the **Streaming Platform Subscription Predictor**! This repository houses an end-to-end Machine Learning pipeline designed to solve one of the biggest problems in the streaming industry (Netflix, Spotify, Hulu, etc.): **Customer Churn**.

By analyzing user engagement metrics, account settings, and friction points from historical data, this tool predicts which currently active users are at a high risk of canceling their subscription, allowing businesses to proactively intervene.

---

## 🎯 What Does This Project Do?

1. **Analyzes Historical Data**: It ingests a massive dataset of 100,000 users from the past financial year, including features like `monthly_active_days`, `avg_watch_time_hours`, `subscription_tier`, and `auto_renew_enabled`.
2. **Handles Imbalanced Data**: Because most users *don't* cancel, churn data is naturally imbalanced. This project uses **SMOTE** (Synthetic Minority Over-sampling Technique) to artificially balance the training data, ensuring the algorithms learn effectively.
3. **Trains Predictive Models**: It trains and evaluates three distinct algorithms:
   - **Logistic Regression**: A powerful statistical baseline.
   - **Decision Tree**: Highly interpretable, allowing us to see *why* users leave.
   - **Neural Network (MLP)**: A deep learning model that captures complex behavioral patterns.
4. **Real-World Inference**: It applies the trained Neural Network to a database of 120,000 *current* active subscribers and categorizes their flight risk into High, Medium, and Low tiers for the Marketing Team.

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
Install all required libraries (like `pandas`, `scikit-learn`, and `joblib`):
```bash
pip install -r requirements.txt
```

### 4. Run the Machine Learning Pipeline
This script loads the historical data (`data/historical_data_fy25.csv`), processes it, trains the Artificial Intelligence, and outputs performance metrics and visualizations.
```bash
python src/churn_prediction.py
```

### 5. Predict Real-World Cancellations (Inference)
Now that the AI is trained, you can run the inference script. This script loads the 120,000 *currently active* customers (`data/current_active_subscribers.csv`) and feeds them through the trained Neural Network.
```bash
python src/predict_active_customers.py
```
This will output three separate spreadsheets in your `results/` folder so the marketing team can prioritize their efforts:
- `high_risk_customers.csv` (76% - 100% chance of churning)
- `medium_risk_customers.csv` (41% - 75% chance of churning)
- `low_risk_customers.csv` (0% - 40% chance of churning)

### 6. View the Full Results
Once the training script finishes, a `results/` folder will appear! Open it to find:
- `model_comparison.csv`: A spreadsheet summarizing Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- `feature_importance.png`: A chart showing the top reasons users cancel (e.g., turning off auto-renew).
- `roc_curves.png`: A graph comparing the predictive power of all three models.
- `cm_*.png`: Confusion matrices showing exactly how many predictions were correct vs. incorrect for each model.

---

## 🧠 Key Insights
If you look at the generated `feature_importance.png`, you'll notice a few trends:
- **Auto-Renew Disabled**: Users who manually turn off auto-renew are the absolute highest flight risk.
- **Watch Time & Active Days**: A sudden drop in engagement is a strong indicator of churn.
- **Support Tickets & Payment Failures**: Friction points highly correlate with user frustration and eventual cancellation.

By using this pipeline, a streaming company could identify these users *before* they cancel and automatically send them a targeted retention offer (e.g., "Get your next month 50% off!").

---
*Created as part of a Data Science Internship Project.*

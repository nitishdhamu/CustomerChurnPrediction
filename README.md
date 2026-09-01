# 📺 Streaming Platform Subscription Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Enabled-orange)
![Status](https://img.shields.io/badge/Status-Complete-success)

Welcome to the **Streaming Platform Subscription Predictor**! This repository houses an end-to-end Machine Learning pipeline designed to solve one of the biggest problems in the streaming industry: **Customer Churn**.

By analyzing user engagement metrics, account settings, and friction points from historical data, this tool predicts which currently active users are at a high risk of canceling their subscription, allowing businesses to proactively intervene.

---

## 🚀 What Does This Project Do?

1. **Generates Synthetic Data**: It generates massive, hyper-realistic databases mimicking popular streaming services (Netflix, Prime, AppleTV, JioHotstar) with millions of unique users.
2. **Trains Predictive Models**: It isolates data per platform, drops PII (so the AI learns behavior, not names), and trains three distinct algorithms:
   - **Logistic Regression**: A powerful statistical baseline.
   - **Decision Tree**: Highly interpretable, allowing us to see *why* users leave.
   - **Neural Network (MLP)**: A deep learning model that captures complex behavioral patterns.
3. **Real-World Inference**: It applies the trained platform-specific Neural Networks to currently active subscribers and categorizes their flight risk into High, Medium, and Low tiers for the Marketing Team.

---

## 📖 Quick Start Guide

### Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Customer_Churn_Prediction
```

### 2. Set Up a Virtual Environment (Recommended)
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
```bash
pip install -r requirements.txt
```

### 4. Generate Some Sample Databases (If needed)
Use this script to mathematically simulate millions of users across different platforms (Netflix, Prime, etc.).
```bash
python src/generate_data.py
```
*Note: The generated CSV files will be saved to `data/` and are automatically ignored by Git to prevent repository bloat.*

### 5. Run the Machine Learning Pipeline (Training)
This script trains independent Neural Networks for your datasets. Simply run the script to launch the **Interactive CLI Menu**, which will automatically scan your `data/` folder and display the exact CSV file names present.
```bash
python src/train_models.py
```
*(You can select one, multiple comma-separated numbers, or type 'All')*

When finished, the isolated models will be saved to the `models/` directory, and accuracy summaries will be saved to `metrics/<prefix>_metrics.csv`.

### 6. Predict Real-World Cancellations (Inference)
Now that the AI is trained, run the inference engine to target at-risk users! 

Just like training, run the script to trigger the **Interactive Menu**:
```bash
python src/predict_churn.py
```

This evaluates all currently active users on that dataset and outputs three separate spreadsheets in your `results/` folder so the marketing team can prioritize their efforts:
- `<prefix>_high_risk.csv` (76% - 100% chance of churning)
- `<prefix>_medium_risk.csv` (41% - 75% chance of churning)
- `<prefix>_low_risk.csv` (0% - 40% chance of churning)

---
*Created as part of a Data Science Internship Project.*



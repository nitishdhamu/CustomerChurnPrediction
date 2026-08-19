# Customer Churn Prediction

A machine learning project to predict customer churn for streaming services like Netflix, Prime, AppleTV, and JioHotstar.

The goal is to analyze user engagement, account info, and support history to find which subscribers are likely to cancel their subscriptions.

## Overview

1. Data Generation: Generates sample customer datasets with activity metrics, billing info, and churn flags.
2. Model Training: Trains three models (Logistic Regression, Decision Tree, and MLP Neural Network) on each platform's data and saves evaluation metrics.
3. Prediction: Loads the trained model to score active customers and groups them into high, medium, and low risk lists.

## Setup

### Prerequisites
Python 3.8 or higher.

### 1. Clone the repository
```bash
git clone https://github.com/nitishdhamu/CustomerChurnPrediction.git
cd CustomerChurnPrediction
```

### 2. Create virtual environment
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Mac / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

## Running the Code

### 1. Generate data
To create sample datasets:
```bash
python src/generate_data.py
```
This lets you pick which platform to generate data for. The generated files are saved in the `data/` folder.

### 2. Train models
To train the models on your data:
```bash
python src/train_models.py
```
You can choose which dataset to train on from the menu (or it will auto-run if only one dataset exists). The trained model bundle is saved in `models/` and evaluation metrics are saved in `metrics/`.

### 3. Predict churn
To predict churn for active subscribers:
```bash
python src/predict_churn.py
```
This scans for trained models, runs predictions on active users, and saves the categorized lists into the `results/` folder:
- `<platform>_high_risk.csv` (churn probability > 75%)
- `<platform>_medium_risk.csv` (churn probability between 40% and 75%)
- `<platform>_low_risk.csv` (churn probability <= 40%)

# Customer Churn Prediction 🚀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Enabled-orange)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📌 Overview
Customer churn is a critical metric for subscription-based businesses. This project provides an end-to-end Machine Learning pipeline to identify customers who are likely to leave a service. By accurately predicting churn, businesses can proactively offer incentives or targeted support to retain at-risk customers.

This repository includes:
1. **Synthetic Data Generation**: A robust script to create realistic, highly imbalanced customer data (usage, support calls, activity logs, etc.).
2. **Data Preprocessing**: Handles categorical encoding, feature scaling, and addresses class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).
3. **Machine Learning Models**: Trains and evaluates Logistic Regression, Decision Trees, and Neural Networks (Multi-Layer Perceptron).
4. **Evaluation & Visualization**: Automatically generates Confusion Matrices, ROC curves, and Feature Importance plots.

## ⚙️ Features & Architecture
- **`generate_data.py`**: Simulates 10,000 customer profiles with features such as `tenure_months`, `monthly_charges`, `total_usage_gb`, `contract_type`, and `support_calls`. It injects realistic statistical correlations (e.g., lower tenure + high support calls = higher probability of churn).
- **`churn_prediction.py`**: The core ML pipeline.
  - **Preprocessing**: `StandardScaler` for numerical data, `pd.get_dummies()` for categorical variables.
  - **Oversampling**: Applies `SMOTE` only to the training set to prevent data leakage while balancing the 80/20 non-churn to churn ratio.
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
   This will create a `data/` directory and populate it with a `customer_churn_data.csv` file.
   ```bash
   python src/generate_data.py
   ```

2. **Run the Prediction Pipeline**:
   This script loads the data, trains the models, and outputs all visualizations and metrics to a newly created `results/` folder.
   ```bash
   python src/churn_prediction.py
   ```

## 📊 Results & Performance
The pipeline compares three algorithms. During recent runs, the **Neural Network** achieved the highest overall performance on the test data:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | ~97% | ~90% | 100% | ~0.95 | ~0.99 |
| **Decision Tree** | ~95% | ~84% | 96% | ~0.90 | ~0.98 |
| **Neural Network** | **~99.5%**| **~98.5%**| **~99%** | **~0.98** | **~0.99** |

*Note: Results may vary slightly due to the random initialization of the synthetic dataset.*

**Key Insights:**
- **Support Calls** and **Tenure** are consistently identified as the strongest predictors of churn.
- Because customer churn heavily relies on identifying at-risk users, **Recall** is prioritized. The models perform exceptionally well at catching true positives.

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

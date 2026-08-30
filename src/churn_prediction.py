import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_data(filepath):
    """
    Loads the master database, filters for mature accounts (training data), 
    drops PII, and encodes features.
    """
    print(f"[*] Loading massive database from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"[*] Total users in database: {len(df):,}")
    
    # BUSINESS LOGIC: We only train on users who have been around for more than 6 months 
    # to ensure their behavior patterns are fully matured.
    train_df = df[df['tenure_months'] > 6].copy()
    print(f"[*] Filtered to {len(train_df):,} mature users (tenure > 6 months) for AI training.")
    
    # DROP PII (Names, Emails, IDs) - AI does not need these!
    train_df = train_df.drop(['customer_id', 'name', 'email'], axis=1)
    
    print("[*] Encoding categorical features...")
    categorical_cols = ['subscription_tier', 'billing_cycle', 'auto_renew_enabled', 'customer_acquisition_channel', 'primary_device']
    train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
    
    feature_columns = train_df.drop('churn', axis=1).columns
    os.makedirs("models", exist_ok=True)
    joblib.dump(feature_columns, "models/feature_columns.pkl")
    
    X = train_df.drop('churn', axis=1)
    y = train_df['churn']
    
    print("[*] Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[*] Scaling numerical features...")
    scaler = StandardScaler()
    num_cols = ['user_age', 'tenure_months', 'days_since_last_login', 'avg_watch_time_hours', 'content_completion_rate', 'payment_failures', 'support_tickets', 'support_resolution_time_days']
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    joblib.dump(scaler, "models/scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print(f"\n[*] Original training set class distribution: 0: {sum(y_train==0)}, 1: {sum(y_train==1)}")
    
    # NOTE: With 1 Million rows, we have hundreds of thousands of minority class samples. 
    # SMOTE is an anti-pattern for Big Data as it causes memory explosions (O(N^2) KNN). 
    # We will train directly on the massive dataset.

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, random_state=42) # Reduced epochs for Big Data
    }

    results = {}
    print("\n[*] Training models on massive dataset...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'ROC-AUC': roc_auc}
        
        print(f"\n--- {name} Performance ---")
        print(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-Score: {f1:.4f}\nROC-AUC: {roc_auc:.4f}")

        if name == "Neural Network":
            joblib.dump(model, "models/best_model_nn.pkl")
            print(f"[*] Saved Neural Network model to models/best_model_nn.pkl")

    results_df = pd.DataFrame(results).T
    print("\n==== Summary of Results ====")
    print(results_df)
    results_df.to_csv("results/model_comparison.csv")

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    data_path = os.path.join("data", "streaming_users_database.csv")
    if not os.path.exists(data_path):
        print(f"[!] Error: Data file not found at {data_path}. Please run src/generate_database.py first.")
        return
        
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(data_path)
    train_and_evaluate_models(X_train, X_test, y_train, y_test)
    print("\n[+] Training Pipeline completed successfully! Check the 'results/' folder for CSV summaries.")

if __name__ == "__main__":
    main()

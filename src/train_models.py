import pandas as pd
import numpy as np
import os
import joblib
import glob
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_data(filepath, platform_name):
    """
    Loads a specific master database, filters for mature accounts, 
    drops PII, and encodes features. Samples rows to prevent memory overload.
    """
    print(f"\n[*] =========================================")
    print(f"[*] Processing data for platform: {platform_name}")
    print(f"[*] =========================================")
    
    print(f"[*] Loading {filepath}...")
    df = pd.read_csv(filepath)
    print(f"[*] Total users for {platform_name}: {len(df):,}")
    
    # BUSINESS LOGIC: We only train on users who have been around for more than 6 months 
    train_df = df[df['tenure_months'] > 6].copy()
    print(f"[*] Filtered to {len(train_df):,} mature users (tenure > 6 months) for AI training.")
    
    if len(train_df) > 500000:
        print("[*] Sampling 500,000 users for efficient model training...")
        train_df = train_df.sample(n=500000, random_state=42)
    
    # DROP PII (Names, Emails, IDs)
    train_df = train_df.drop(['customer_id', 'name', 'email'], axis=1)
    
    print("[*] Encoding categorical features...")
    categorical_cols = ['subscription_tier', 'billing_cycle', 'auto_renew_enabled', 'customer_acquisition_channel', 'primary_device']
    train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
    
    feature_columns = train_df.drop('churn', axis=1).columns
    os.makedirs("models", exist_ok=True)
    joblib.dump(feature_columns, f"models/{platform_name}_feature_columns.pkl")
    
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
    
    joblib.dump(scaler, f"models/{platform_name}_scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns

def train_and_evaluate(X_train, X_test, y_train, y_test, platform_name):
    print(f"[*] Initializing Artificial Intelligence Models for {platform_name}...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(max_depth=7, class_weight='balanced'),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, early_stopping=True)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"    -> Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        
        if name == 'Neural Network':
            print(f"[*] Saving {name} as the production model for {platform_name}...")
            joblib.dump(model, f"models/{platform_name}_best_model_nn.pkl")
            
    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    out_csv = f"results/{platform_name}_model_comparison.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"[*] Training Complete! Results saved to {out_csv}")
    print(results_df.to_string(index=False))
    print("\n")

def main():
    parser = argparse.ArgumentParser(description="Train Churn Prediction Model on streaming data")
    parser.add_argument('--platforms', nargs='*', help="Specify which platforms to train on (e.g., --platforms Netflix Prime_Video). Leave blank to train on all independently.")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    
    if not args.platforms:
        files = glob.glob('data/*_users_database.csv')
    else:
        files = [f"data/{p}_users_database.csv" for p in args.platforms]
        
    valid_files = [f for f in files if os.path.exists(f)]
    if not valid_files:
        print("[!] Error: No valid database files found for the requested platforms.")
        return

    for filepath in valid_files:
        platform_name = os.path.basename(filepath).replace('_users_database.csv', '')
        X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(filepath, platform_name)
        if X_train is not None:
            train_and_evaluate(X_train, X_test, y_train, y_test, platform_name)

if __name__ == "__main__":
    main()

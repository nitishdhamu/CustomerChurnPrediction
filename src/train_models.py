import pandas as pd
import numpy as np
import os
import joblib
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_interactive_files():
    files = glob.glob('data/*.csv')
    if not files:
        print("[!] No CSV datasets found in the data/ folder.")
        return []
        
    files = sorted(files)
    
    if len(files) == 1:
        print(f"\n[*] Only one dataset found: {os.path.basename(files[0])}. Automatically selecting it...")
        return files
    
    print("\n[*] Available Datasets:")
    for i, f in enumerate(files, 1):
        print(f"    {i}. {os.path.basename(f)}")
    print(f"    {len(files) + 1}. All Datasets")
    
    choice = input("\n[?] Select datasets to train on (e.g., '1', '1,3', or 'All'): ").strip()
    
    if choice.lower() == 'all' or choice == str(len(files) + 1):
        return files
        
    selected_files = []
    for c in choice.split(','):
        c = c.strip()
        if c.isdigit():
            idx = int(c) - 1
            if 0 <= idx < len(files):
                selected_files.append(files[idx])
        else:
            for f in files:
                if os.path.basename(f) == c:
                    selected_files.append(f)
                    
    return selected_files

def load_and_preprocess_data(filepath, prefix):
    print(f"\n[*] =========================================")
    print(f"[*] Processing dataset: {os.path.basename(filepath)}")
    print(f"[*] =========================================")
    
    print(f"[*] Loading {filepath}...")
    df = pd.read_csv(filepath)
    print(f"[*] Total rows found: {len(df):,}")
    
    train_df = df[df['tenure_months'] > 6].copy()
    print(f"[*] Filtered to {len(train_df):,} mature users (tenure > 6 months) for AI training.")
    
    if len(train_df) > 500000:
        print("[*] Sampling 500,000 users for efficient model training...")
        train_df = train_df.sample(n=500000, random_state=42)
    
    train_df = train_df.drop(['customer_id', 'name', 'email'], axis=1)
    
    print("[*] Encoding categorical features...")
    categorical_cols = ['billing_cycle', 'auto_renew_enabled']
    train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
    
    feature_columns = train_df.drop('churn', axis=1).columns
    
    X = train_df.drop('churn', axis=1)
    y = train_df['churn']
    
    print("[*] Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[*] Scaling numerical features...")
    scaler = StandardScaler()
    num_cols = ['age', 'tenure_months', 'days_since_last_login', 'avg_watch_time_hours', 'payment_failures', 'support_tickets', 'support_resolution_time_days']
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns, scaler

def train_and_evaluate(X_train, X_test, y_train, y_test, prefix, feature_columns, scaler):
    print(f"[*] Initializing Artificial Intelligence Models for {prefix}...")
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
            print(f"[*] Saving unified {name} bundle (Model + Scaler + Features) for {prefix}...")
            os.makedirs("models", exist_ok=True)
            model_bundle = {
                'model': model,
                'scaler': scaler,
                'features': feature_columns
            }
            joblib.dump(model_bundle, f"models/{prefix}_model.pkl")
            
    results_df = pd.DataFrame(results)
    os.makedirs("metrics", exist_ok=True)
    out_csv = f"metrics/{prefix}_metrics.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"[*] Training Complete! Results saved to {out_csv}")
    print(results_df.to_string(index=False))
    print("\n")

def main():
    print("[*] Starting AI Training Engine...")
    os.makedirs("metrics", exist_ok=True)
    
    files_to_run = get_interactive_files()
    
    if not files_to_run:
        print("[!] No datasets selected. Exiting.")
        return

    for filepath in files_to_run:
        prefix = os.path.basename(filepath).replace('.csv', '').split('_')[0]
        X_train, X_test, y_train, y_test, feature_columns, scaler = load_and_preprocess_data(filepath, prefix)
        if X_train is not None:
            train_and_evaluate(X_train, X_test, y_train, y_test, prefix, feature_columns, scaler)

if __name__ == "__main__":
    main()

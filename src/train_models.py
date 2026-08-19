import pandas as pd
import numpy as np
import os
import joblib
import glob
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')

def get_interactive_files():
    files = glob.glob('data/*.csv')
    if not files:
        print("\nNo CSV files found in data/ folder.\n")
        return []
        
    files = sorted(files)
    
    if len(files) == 1:
        print(f"\nFound only one dataset ({os.path.basename(files[0])}), using it automatically.")
        return files
    
    print("\nSelect datasets to train on:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {os.path.basename(f)}")
    print(f"  {len(files) + 1}. All Datasets")
    
    choice = input("\nEnter choice (e.g. 1, 1,3, or All): ").strip()
    
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
    print(f"\n--- Training models for {prefix} ---")
    print(f"Loading {os.path.basename(filepath)}...")
    df = pd.read_csv(filepath)
    
    if 'churn' not in df.columns:
        print(f"Error: 'churn' column not found in {filepath}. Cannot train.")
        return None, None, None, None, None, None
    
    train_df = df[df['tenure_months'] > 6].copy() if 'tenure_months' in df.columns else df.copy()
    if len(train_df) < 50:
        train_df = df.copy()
    
    if len(train_df) > 500000:
        print("Subsampling to 500,000 rows for faster training...")
        train_df = train_df.sample(n=500000, random_state=42)
    
    drop_cols = [c for c in ['customer_id', 'name', 'email'] if c in train_df.columns]
    train_df = train_df.drop(drop_cols, axis=1)
    
    # one-hot encode categorical features
    cat_candidates = ['billing_cycle', 'auto_renew_enabled']
    categorical_cols = [c for c in cat_candidates if c in train_df.columns]
    if categorical_cols:
        train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
    
    feature_columns = train_df.drop('churn', axis=1).columns
    
    X = train_df.drop('churn', axis=1)
    y = train_df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # scale numeric features
    scaler = StandardScaler()
    num_candidates = ['age', 'tenure_months', 'days_since_last_login', 'avg_watch_time_hours', 'payment_failures', 'support_tickets', 'support_resolution_time_days']
    num_cols = [c for c in num_candidates if c in X_train.columns]
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if num_cols:
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns, scaler

def train_and_evaluate(X_train, X_test, y_train, y_test, prefix, feature_columns, scaler):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(max_depth=7, class_weight='balanced'),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, early_stopping=True)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
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
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4)
        })
        
        if name == 'Neural Network':
            os.makedirs("models", exist_ok=True)
            model_bundle = {
                'model': model,
                'model_name': name,
                'model_type': name,
                'scaler': scaler,
                'features': feature_columns
            }
            joblib.dump(model_bundle, f"models/{prefix}_model.pkl")
            print(f"Saved model bundle to models/{prefix}_model.pkl")
            
    results_df = pd.DataFrame(results)
    os.makedirs("metrics", exist_ok=True)
    out_csv = f"metrics/{prefix}_metrics.csv"
    results_df.to_csv(out_csv, index=False)
    
    print(f"Saved evaluation metrics to {out_csv}\n")
    print(results_df.to_string(index=False))
    print()

def main():
    print("=== Model Training ===")
    os.makedirs("metrics", exist_ok=True)
    
    files_to_run = get_interactive_files()
    
    if not files_to_run:
        print("No datasets selected. Exiting.")
        return

    for filepath in files_to_run:
        prefix = os.path.basename(filepath).replace('.csv', '').split('_')[0]
        X_train, X_test, y_train, y_test, feature_columns, scaler = load_and_preprocess_data(filepath, prefix)
        if X_train is not None:
            train_and_evaluate(X_train, X_test, y_train, y_test, prefix, feature_columns, scaler)
            
    print("All models trained successfully.")

if __name__ == "__main__":
    main()

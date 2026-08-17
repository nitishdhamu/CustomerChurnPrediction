import pandas as pd
import numpy as np
import os
import joblib
import glob

def get_interactive_files():
    files = glob.glob('data/*.csv')
    if not files:
        print("[!] No CSV datasets found in the data/ folder.")
        return []
    
    files = sorted(files)
    
    print("\n[*] Available Datasets:")
    for i, f in enumerate(files, 1):
        print(f"    {i}. {os.path.basename(f)}")
    print(f"    {len(files) + 1}. All Datasets")
    
    choice = input("\n[?] Select datasets to run inference on (e.g., '1', '1,3', or 'All'): ").strip()
    
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

def predict_for_dataset(filepath, prefix):
    print(f"\n[*] Preprocessing active customer data for {os.path.basename(filepath)}...")
    
    model_path = f"models/{prefix}_model.pkl"
    scaler_path = f"models/{prefix}_scaler.pkl"
    feature_path = f"models/{prefix}_features.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feature_path)):
        print(f"[!] Error: AI models for {prefix} not found! Please run train_models.py first.")
        return
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_path)
    
    df = pd.read_csv(filepath)
    
    active_customers = df[df['churn'] == 0].copy()
    print(f"[*] Found {len(active_customers):,} currently active subscribers.")
    
    inference_df = active_customers.drop(['customer_id', 'name', 'email', 'churn'], axis=1)
    
    categorical_cols = ['billing_cycle', 'auto_renew_enabled']
    inference_df = pd.get_dummies(inference_df, columns=categorical_cols, drop_first=True)
    
    for col in feature_columns:
        if col not in inference_df.columns:
            inference_df[col] = 0
            
    inference_df = inference_df[feature_columns]
    
    num_cols = ['age', 'tenure_months', 'days_since_last_login', 'avg_watch_time_hours', 'payment_failures', 'support_tickets', 'support_resolution_time_days']
    inference_df[num_cols] = scaler.transform(inference_df[num_cols])
    
    print(f"[*] Running AI inference...")
    churn_probs = model.predict_proba(inference_df)[:, 1]
    
    active_customers['churn_probability'] = np.round(churn_probs, 4)
    active_customers = active_customers.sort_values(by='churn_probability', ascending=False)
    
    high_risk = active_customers[active_customers['churn_probability'] > 0.75]
    medium_risk = active_customers[(active_customers['churn_probability'] > 0.40) & (active_customers['churn_probability'] <= 0.75)]
    low_risk = active_customers[active_customers['churn_probability'] <= 0.40]
    
    os.makedirs("results", exist_ok=True)
    high_risk.to_csv(f"results/{prefix}_high_risk.csv", index=False)
    medium_risk.to_csv(f"results/{prefix}_medium_risk.csv", index=False)
    low_risk.to_csv(f"results/{prefix}_low_risk.csv", index=False)
    
    print("="*77)
    print(f"[+] Prediction Complete for {os.path.basename(filepath)}!")
    print(f"    - High Risk (76-100%): {len(high_risk):,} users")
    print(f"    - Medium Risk (41-75%): {len(medium_risk):,} users")
    print(f"    - Low Risk (0-40%): {len(low_risk):,} users")
    print("="*77)

def main():
    print("[*] Starting AI inference engine...")
    
    files_to_run = get_interactive_files()
    
    if not files_to_run:
        print("[!] No datasets selected. Exiting.")
        return

    for f in files_to_run:
        prefix = os.path.basename(f).replace('.csv', '')
        predict_for_dataset(f, prefix)

if __name__ == "__main__":
    main()

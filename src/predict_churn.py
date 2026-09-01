import pandas as pd
import numpy as np
import os
import joblib
import glob
import argparse

def get_platform_name(filename):
    base = os.path.basename(filename)
    for suffix in ['_dataset.csv', '_users.csv', '.csv']:
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base

def get_interactive_platforms():
    files = glob.glob('data/*.csv')
    if not files:
        print("[!] No CSV datasets found in the data/ folder.")
        return []
    
    platforms = [get_platform_name(f) for f in files]
    platforms = sorted(list(set(platforms)))
    
    print("\n[*] Available Datasets:")
    for i, p in enumerate(platforms, 1):
        print(f"    {i}. {p}")
    print(f"    {len(platforms) + 1}. All Platforms")
    
    choice = input("\n[?] Select datasets to run inference on (e.g., '1', '1,3', or 'All'): ").strip()
    
    if choice.lower() == 'all' or choice == str(len(platforms) + 1):
        return platforms
        
    selected = []
    for c in choice.split(','):
        c = c.strip()
        if c.isdigit():
            idx = int(c) - 1
            if 0 <= idx < len(platforms):
                selected.append(platforms[idx])
        elif c in platforms:
            selected.append(c)
            
    return selected

def predict_for_platform(filepath, platform_name):
    print(f"\n[*] Preprocessing active customer data for {platform_name}...")
    
    model_path = f"models/{platform_name}_model.pkl"
    scaler_path = f"models/{platform_name}_scaler.pkl"
    feature_path = f"models/{platform_name}_features.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feature_path)):
        print(f"[!] Error: AI models for {platform_name} not found! Please run train_models.py first.")
        return
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_path)
    
    df = pd.read_csv(filepath)
    
    active_customers = df[df['churn'] == 0].copy()
    print(f"[*] Found {len(active_customers):,} currently active subscribers in {platform_name}.")
    
    inference_df = active_customers.drop(['customer_id', 'name', 'email', 'churn'], axis=1)
    
    categorical_cols = ['billing_cycle', 'auto_renew_enabled']
    inference_df = pd.get_dummies(inference_df, columns=categorical_cols, drop_first=True)
    
    for col in feature_columns:
        if col not in inference_df.columns:
            inference_df[col] = 0
            
    inference_df = inference_df[feature_columns]
    
    num_cols = ['age', 'tenure_months', 'days_since_last_login', 'avg_watch_time_hours', 'payment_failures', 'support_tickets', 'support_resolution_time_days']
    inference_df[num_cols] = scaler.transform(inference_df[num_cols])
    
    print(f"[*] Running AI inference for {platform_name}...")
    churn_probs = model.predict_proba(inference_df)[:, 1]
    
    active_customers['churn_probability'] = np.round(churn_probs, 4)
    active_customers = active_customers.sort_values(by='churn_probability', ascending=False)
    
    high_risk = active_customers[active_customers['churn_probability'] > 0.75]
    medium_risk = active_customers[(active_customers['churn_probability'] > 0.40) & (active_customers['churn_probability'] <= 0.75)]
    low_risk = active_customers[active_customers['churn_probability'] <= 0.40]
    
    os.makedirs("results", exist_ok=True)
    high_risk.to_csv(f"results/{platform_name}_high_risk.csv", index=False)
    medium_risk.to_csv(f"results/{platform_name}_medium_risk.csv", index=False)
    low_risk.to_csv(f"results/{platform_name}_low_risk.csv", index=False)
    
    print("="*77)
    print(f"[+] Prediction Complete for {platform_name}!")
    print(f"    - High Risk (76-100%): {len(high_risk):,} users")
    print(f"    - Medium Risk (41-75%): {len(medium_risk):,} users")
    print(f"    - Low Risk (0-40%): {len(low_risk):,} users")
    print("="*77)

def main():
    parser = argparse.ArgumentParser(description="Predict Churn on streaming data")
    parser.add_argument('--platforms', nargs='*', help="Specify which platforms to predict on.")
    args = parser.parse_args()

    print("[*] Starting AI inference engine...")
    
    platforms_to_run = args.platforms if args.platforms else get_interactive_platforms()
    
    if not platforms_to_run:
        print("[!] No platforms selected. Exiting.")
        return
        
    files = glob.glob('data/*.csv')
    valid_files = []
    for f in files:
        if get_platform_name(f) in platforms_to_run:
            valid_files.append(f)
            
    if not valid_files:
        print("[!] Error: No valid database files found for the requested platforms.")
        return

    for f in valid_files:
        platform_name = get_platform_name(f)
        predict_for_platform(f, platform_name)

if __name__ == "__main__":
    main()

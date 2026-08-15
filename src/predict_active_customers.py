import pandas as pd
import numpy as np
import os
import joblib
import glob
import argparse

def predict_for_platform(filepath, platform_name):
    print(f"\n[*] Preprocessing active customer data for {platform_name}...")
    
    model_path = f"models/{platform_name}_best_model_nn.pkl"
    scaler_path = f"models/{platform_name}_scaler.pkl"
    feature_path = f"models/{platform_name}_feature_columns.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feature_path)):
        print(f"[!] Error: AI models for {platform_name} not found! Please run churn_prediction.py first.")
        return
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_path)
    
    df = pd.read_csv(filepath)
    
    active_customers = df[df['churn'] == 0].copy()
    print(f"[*] Found {len(active_customers):,} currently active subscribers in {platform_name}.")
    
    inference_df = active_customers.drop(['customer_id', 'name', 'email', 'churn'], axis=1)
    
    categorical_cols = ['subscription_tier', 'billing_cycle', 'auto_renew_enabled', 'customer_acquisition_channel', 'primary_device']
    inference_df = pd.get_dummies(inference_df, columns=categorical_cols, drop_first=True)
    
    for col in feature_columns:
        if col not in inference_df.columns:
            inference_df[col] = 0
            
    inference_df = inference_df[feature_columns]
    
    num_cols = ['user_age', 'tenure_months', 'days_since_last_login', 'avg_watch_time_hours', 'content_completion_rate', 'payment_failures', 'support_tickets', 'support_resolution_time_days']
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
    print("-" * 77)
    
    print("Top 3 Highest Risk Customers (Marketing Target List):")
    print(f"| {'Name':<25} | {'Email':<35} | {'Risk %':<8} |")
    print("-" * 77)
    for _, row in high_risk.head(3).iterrows():
        risk_pct = f"{row['churn_probability']*100:.2f}%"
        print(f"| {row['name']:<25} | {row['email']:<35} | {risk_pct:>8} |")
    print("="*77)

def main():
    parser = argparse.ArgumentParser(description="Predict Churn on streaming data")
    parser.add_argument('--platforms', nargs='*', help="Specify which platforms to predict (e.g., --platforms Netflix Prime_Video). Leave blank to predict on all.")
    args = parser.parse_args()

    print("[*] Starting batch AI inference engine...")
    
    if not args.platforms:
        files = glob.glob('data/*_users_database.csv')
    else:
        files = [f"data/{p}_users_database.csv" for p in args.platforms]
        
    valid_files = [f for f in files if os.path.exists(f)]
    if not valid_files:
        print("[!] Error: No valid database files found for the requested platforms.")
        return

    for f in valid_files:
        platform_name = os.path.basename(f).replace('_users_database.csv', '')
        predict_for_platform(f, platform_name)

if __name__ == "__main__":
    main()

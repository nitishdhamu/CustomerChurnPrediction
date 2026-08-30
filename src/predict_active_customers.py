import pandas as pd
import numpy as np
import joblib
import os

def main():
    print("[*] Loading Master Database...")
    data_path = os.path.join("data", "streaming_users_database.csv")
    
    if not os.path.exists(data_path):
        print(f"[!] Error: {data_path} not found. Please run src/generate_database.py first.")
        return
        
    master_df = pd.read_csv(data_path)
    
    # BUSINESS LOGIC: We only predict churn for currently ACTIVE users.
    # We drop anyone who already churned (churn == 1)
    active_customers_df = master_df[master_df['churn'] == 0].copy()
    print(f"[*] Found {len(active_customers_df):,} currently active subscribers out of {len(master_df):,}.")
    
    # We drop the actual target column since we are predicting it
    active_customers_df = active_customers_df.drop('churn', axis=1)
    
    # 1. Load the saved "Brains" from our training phase
    print("[*] Loading trained Neural Network model, scaler, and feature columns...")
    try:
        model = joblib.load("models/best_model_nn.pkl")
        scaler = joblib.load("models/scaler.pkl")
        expected_columns = joblib.load("models/feature_columns.pkl")
    except FileNotFoundError:
        print("[!] Error: Model files not found. Please run src/churn_prediction.py first to train the models.")
        return

    # 2. Preprocess the active customers identically to how we preprocessed the training data
    print("[*] Preprocessing active customer data (Stripping PII for AI)...")
    # Strip PII before sending to AI
    process_df = active_customers_df.drop(['customer_id', 'name', 'email'], axis=1)
    
    # One-Hot Encoding
    categorical_cols = ['subscription_tier', 'billing_cycle', 'auto_renew_enabled', 'customer_acquisition_channel', 'primary_device']
    process_df = pd.get_dummies(process_df, columns=categorical_cols, drop_first=True)
    
    # Ensure columns match exactly (add missing columns as 0, order them correctly)
    for col in expected_columns:
        if col not in process_df.columns:
            process_df[col] = 0
    process_df = process_df[expected_columns]
    
    # Scale numerical features using the SAVED scaler
    num_cols = ['user_age', 'tenure_months', 'days_since_last_login', 'avg_watch_time_hours', 'content_completion_rate', 'payment_failures', 'support_tickets', 'support_resolution_time_days']
    process_df[num_cols] = scaler.transform(process_df[num_cols])
    
    # 3. Predict!
    print("[*] Running AI inference (Predicting churn probabilities)...")
    probabilities = model.predict_proba(process_df)[:, 1] # Get probability of class 1 (Churn)
    
    # Add probabilities back to our readable dataframe (which still contains PII Names/Emails!)
    active_customers_df['churn_probability'] = probabilities.round(4)
    
    def assign_risk(p):
        if p > 0.75:
            return 'High Risk'
        elif p > 0.40:
            return 'Medium Risk'
        else:
            return 'Low Risk'
            
    active_customers_df['churn_risk'] = active_customers_df['churn_probability'].apply(assign_risk)
    
    # Sort them by highest risk first
    active_customers_df = active_customers_df.sort_values(by='churn_probability', ascending=False)
    
    # To keep files small and strictly for marketing, we only output the vital info
    output_columns = ['customer_id', 'name', 'email', 'churn_probability', 'churn_risk', 'tenure_months', 'days_since_last_login', 'avg_watch_time_hours', 'auto_renew_enabled']
    
    high_risk_df = active_customers_df[active_customers_df['churn_risk'] == 'High Risk'][output_columns]
    medium_risk_df = active_customers_df[active_customers_df['churn_risk'] == 'Medium Risk'][output_columns]
    low_risk_df = active_customers_df[active_customers_df['churn_risk'] == 'Low Risk'][output_columns]
    
    # 5. Save the three separate CSV files
    os.makedirs("results", exist_ok=True)
    
    high_risk_path = "results/high_risk_customers.csv"
    medium_risk_path = "results/medium_risk_customers.csv"
    low_risk_path = "results/low_risk_customers.csv"
    
    high_risk_df.to_csv(high_risk_path, index=False)
    medium_risk_df.to_csv(medium_risk_path, index=False)
    low_risk_df.to_csv(low_risk_path, index=False)
    
    print("\n" + "="*50)
    print(f"[+] Prediction Complete! Analyzed {len(active_customers_df):,} currently active customers.")
    print(f"    - High Risk (76-100%): {len(high_risk_df):,} users saved to {high_risk_path}")
    print(f"    - Medium Risk (41-75%): {len(medium_risk_df):,} users saved to {medium_risk_path}")
    print(f"    - Low Risk (0-40%): {len(low_risk_df):,} users saved to {low_risk_path}")
    print("="*50)
    
    print("\nTop 3 Highest Risk Customers (Marketing Target List):")
    print(high_risk_df[['name', 'email', 'churn_probability']].head(3).to_string(index=False))

if __name__ == "__main__":
    main()

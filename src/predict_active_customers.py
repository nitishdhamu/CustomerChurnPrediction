import pandas as pd
import numpy as np
import joblib
import os

def main():
    print("[*] Loading Current Active Subscribers database...")
    data_path = os.path.join("data", "current_active_subscribers.csv")
    
    if not os.path.exists(data_path):
        print(f"[!] Error: {data_path} not found. Ensure the CSV is in the data/ folder.")
        return
        
    active_customers_df = pd.read_csv(data_path)
    print(f"[*] Loaded {len(active_customers_df):,} active subscribers.")
    
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
    print("[*] Preprocessing active customer data...")
    process_df = active_customers_df.drop('customer_id', axis=1)
    
    # One-Hot Encoding
    categorical_cols = ['subscription_tier', 'device_type', 'auto_renew_enabled']
    process_df = pd.get_dummies(process_df, columns=categorical_cols, drop_first=True)
    
    # Ensure columns match exactly (add missing columns as 0, order them correctly)
    for col in expected_columns:
        if col not in process_df.columns:
            process_df[col] = 0
    process_df = process_df[expected_columns]
    
    # Scale numerical features using the SAVED scaler
    num_cols = ['tenure_months', 'monthly_active_days', 'avg_watch_time_hours', 'payment_failures', 'support_tickets']
    process_df[num_cols] = scaler.transform(process_df[num_cols])
    
    # 3. Predict!
    print("[*] Running AI inference (Predicting churn probabilities)...")
    probabilities = model.predict_proba(process_df)[:, 1] # Get probability of class 1 (Churn)
    
    # Add probabilities back to our readable dataframe
    active_customers_df['churn_probability'] = probabilities.round(4)
    
    # Assign risk categories based on requested thresholds
    def assign_risk(p):
        if p > 0.75:
            return 'High Risk'
        elif p > 0.40:
            return 'Medium Risk'
        else:
            return 'Low Risk'
            
    active_customers_df['churn_risk'] = active_customers_df['churn_probability'].apply(assign_risk)
    
    # 4. Split into three separate dataframes
    high_risk_df = active_customers_df[active_customers_df['churn_risk'] == 'High Risk'].sort_values(by='churn_probability', ascending=False)
    medium_risk_df = active_customers_df[active_customers_df['churn_risk'] == 'Medium Risk'].sort_values(by='churn_probability', ascending=False)
    low_risk_df = active_customers_df[active_customers_df['churn_risk'] == 'Low Risk'].sort_values(by='churn_probability', ascending=False)
    
    # 5. Save the three separate CSV files
    os.makedirs("results", exist_ok=True)
    
    high_risk_path = "results/high_risk_customers.csv"
    medium_risk_path = "results/medium_risk_customers.csv"
    low_risk_path = "results/low_risk_customers.csv"
    
    high_risk_df.to_csv(high_risk_path, index=False)
    medium_risk_df.to_csv(medium_risk_path, index=False)
    low_risk_df.to_csv(low_risk_path, index=False)
    
    # If the old single file exists, remove it to avoid confusion
    if os.path.exists("results/at_risk_customers_list.csv"):
        os.remove("results/at_risk_customers_list.csv")
    
    print("\n" + "="*50)
    print(f"[+] Prediction Complete! Analyzed {len(active_customers_df):,} total customers.")
    print(f"    - High Risk (76-100%): {len(high_risk_df):,} users saved to {high_risk_path}")
    print(f"    - Medium Risk (41-75%): {len(medium_risk_df):,} users saved to {medium_risk_path}")
    print(f"    - Low Risk (0-40%): {len(low_risk_df):,} users saved to {low_risk_path}")
    print("="*50)
    
if __name__ == "__main__":
    main()

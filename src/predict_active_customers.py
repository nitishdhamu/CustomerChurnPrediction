import pandas as pd
import numpy as np
import joblib
import os

def main():
    print("[*] Loading Current Active Subscribers database...")
    data_path = os.path.join("data", "current_active_subscribers.csv")
    
    if not os.path.exists(data_path):
        print(f"[!] Error: {data_path} not found. Please run src/generate_data.py first.")
        return
        
    active_customers_df = pd.read_csv(data_path)
    print(f"[*] Loaded {len(active_customers_df)} active subscribers.")
    
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
    active_customers_df['churn_risk'] = active_customers_df['churn_probability'].apply(
        lambda p: 'High Risk' if p >= 0.75 else ('Risk' if p >= 0.4 else 'Low Risk')
    )
    
    # 4. Sort all customers by probability and save the complete list
    all_customers_sorted = active_customers_df.sort_values(by='churn_probability', ascending=False)
    
    os.makedirs("results", exist_ok=True)
    output_path = "results/all_customers_churn_predictions.csv"
    all_customers_sorted.to_csv(output_path, index=False)
    
    # Count how many are in each category
    risk_counts = all_customers_sorted['churn_risk'].value_counts()
    
    print("\n" + "="*50)
    print(f"[+] Prediction Complete for all {len(active_customers_df):,} customers.")
    print(f"    - High Risk (>= 75%): {risk_counts.get('High Risk', 0):,}")
    print(f"    - Risk (40% - 74%):   {risk_counts.get('Risk', 0):,}")
    print(f"    - Low Risk (< 40%):   {risk_counts.get('Low Risk', 0):,}")
    print(f"[+] The full list has been saved to: {output_path}")
    print("="*50)
    
    # Preview top 5 High Risk and top 5 Low Risk
    print("\nTop 5 MOST At-Risk Customers:")
    print(all_customers_sorted[['customer_id', 'churn_probability', 'churn_risk', 'auto_renew_enabled', 'monthly_active_days']].head(5).to_string(index=False))

if __name__ == "__main__":
    main()

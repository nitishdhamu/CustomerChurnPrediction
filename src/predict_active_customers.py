import pandas as pd
import numpy as np
import joblib
import os

def generate_active_customers(num_samples=100, random_state=99):
    """
    Generates a list of CURRENTLY ACTIVE customers.
    Notice that this dataset DOES NOT have a 'churn' column, because we don't know
    if they are going to churn yet. We need the AI to predict that!
    """
    np.random.seed(random_state)
    
    customer_id = [f"ACTIVE_USER_{i:04d}" for i in range(1, num_samples + 1)]
    tenure_months = np.random.randint(1, 73, size=num_samples)
    
    tiers = ['Basic', 'Standard', 'Premium']
    subscription_tier = np.random.choice(tiers, size=num_samples, p=[0.4, 0.4, 0.2])
    
    devices = ['Mobile', 'Smart TV', 'Web', 'Multiple']
    device_type = np.random.choice(devices, size=num_samples, p=[0.35, 0.4, 0.1, 0.15])
    
    auto_renew_enabled = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.8, 0.2])
    
    monthly_active_days = np.clip(np.random.normal(loc=15, scale=10), 0, 30).astype(int)
    avg_watch_time_hours = np.clip(np.random.normal(loc=monthly_active_days * 2, scale=20), 0, 200).round(1)
    
    payment_failures = np.random.poisson(lam=0.1, size=num_samples)
    support_tickets = np.random.poisson(lam=0.5, size=num_samples)
    
    df = pd.DataFrame({
        'customer_id': customer_id,
        'tenure_months': tenure_months,
        'subscription_tier': subscription_tier,
        'monthly_active_days': monthly_active_days,
        'avg_watch_time_hours': avg_watch_time_hours,
        'device_type': device_type,
        'auto_renew_enabled': auto_renew_enabled,
        'payment_failures': payment_failures,
        'support_tickets': support_tickets
    })
    
    return df

def main():
    print("[*] Generating a list of 100 currently active customers...")
    active_customers_df = generate_active_customers(100)
    
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
    customer_ids = active_customers_df['customer_id']
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
        lambda p: 'High Risk' if p >= 0.75 else ('Medium Risk' if p >= 0.4 else 'Low Risk')
    )
    
    # 4. Filter and save the results for the Marketing Team
    at_risk_customers = active_customers_df[active_customers_df['churn_risk'] == 'High Risk'].sort_values(by='churn_probability', ascending=False)
    
    os.makedirs("results", exist_ok=True)
    output_path = "results/at_risk_customers_list.csv"
    at_risk_customers.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print(f"[+] Prediction Complete! Found {len(at_risk_customers)} High-Risk customers.")
    print(f"[+] The list has been saved to: {output_path}")
    print("="*50)
    
    # Preview top 5
    print("\nTop 5 Most At-Risk Customers:")
    print(at_risk_customers[['customer_id', 'churn_probability', 'auto_renew_enabled', 'monthly_active_days', 'avg_watch_time_hours']].head(5).to_string(index=False))

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os

def generate_user_features(num_samples, start_id, tenure_range, random_state=None):
    """Helper function to generate base features for a batch of users."""
    if random_state:
        np.random.seed(random_state)
        
    customer_id = [f"USER_{i:07d}" for i in range(start_id, start_id + num_samples)]
    tenure_months = np.random.randint(tenure_range[0], tenure_range[1], size=num_samples)
    
    tiers = ['Basic', 'Standard', 'Premium']
    subscription_tier = np.random.choice(tiers, size=num_samples, p=[0.4, 0.4, 0.2])
    
    devices = ['Mobile', 'Smart TV', 'Web', 'Multiple']
    device_type = np.random.choice(devices, size=num_samples, p=[0.35, 0.4, 0.1, 0.15])
    
    auto_renew_enabled = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.75, 0.25])
    
    monthly_active_days = np.clip(np.random.normal(loc=np.where(tenure_months > 12, 20, 12), scale=7), 0, 30).astype(int)
    avg_watch_time_hours = np.clip(np.random.normal(loc=monthly_active_days * 2.5, scale=15), 0, 200).round(1)
    
    payment_failures = np.random.poisson(lam=0.2, size=num_samples)
    support_tickets = np.random.poisson(lam=np.where(tenure_months < 6, 1.5, 0.3))
    
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

def calculate_churn(df):
    """Calculates churn based on business logic."""
    churn_prob = np.zeros(len(df))
    
    churn_prob += np.where(df['auto_renew_enabled'] == 'No', 1.5, 0.0)
    churn_prob += (df['payment_failures'] * 1.2)
    churn_prob += (df['support_tickets'] * 0.5)
    churn_prob += np.where(df['subscription_tier'] == 'Basic', 0.3, 0.0)
    
    churn_prob -= (df['monthly_active_days'] * 0.1)
    churn_prob -= (df['avg_watch_time_hours'] * 0.01)
    churn_prob -= (df['tenure_months'] * 0.02)
    
    churn_prob = 1 / (1 + np.exp(-(churn_prob - 0.5)))
    
    # Threshold for ~30-40% churn (lower threshold than before)
    threshold = np.percentile(churn_prob, 65) 
    df['churn'] = (churn_prob > threshold).astype(int)
    return df

def generate_datasets():
    print("[*] Generating Historical Data (Past Financial Year)...")
    historic_num = 100000
    historic_df = generate_user_features(historic_num, start_id=1, tenure_range=(1, 72), random_state=42)
    historic_df = calculate_churn(historic_df)
    
    retained_users = historic_df[historic_df['churn'] == 0].copy()
    churned_users = historic_df[historic_df['churn'] == 1].copy()
    print(f"    - Historical Users: {len(historic_df)}")
    print(f"    - Retained: {len(retained_users)} ({(len(retained_users)/historic_num)*100:.1f}%)")
    print(f"    - Churned: {len(churned_users)} ({(len(churned_users)/historic_num)*100:.1f}%)")
    
    print("\n[*] Generating Current Active Subscribers...")
    current_num = 120000
    
    # 50-55% of 120k is 60k-66k. We take 62k users from the retained historic pool
    overlap_count = 62000 
    
    # If we somehow have fewer retained than overlap_count, just take all retained
    if overlap_count > len(retained_users):
        overlap_count = len(retained_users)
        
    # 1. Carry over the loyal users
    carry_over_df = retained_users.sample(n=overlap_count, random_state=42).copy()
    carry_over_df = carry_over_df.drop('churn', axis=1) # They are active, churn is unknown
    
    # Simulate time passing (Add 12 months to their tenure, slightly perturb their usage)
    carry_over_df['tenure_months'] += 12
    carry_over_df['monthly_active_days'] = np.clip(carry_over_df['monthly_active_days'] + np.random.randint(-3, 4, size=overlap_count), 0, 30)
    
    # 2. Generate brand new users to fill the rest of the 120k
    new_users_count = current_num - overlap_count
    # New users get IDs starting after the historic users (100001+)
    new_users_df = generate_user_features(new_users_count, start_id=historic_num + 1, tenure_range=(1, 12), random_state=99)
    
    # Combine and shuffle
    current_active_df = pd.concat([carry_over_df, new_users_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"    - Total Current Active Users: {len(current_active_df)}")
    print(f"    - Loyal Users (From Historic): {len(carry_over_df)} ({(len(carry_over_df)/current_num)*100:.1f}%)")
    print(f"    - Brand New Users: {len(new_users_df)} ({(len(new_users_df)/current_num)*100:.1f}%)")
    
    os.makedirs("data", exist_ok=True)
    
    hist_path = os.path.join("data", "historical_data_fy25.csv")
    curr_path = os.path.join("data", "current_active_subscribers.csv")
    
    historic_df.to_csv(hist_path, index=False)
    current_active_df.to_csv(curr_path, index=False)
    
    print(f"\n[+] Successfully saved {hist_path}")
    print(f"[+] Successfully saved {curr_path}")

if __name__ == "__main__":
    generate_datasets()

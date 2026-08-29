import pandas as pd
import numpy as np
import os

def generate_streaming_data(num_samples=10000, random_state=42):
    """
    Generates a synthetic dataset for a streaming platform to predict subscription churn.
    
    This function creates realistic user profiles and engagement metrics, and uses a weighted 
    probability model to determine whether a user will churn, ensuring that the resulting dataset 
    reflects real-world business logic (e.g., users who turn off auto-renew are highly likely to churn).
    
    Args:
        num_samples (int): The number of user records to generate.
        random_state (int): Seed for reproducibility.
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing the generated user data.
    """
    print(f"[*] Initializing data generation for {num_samples} users...")
    np.random.seed(random_state)
    
    # 1. Generate independent features (Demographics & Account Details)
    customer_id = [f"USER_{i:06d}" for i in range(1, num_samples + 1)]
    tenure_months = np.random.randint(1, 73, size=num_samples) # 1 to 72 months
    
    tiers = ['Basic', 'Standard', 'Premium']
    subscription_tier = np.random.choice(tiers, size=num_samples, p=[0.4, 0.4, 0.2])
    
    devices = ['Mobile', 'Smart TV', 'Web', 'Multiple']
    device_type = np.random.choice(devices, size=num_samples, p=[0.35, 0.4, 0.1, 0.15])
    
    auto_renew_enabled = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.75, 0.25])
    
    # 2. Generate dependent-ish features (Engagement & Support)
    # Users with longer tenure tend to be more active
    monthly_active_days = np.clip(np.random.normal(loc=np.where(tenure_months > 12, 20, 12), scale=7), 0, 30).astype(int)
    
    # Watch time correlates with active days
    avg_watch_time_hours = np.clip(np.random.normal(loc=monthly_active_days * 2.5, scale=15), 0, 200).round(1)
    
    # Payment failures are rare but highly impactful
    payment_failures = np.random.poisson(lam=0.2, size=num_samples)
    
    # Newer users tend to submit more support tickets
    support_tickets = np.random.poisson(lam=np.where(tenure_months < 6, 1.5, 0.3))
    
    # 3. Calculate Churn Probability
    churn_prob = np.zeros(num_samples)
    
    # Positive factors (Increases Churn Risk)
    churn_prob += np.where(auto_renew_enabled == 'No', 1.5, 0.0) # High risk if auto-renew is off
    churn_prob += (payment_failures * 1.2)                       # High risk if payment fails
    churn_prob += (support_tickets * 0.5)                        # Frustrated users
    churn_prob += np.where(subscription_tier == 'Basic', 0.3, 0.0) # Basic users churn slightly more
    
    # Negative factors (Decreases Churn Risk)
    churn_prob -= (monthly_active_days * 0.1)                    # Active users stay
    churn_prob -= (avg_watch_time_hours * 0.01)                  # Engaged users stay
    churn_prob -= (tenure_months * 0.02)                         # Loyal users stay
    
    # Normalize probabilities using a sigmoid function to keep them between 0 and 1
    churn_prob = 1 / (1 + np.exp(-(churn_prob - 0.5)))
    
    # 4. Generate highly imbalanced churn labels (~20% churn)
    threshold = np.percentile(churn_prob, 80)
    churn = (churn_prob > threshold).astype(int)
    
    # 5. Assemble DataFrame
    df = pd.DataFrame({
        'customer_id': customer_id,
        'tenure_months': tenure_months,
        'subscription_tier': subscription_tier,
        'monthly_active_days': monthly_active_days,
        'avg_watch_time_hours': avg_watch_time_hours,
        'device_type': device_type,
        'auto_renew_enabled': auto_renew_enabled,
        'payment_failures': payment_failures,
        'support_tickets': support_tickets,
        'churn': churn
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_streaming_data(num_samples=10000)
    
    output_path = os.path.join("data", "streaming_churn_data.csv")
    df.to_csv(output_path, index=False)
    
    print(f"[*] Data generation complete. Saved to '{output_path}'")
    print(f"[*] Dataset shape: {df.shape}")
    print("[*] Churn Distribution:")
    print(df['churn'].value_counts(normalize=True) * 100)

import pandas as pd
import numpy as np
import os

def generate_customer_data(num_samples=5000, random_state=42):
    np.random.seed(random_state)
    
    # Generate independent features
    customer_id = [f"CUST_{i:05d}" for i in range(1, num_samples + 1)]
    tenure_months = np.random.randint(1, 73, size=num_samples)
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=num_samples), 2)
    contract_types = ['Month-to-month', 'One year', 'Two year']
    contract_type = np.random.choice(contract_types, size=num_samples, p=[0.5, 0.3, 0.2])
    
    payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
    payment_method = np.random.choice(payment_methods, size=num_samples)
    
    total_usage_gb = np.round(np.random.uniform(10.0, 500.0, size=num_samples), 2)
    
    # Generate dependent-ish features (simulating realistic correlations)
    # Customers with longer tenure might have fewer support calls on average
    support_calls = np.random.poisson(lam=np.where(tenure_months < 12, 2.5, 0.8))
    
    # Activity logs might be higher for more engaged customers
    activity_log_count = np.random.poisson(lam=np.where(tenure_months > 24, 50, 20))
    
    # Calculate churn probability based on features
    # Base probability
    churn_prob = np.zeros(num_samples)
    
    # Adjust probability based on features
    churn_prob += np.where(contract_type == 'Month-to-month', 0.3, 0.0)
    churn_prob += np.where(contract_type == 'Two year', -0.2, 0.0)
    
    churn_prob += (support_calls * 0.1) # More calls -> higher churn
    churn_prob -= (tenure_months * 0.005) # Longer tenure -> lower churn
    churn_prob -= (activity_log_count * 0.002) # More activity -> lower churn
    churn_prob += (monthly_charges * 0.001) # Higher charges -> slightly higher churn
    
    # Normalize probabilities to be between 0 and 1
    churn_prob = 1 / (1 + np.exp(-(churn_prob - 1))) # Sigmoid function to smooth
    
    # Generate churn labels (imbalanced)
    # We want roughly 15-20% churn
    threshold = np.percentile(churn_prob, 80)
    churn = (churn_prob > threshold).astype(int)
    
    df = pd.DataFrame({
        'customer_id': customer_id,
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'total_usage_gb': total_usage_gb,
        'contract_type': contract_type,
        'payment_method': payment_method,
        'support_calls': support_calls,
        'activity_log_count': activity_log_count,
        'churn': churn
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("Generating synthetic customer data...")
    df = generate_customer_data(num_samples=10000)
    
    output_path = os.path.join("data", "customer_churn_data.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Data generation complete. Saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Churn distribution:\n{df['churn'].value_counts(normalize=True) * 100}")

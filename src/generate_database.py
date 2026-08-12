import pandas as pd
import numpy as np
import os

def generate_names_and_emails(num_samples):
    """Generates realistic fake names and completely randomized emails."""
    first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth', 'David', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica', 'Thomas', 'Sarah', 'Charles', 'Karen', 'Alex', 'Sam', 'Taylor', 'Jordan', 'Casey']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzales', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin']
    domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com']
    
    random_words = ['skater', 'gamer', 'star', 'blue', 'red', 'ninja', 'shadow', 'cool', 'super', 'music', 'coder', 'hero', 'pizza', 'sunny', 'moon', 'coffee', 'tech', 'happy', 'swift', 'magic']
    
    f_names = np.random.choice(first_names, num_samples)
    l_names = np.random.choice(last_names, num_samples)
    names = np.char.add(np.char.add(f_names, ' '), l_names)
    
    words1 = np.random.choice(random_words, num_samples)
    words2 = np.random.choice(random_words, num_samples)
    random_nums = np.random.randint(100, 9999, num_samples).astype(str)
    email_domains = np.random.choice(domains, num_samples)
    
    emails = np.char.add(words1, words2)
    emails = np.char.add(emails, random_nums)
    emails = np.char.add(np.char.add(emails, '@'), email_domains)
    
    return names, emails

def generate_million_user_database():
    print("[*] Initializing Database Generation Engine...")
    
    num_samples = np.random.randint(980000, 1050000)
    print(f"[*] Simulating {num_samples:,} users over the past 5 years...")
    
    np.random.seed(42)
    
    customer_id = [f"USER_{i:07d}" for i in range(1, num_samples + 1)]
    names, emails = generate_names_and_emails(num_samples)
    
    tenure_months = np.random.randint(1, 61, size=num_samples)
    user_age = np.clip(np.random.normal(loc=35, scale=12, size=num_samples), 18, 80).astype(int)
    
    tiers = ['Basic', 'Standard', 'Premium']
    subscription_tier = np.random.choice(tiers, size=num_samples, p=[0.4, 0.4, 0.2])
    
    billing_cycle = np.random.choice(['Monthly', 'Annual'], size=num_samples, p=[0.8, 0.2])
    auto_renew_enabled = np.where(billing_cycle == 'Annual', 
                                  np.random.choice(['Yes', 'No'], size=num_samples, p=[0.9, 0.1]),
                                  np.random.choice(['Yes', 'No'], size=num_samples, p=[0.7, 0.3]))
                                  
    acq_channel = ['Organic', 'Social_Ads', 'Referral', 'Promo_Code']
    customer_acquisition_channel = np.random.choice(acq_channel, size=num_samples, p=[0.4, 0.3, 0.1, 0.2])
    
    primary_device = np.random.choice(['Mobile', 'Smart TV', 'Web', 'Console'], size=num_samples, p=[0.4, 0.4, 0.1, 0.1])
    
    days_since_last_login = np.clip(np.random.exponential(scale=7, size=num_samples), 0, 60).astype(int)
    
    base_watch = np.where(primary_device == 'Smart TV', 40, 20)
    avg_watch_time_hours = np.clip(np.random.normal(loc=base_watch, scale=15), 0, 300).round(1)
    
    content_completion_rate = np.clip(np.random.normal(loc=0.6, scale=0.2, size=num_samples), 0.0, 1.0).round(2)
    
    payment_failures = np.random.poisson(lam=0.1, size=num_samples)
    
    ticket_lam = np.where(tenure_months < 3, 1.0, 0.2)
    support_tickets = np.random.poisson(lam=ticket_lam)
    support_resolution_time_days = np.where(support_tickets > 0, 
                                            np.clip(np.random.normal(loc=2, scale=2), 0, 14).astype(int), 
                                            0)
    
    df = pd.DataFrame({
        'customer_id': customer_id,
        'name': names,
        'email': emails,
        'user_age': user_age,
        'tenure_months': tenure_months,
        'subscription_tier': subscription_tier,
        'billing_cycle': billing_cycle,
        'auto_renew_enabled': auto_renew_enabled,
        'customer_acquisition_channel': customer_acquisition_channel,
        'primary_device': primary_device,
        'days_since_last_login': days_since_last_login,
        'avg_watch_time_hours': avg_watch_time_hours,
        'content_completion_rate': content_completion_rate,
        'payment_failures': payment_failures,
        'support_tickets': support_tickets,
        'support_resolution_time_days': support_resolution_time_days
    })
    
    print("[*] Applying complex business logic to determine Churn status...")
    churn_risk = np.zeros(num_samples)
    
    churn_risk += (df['days_since_last_login'] ** 1.2) * 0.05
    churn_risk += np.where((df['customer_acquisition_channel'] == 'Promo_Code') & (df['billing_cycle'] == 'Monthly') & (df['tenure_months'] <= 2), 2.5, 0.0)
    churn_risk += np.where((df['subscription_tier'] == 'Premium') & (df['avg_watch_time_hours'] < 15), 3.0, 0.0)
    churn_risk += (df['payment_failures'] * 1.5)
    churn_risk += np.where(df['auto_renew_enabled'] == 'No', 2.0, 0.0)
    churn_risk += np.where((df['support_tickets'] > 0) & (df['support_resolution_time_days'] > 3), df['support_resolution_time_days'] * 0.5, 0.0)
    churn_risk -= (df['content_completion_rate'] * 3.0)
    churn_risk -= np.log1p(df['tenure_months']) * 0.8
    churn_risk += np.random.normal(loc=0, scale=3.5, size=num_samples)
    
    churn_prob = 1 / (1 + np.exp(-(churn_risk - 1.0)))
    threshold = np.percentile(churn_prob, 65) 
    df['churn'] = (churn_prob > threshold).astype(int)
    
    os.makedirs("data", exist_ok=True)
    out_path = "data/streaming_users_database.csv"
    print(f"[*] Saving massive database to {out_path}...")
    df.to_csv(out_path, index=False)
    print(f"[+] Successfully generated {num_samples:,} users!")

if __name__ == "__main__":
    generate_million_user_database()

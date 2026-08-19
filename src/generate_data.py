import pandas as pd
import numpy as np
import os
import string
import warnings

from faker import Faker

# Suppress all warnings for clean terminal UI
warnings.filterwarnings('ignore')

def generate_names_and_emails(num_samples, start_id=1):
    print("    -> Generating unique demographic profiles...")
    fake = Faker(['en_US', 'en_GB'])
    
    first_names_set = set()
    last_names_set = set()
    for _ in range(20000):
        first_names_set.add(fake.first_name())
        last_names_set.add(fake.last_name())
        
    first_names = sorted(list(first_names_set))
    last_names = sorted(list(last_names_set))
    middle_initials = [f" {c}. " for c in string.ascii_uppercase]
    
    domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com']
    
    random_words = ['skater', 'gamer', 'star', 'blue', 'red', 'ninja', 'shadow', 'cool', 'super', 'music', 'coder', 'hero', 'pizza', 'sunny', 'moon', 'coffee', 'tech', 'happy', 'swift', 'magic']
    
    f_names = np.random.choice(first_names, num_samples)
    m_names = np.random.choice(middle_initials, num_samples)
    l_names = np.random.choice(last_names, num_samples)
    
    names = np.char.add(np.char.add(f_names, m_names), l_names)
    
    words1 = np.random.choice(random_words, num_samples)
    words2 = np.random.choice(random_words, num_samples)
    
    unique_ids = np.arange(start_id, start_id + num_samples).astype(str)
    email_domains = np.random.choice(domains, num_samples)
    
    emails = np.char.add(words1, words2)
    emails = np.char.add(emails, unique_ids)
    emails = np.char.add(np.char.add(emails, '@'), email_domains)
    
    return names, emails

def generate_streaming_database(platform_name, min_users, max_users, start_id, seed):
    print(f"\n[*] --------------------------------------------------")
    print(f"[*] BUILDING DATASET: {platform_name.upper()}")
    print(f"[*] --------------------------------------------------")
    
    np.random.seed(seed)
    Faker.seed(seed)
    
    num_samples = np.random.randint(min_users, max_users)
    print(f"    -> Target Size: {num_samples:,} simulated users.")
    
    customer_id = [f"{platform_name[:3].upper()}_{i:08d}" for i in range(1, num_samples + 1)]
    names, emails = generate_names_and_emails(num_samples, start_id)
    
    print("    -> Simulating 5 years of historical activity...")
    tenure_months = np.random.randint(1, 61, size=num_samples)
    age = np.clip(np.random.normal(loc=35, scale=12, size=num_samples), 18, 80).astype(int)
    
    billing_cycle = np.random.choice(['Monthly', 'Annual'], size=num_samples, p=[0.8, 0.2])
    auto_renew_enabled = np.where(billing_cycle == 'Annual', 
                                  np.random.choice(['Yes', 'No'], size=num_samples, p=[0.9, 0.1]),
                                  np.random.choice(['Yes', 'No'], size=num_samples, p=[0.7, 0.3]))
                                  
    days_since_last_login = np.clip(np.random.exponential(scale=7, size=num_samples), 0, 60).astype(int)
    
    base_watch = 30
    avg_watch_time_hours = np.clip(np.random.normal(loc=base_watch, scale=15), 0, 300).round(1)
    
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
        'age': age,
        'tenure_months': tenure_months,
        'billing_cycle': billing_cycle,
        'auto_renew_enabled': auto_renew_enabled,
        'days_since_last_login': days_since_last_login,
        'avg_watch_time_hours': avg_watch_time_hours,
        'payment_failures': payment_failures,
        'support_tickets': support_tickets,
        'support_resolution_time_days': support_resolution_time_days
    })
    
    print("    -> Calculating churn risk trajectories...")
    churn_risk = np.zeros(num_samples)
    
    churn_risk += (df['days_since_last_login'] ** 1.2) * 0.05
    churn_risk += (df['payment_failures'] * 1.5)
    churn_risk += np.where(df['auto_renew_enabled'] == 'No', 2.0, 0.0)
    churn_risk += np.where((df['support_tickets'] > 0) & (df['support_resolution_time_days'] > 3), df['support_resolution_time_days'] * 0.5, 0.0)
    churn_risk -= np.log1p(df['tenure_months']) * 0.8
    churn_risk += np.random.normal(loc=0, scale=3.5, size=num_samples)
    
    churn_prob = 1 / (1 + np.exp(-(churn_risk - 1.0)))
    threshold = np.percentile(churn_prob, 65) 
    df['churn'] = (churn_prob > threshold).astype(int)
    
    os.makedirs("data", exist_ok=True)
    out_path = f"data/{platform_name}_dataset.csv"
    print(f"    -> Saving compiled database to {out_path}...")
    df.to_csv(out_path, index=False)
    print(f"[+] COMPLETE: {platform_name} Dataset Generated.")
    
    return num_samples

def get_interactive_platforms(platforms):
    print("\n[?] SELECT TARGET PLATFORMS")
    print("--------------------------------------------------")
    for i, plat in enumerate(platforms, 1):
        print(f"    {i}. {plat[0]}")
    print(f"    {len(platforms) + 1}. All Platforms")
    print("--------------------------------------------------")
    
    choice = input("\nEnter choice (e.g., '1', '1,3', or 'All'): ").strip()
    
    if choice.lower() == 'all' or choice == str(len(platforms) + 1):
        return platforms
        
    selected_platforms = []
    for c in choice.split(','):
        c = c.strip()
        if c.isdigit():
            idx = int(c) - 1
            if 0 <= idx < len(platforms):
                selected_platforms.append(platforms[idx])
        else:
            for p in platforms:
                if p[0].lower() == c.lower():
                    selected_platforms.append(p)
                    
    return selected_platforms

if __name__ == "__main__":
    print("\n==================================================")
    print("     SYNTHETIC DATA GENERATION ENGINE")
    print("==================================================")
    
    all_platforms = [
        ('Netflix', 2000000, 2200000),
        ('Prime', 1500000, 1600000),
        ('JioHotstar', 1200000, 1300000),
        ('AppleTV', 1000000, 1100000)
    ]
    
    platforms_to_run = get_interactive_platforms(all_platforms)
    
    if not platforms_to_run:
        print("\n[!] Operation cancelled. No platforms selected.\n")
    else:
        start_id = 1
        seed_val = 42
        for plat, min_u, max_u in platforms_to_run:
            num_gen = generate_streaming_database(plat, min_u, max_u, start_id, seed_val)
            start_id += num_gen
            seed_val += 1
        
        print("\n==================================================")
        print("  ALL REQUESTED DATASETS GENERATED SUCCESSFULLY")
        print("==================================================\n")

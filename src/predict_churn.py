import pandas as pd
import numpy as np
import os
import joblib
import glob

def get_interactive_models():
    model_files = glob.glob('models/*_model.pkl')
    if not model_files:
        print("[!] No AI models found in the models/ folder. Please run train_models.py first.")
        return []
    
    model_files = sorted(model_files)
    prefixes = [os.path.basename(m).replace('_model.pkl', '') for m in model_files]
    
    if len(prefixes) == 1:
        # User requested to "only list the csv files like that"
        # Map the prefix to the CSV file display
        csv_files = glob.glob(f"data/{prefixes[0]}*.csv")
        display_name = os.path.basename(csv_files[0]) if csv_files else f"{prefixes[0]} (CSV missing)"
        print(f"\n[*] Only one dataset model found: {display_name}. Automatically selecting it...")
        return prefixes

    print("\n[*] Available Datasets (with Trained Models):")
    for i, p in enumerate(prefixes, 1):
        csv_files = glob.glob(f"data/{p}*.csv")
        display_name = os.path.basename(csv_files[0]) if csv_files else f"{p} (CSV missing)"
        print(f"    {i}. {display_name}")
    print(f"    {len(prefixes) + 1}. All Datasets")
    
    choice = input("\n[?] Select datasets to run inference with (e.g., '1', '1,3', or 'All'): ").strip()
    
    if choice.lower() == 'all' or choice == str(len(prefixes) + 1):
        return prefixes
        
    selected_prefixes = []
    for c in choice.split(','):
        c = c.strip()
        if c.isdigit():
            idx = int(c) - 1
            if 0 <= idx < len(prefixes):
                selected_prefixes.append(prefixes[idx])
        else:
            for p in prefixes:
                # User might type the CSV name or the prefix name
                csv_files = glob.glob(f"data/{p}*.csv")
                display_name = os.path.basename(csv_files[0]) if csv_files else p
                if c.lower() == display_name.lower() or c.lower() == p.lower():
                    selected_prefixes.append(p)
                    
    return selected_prefixes

def predict_for_dataset(prefix):
    # Find the corresponding dataset in the data folder
    potential_files = glob.glob(f"data/{prefix}*.csv")
    if not potential_files:
        print(f"\n[!] Error: No dataset found in data/ starting with '{prefix}'.")
        return
        
    # Default to the first match
    filepath = potential_files[0]
    
    print(f"\n[*] Preprocessing active customer data for {os.path.basename(filepath)} using {prefix} model...")
    
    model_path = f"models/{prefix}_model.pkl"
    model_bundle = joblib.load(model_path)
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_columns = model_bundle['features']
    
    df = pd.read_csv(filepath)
    
    if 'churn' not in df.columns:
        # If it's a completely new unseen dataset without churn labels
        active_customers = df.copy()
    else:
        active_customers = df[df['churn'] == 0].copy()
        
    print(f"[*] Found {len(active_customers):,} currently active subscribers.")
    
    # Drop identifying columns that shouldn't be predicted on
    cols_to_drop = ['customer_id', 'name', 'email', 'churn']
    inference_df = active_customers.drop([c for c in cols_to_drop if c in active_customers.columns], axis=1)
    
    categorical_cols = ['billing_cycle', 'auto_renew_enabled']
    inference_df = pd.get_dummies(inference_df, columns=categorical_cols, drop_first=True)
    
    for col in feature_columns:
        if col not in inference_df.columns:
            inference_df[col] = 0
            
    inference_df = inference_df[feature_columns]
    
    num_cols = ['age', 'tenure_months', 'days_since_last_login', 'avg_watch_time_hours', 'payment_failures', 'support_tickets', 'support_resolution_time_days']
    inference_df[num_cols] = scaler.transform(inference_df[num_cols])
    
    print(f"[*] Running AI inference...")
    churn_probs = model.predict_proba(inference_df)[:, 1]
    
    active_customers['churn_probability'] = np.round(churn_probs, 4)
    active_customers = active_customers.sort_values(by='churn_probability', ascending=False)
    
    high_risk = active_customers[active_customers['churn_probability'] > 0.75]
    medium_risk = active_customers[(active_customers['churn_probability'] > 0.40) & (active_customers['churn_probability'] <= 0.75)]
    low_risk = active_customers[active_customers['churn_probability'] <= 0.40]
    
    os.makedirs("results", exist_ok=True)
    high_risk.to_csv(f"results/{prefix}_high_risk.csv", index=False)
    medium_risk.to_csv(f"results/{prefix}_medium_risk.csv", index=False)
    low_risk.to_csv(f"results/{prefix}_low_risk.csv", index=False)
    
    print("="*77)
    print(f"[+] Prediction Complete for {prefix}!")
    print(f"    - High Risk (76-100%): {len(high_risk):,} users")
    print(f"    - Medium Risk (41-75%): {len(medium_risk):,} users")
    print(f"    - Low Risk (0-40%): {len(low_risk):,} users")
    print("="*77)

def main():
    print("[*] Starting AI inference engine...")
    
    prefixes_to_run = get_interactive_models()
    
    if not prefixes_to_run:
        print("[!] No models selected. Exiting.")
        return

    for prefix in prefixes_to_run:
        predict_for_dataset(prefix)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve

from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(filepath):
    """
    Loads the streaming dataset, encodes categorical features, and scales numerical features.
    It returns the fully preprocessed training and testing sets.
    """
    print(f"[*] Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Drop customer_id as it's an identifier, not a predictive feature
    df = df.drop('customer_id', axis=1)
    
    # 1. Encode categorical variables using One-Hot Encoding
    print("[*] Encoding categorical features...")
    categorical_cols = ['subscription_tier', 'device_type', 'auto_renew_enabled']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 2. Separate features (X) and target (y)
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # 3. Split the data (80% train, 20% test). 'stratify=y' ensures the churn ratio is maintained in both splits.
    print("[*] Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Scale numerical features (important for Neural Networks and Logistic Regression)
    print("[*] Scaling numerical features...")
    scaler = StandardScaler()
    num_cols = ['tenure_months', 'monthly_active_days', 'avg_watch_time_hours', 'payment_failures', 'support_tickets']
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit the scaler ONLY on the training data to prevent data leakage, then transform both
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def evaluate_model(model, X_test, y_test, model_name):
    """
    Predicts the test set using the trained model and calculates performance metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0]*len(y_test)
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred), # Critical metric for churn (catching true cancellations)
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob) if hasattr(model, 'predict_proba') else None
    }
    
    print(f"\n--- {model_name} Performance ---")
    for k, v in metrics.items():
        if k != 'Model':
            print(f"{k}: {v:.4f}")
            
    return metrics, y_pred, y_prob

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Saves a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_roc_curves(models_probs, y_test, filename):
    """Saves an overlapping ROC curve plot for all trained models."""
    plt.figure(figsize=(8, 6))
    for model_name, y_prob in models_probs.items():
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Streaming Churn)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    os.makedirs("results", exist_ok=True)
    data_path = os.path.join("data", "streaming_churn_data.csv")
    
    if not os.path.exists(data_path):
        print(f"[!] Error: Data file not found at {data_path}. Please run generate_data.py first.")
        return
        
    # 1. Load and Preprocess
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(data_path)
    
    print(f"\n[*] Original training set class distribution: 0 (Retained): {sum(y_train==0)}, 1 (Churned): {sum(y_train==1)}")
    
    # 2. Apply SMOTE to handle class imbalance
    print("[*] Applying SMOTE to balance the training data...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"[*] SMOTE training set class distribution: 0: {sum(y_train_smote==0)}, 1: {sum(y_train_smote==1)}")
    
    # 3. Initialize Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Neural Network": MLPClassifier(random_state=42, hidden_layer_sizes=(64, 32), max_iter=500)
    }
    
    results = []
    models_probs = {}
    
    # 4. Train, Evaluate, and Plot
    print("\n[*] Training models on SMOTE balanced data...")
    for name, model in models.items():
        model.fit(X_train_smote, y_train_smote)
        
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        models_probs[name] = y_prob
        
        # Save confusion matrix
        plot_confusion_matrix(y_test, y_pred, name, f"results/cm_{name.replace(' ', '_').lower()}.png")
        
    # 5. Generate aggregate plots and summary
    print("\n[*] Generating ROC curves...")
    plot_roc_curves(models_probs, y_test, "results/roc_curves.png")
    
    # Save CSV Summary
    results_df = pd.DataFrame(results)
    print("\n==== Summary of Results ====")
    print(results_df.to_string(index=False))
    results_df.to_csv("results/model_comparison.csv", index=False)
    
    # Extract Feature Importance (from Decision Tree)
    print("\n[*] Extracting feature importance...")
    dt_model = models["Decision Tree"]
    importances = dt_model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title('Feature Importance (Streaming Subscriptions)')
    plt.tight_layout()
    plt.savefig("results/feature_importance.png")
    plt.close()
    
    print("\n[+] Pipeline completed successfully! Check the 'results/' folder for plots and summary.")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # NEW: For saving the trained model and scaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve

from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(filepath):
    print(f"[*] Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    df = df.drop('customer_id', axis=1)
    
    print("[*] Encoding categorical features...")
    categorical_cols = ['subscription_tier', 'device_type', 'auto_renew_enabled']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Save the exact columns used during training so we can match them during prediction
    feature_columns = df.drop('churn', axis=1).columns
    os.makedirs("models", exist_ok=True)
    joblib.dump(feature_columns, "models/feature_columns.pkl")
    
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    print("[*] Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[*] Scaling numerical features...")
    scaler = StandardScaler()
    num_cols = ['tenure_months', 'monthly_active_days', 'avg_watch_time_hours', 'payment_failures', 'support_tickets']
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    # NEW: Save the fitted scaler so we can scale new, active customers the exact same way
    joblib.dump(scaler, "models/scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0]*len(y_test)
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob) if hasattr(model, 'predict_proba') else None
    }
    
    print(f"\n--- {model_name} Performance ---")
    for k, v in metrics.items():
        if k != 'Model':
            print(f"{k}: {v:.4f}")
            
    return metrics, y_pred, y_prob

def plot_confusion_matrix(y_true, y_pred, title, filename):
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
    os.makedirs("models", exist_ok=True)
    
    data_path = os.path.join("data", "streaming_churn_data.csv")
    if not os.path.exists(data_path):
        print(f"[!] Error: Data file not found at {data_path}. Please run generate_data.py first.")
        return
        
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(data_path)
    
    print("[*] Applying SMOTE to balance the training data...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Neural Network": MLPClassifier(random_state=42, hidden_layer_sizes=(64, 32), max_iter=500)
    }
    
    results = []
    models_probs = {}
    
    print("\n[*] Training models on SMOTE balanced data...")
    for name, model in models.items():
        model.fit(X_train_smote, y_train_smote)
        
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        models_probs[name] = y_prob
        
        plot_confusion_matrix(y_test, y_pred, name, f"results/cm_{name.replace(' ', '_').lower()}.png")
        
        # NEW: Save the Neural Network model as our final production model
        if name == "Neural Network":
            joblib.dump(model, "models/best_model_nn.pkl")
            print("[*] Saved Neural Network model to models/best_model_nn.pkl")
            
    print("\n[*] Generating ROC curves...")
    plot_roc_curves(models_probs, y_test, "results/roc_curves.png")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/model_comparison.csv", index=False)
    
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
    
    print("\n[+] Training Pipeline completed successfully!")

if __name__ == "__main__":
    main()

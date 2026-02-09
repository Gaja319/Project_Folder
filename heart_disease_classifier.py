"""
Heart Disease Classification using Machine Learning Models
BITS Semester 2 - ML Assignment 2 (2024DC04051)

This module implements and compares 6 different ML classifiers:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest
- XGBoost

The models are trained on the UCI Heart Disease dataset to predict
the presence of heart disease (binary classification).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


def load_and_preprocess_data(csv_file):
    """
    Load and preprocess the heart disease dataset.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
        
    Returns:
    --------
    tuple : (X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler)
    """
    # Load dataset
    df = pd.read_csv(csv_file)
    print(f"Dataset Shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    
    # Target variable handling
    if "num" in df.columns and "target" not in df.columns:
        df["target"] = df["num"].apply(lambda x: 0 if x == 0 else 1)
    
    if "num" in df.columns:
        df.drop("num", axis=1, inplace=True)
    
    # Drop non-predictive columns
    for col in ["id", "dataset"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    # Handle boolean & categorical features
    for col in ["fbs", "exang"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = df[col].astype(int)
    
    # One-hot encode categorical columns
    categorical_cols = ["sex", "cp", "restecg", "slope", "thal"]
    existing_cats = [c for c in categorical_cols if c in df.columns]
    
    if existing_cats:
        df = pd.get_dummies(df, columns=existing_cats, drop_first=True)
    
    # Final numeric NaN handling
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    print("\nPreprocessing complete (OK).")
    print(f"Final shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}\n")
    
    # Feature-Target Split
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
        
    Returns:
    --------
    dict : Performance metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }


def train_and_evaluate_models(X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test):
    """
    Train and evaluate all 6 models.
    
    Parameters:
    -----------
    X_train_scaled, X_test_scaled : array-like
        Scaled train and test features (for algorithms requiring scaling)
    X_train, X_test : array-like
        Unscaled train and test features
    y_train, y_test : array-like
        Train and test labels
        
    Returns:
    --------
    tuple : (results_df, models_dict)
    """
    models_dict = {}
    metrics_dict = {}
    
    print("=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)
    
    # 1. Logistic Regression
    print("\n[1/6] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    models_dict["Logistic Regression"] = lr
    metrics_dict["Logistic Regression"] = evaluate_model(lr, X_test_scaled, y_test)
    print("[DONE] Complete")
    
    # 2. Decision Tree
    print("[2/6] Training Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    models_dict["Decision Tree"] = dt
    metrics_dict["Decision Tree"] = evaluate_model(dt, X_test, y_test)
    print("[DONE] Complete")
    
    # 3. K-Nearest Neighbors
    print("[3/6] Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    models_dict["KNN"] = knn
    metrics_dict["KNN"] = evaluate_model(knn, X_test_scaled, y_test)
    print("[DONE] Complete")
    
    # 4. Naive Bayes (Gaussian)
    print("[4/6] Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    models_dict["Naive Bayes"] = nb
    metrics_dict["Naive Bayes"] = evaluate_model(nb, X_test_scaled, y_test)
    print("[DONE] Complete")
    
    # 5. Random Forest (Ensemble)
    print("[5/6] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models_dict["Random Forest"] = rf
    metrics_dict["Random Forest"] = evaluate_model(rf, X_test, y_test)
    print("[DONE] Complete")
    
    # 6. XGBoost (Ensemble)
    print("[6/6] Training XGBoost...")
    xgb = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    models_dict["XGBoost"] = xgb
    metrics_dict["XGBoost"] = evaluate_model(xgb, X_test, y_test)
    print("[DONE] Complete")
    
    # Create results dataframe
    results_df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    
    return results_df, models_dict


def display_results(results_df):
    """
    Display model comparison results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe with metrics for all models
    """
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{results_df.round(4)}")
    
    # Find best performers
    print("\n" + "=" * 70)
    print("BEST PERFORMERS")
    print("=" * 70)
    for metric in results_df.columns:
        best_model = results_df[metric].idxmax()
        best_score = results_df[metric].max()
        print(f"{metric:15s}: {best_model:20s} ({best_score:.4f})")


def main(csv_file="heart_disease_uci.csv"):
    """
    Main execution function.
    
    Parameters:
    -----------
    csv_file : str
        Path to the heart disease dataset CSV file
    """
    print("=" * 70)
    print("HEART DISEASE CLASSIFICATION - ML MODEL COMPARISON")
    print("=" * 70)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler = \
        load_and_preprocess_data(csv_file)
    
    # Train and evaluate models
    results_df, models_dict = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test
    )
    
    # Display results
    display_results(results_df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70 + "\n")
    
    return results_df, models_dict


if __name__ == "__main__":
    results, models = main()

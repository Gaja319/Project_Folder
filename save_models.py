"""
Train models using the existing pipeline and save them to the `model/` folder.
Produces: model/<ModelName>.pkl, model/scaler.pkl, model/metrics_summary.csv
"""
import os
from joblib import dump
from heart_disease_classifier import load_and_preprocess_data, train_and_evaluate_models


def main():
    csv_file = "heart_disease_uci.csv"
    X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(csv_file)
    results_df, models_dict = train_and_evaluate_models(X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test)

    os.makedirs("model", exist_ok=True)

    # Save models
    for name, model in models_dict.items():
        safe_name = name.replace(" ", "_").replace("/", "_")
        path = os.path.join("model", f"{safe_name}.pkl")
        dump(model, path)
        print(f"Saved model: {path}")

    # Save scaler
    dump(scaler, os.path.join("model", "scaler.pkl"))
    print("Saved scaler: model/scaler.pkl")

    # Save metrics
    results_df.to_csv(os.path.join("model", "metrics_summary.csv"))
    print("Saved metrics: model/metrics_summary.csv")

    print("All models and artifacts saved in the model/ directory.")


if __name__ == "__main__":
    main()

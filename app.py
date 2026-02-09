import streamlit as st
import pandas as pd
import os
from joblib import load
from heart_disease_classifier import load_and_preprocess_data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from save_models import main as save_models_main

st.set_page_config(page_title="Heart Disease Classifier", layout="wide")

st.title("Heart Disease Classification — Model Explorer")

# Paths relative to this file (helps on Streamlit Cloud)
BASE_DIR = Path(__file__).resolve().parent
model_dir = BASE_DIR / "model"

# Load available models
models = []
if model_dir.is_dir():
    models = [f.name for f in model_dir.iterdir() if f.suffix == '.pkl' and f.name != 'scaler.pkl']
    models = sorted(models)

st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload Heart Disease Dataset (CSV)", type=["csv"]) 
selected_model_file = st.sidebar.selectbox("Select model", options=models if models else ["(no models found)"])

if not models:
    st.warning("No saved models found in the 'model/' directory. Ensure model files are committed to the repository.")

# Load metrics summary if exists
metrics_path = model_dir / "metrics_summary.csv"
metrics_df = None
if metrics_path.exists():
    metrics_df = pd.read_csv(metrics_path, index_col=0)

# Prepare data (require uploaded CSV)
if uploaded_file is None:
    st.info("👈 Please upload a Heart Disease dataset CSV file in the sidebar to begin.")
    st.stop()

try:
    st.info("Processing uploaded dataset...")
    temp_path = BASE_DIR / "uploaded_tmp.csv"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(str(temp_path))
except FileNotFoundError as e:
    st.error(f"Dataset file error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading or preprocessing the dataset: {e}")
    st.stop()

# Display metrics table
st.header("Model comparison — saved metrics")
if metrics_df is not None:
    st.dataframe(metrics_df.style.format(precision=4))
else:
    st.write("No metrics_summary.csv found in model/. Run save_models.py to generate it.")

st.header("Model Evaluation")
if selected_model_file and selected_model_file != "(no models found)":
    # Verify model file exists before attempting to load
    model_path = model_dir / selected_model_file
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.warning("**To fix this on Streamlit Cloud:**\n\n1. Run locally: `python save_models.py`\n\n2. Commit and push:\n```bash\ngit add model/\ngit commit -m 'Add saved models'\ngit push origin main\n```\n\n3. Re-deploy the app on Streamlit Cloud dashboard\n\n4. Wait for deployment to complete and refresh the page.")
        st.stop()
    
    model = load(str(model_path))
    st.subheader(f"Selected model: {selected_model_file}")

    # Decide whether model requires scaled input (Logistic, KNN, NaiveBayes were trained on scaled data)
    model_key = selected_model_file.lower()
    scaled_needed = any(k in model_key for k in ['logistic', 'knn', 'naive', 'nb'])
    X_eval = X_test_scaled if scaled_needed else X_test

    y_pred = model.predict(X_eval)
    # Some models may not support predict_proba; guard
    try:
        y_prob = model.predict_proba(X_eval)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float('nan')

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Calculate prediction statistics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    total_samples = len(y_test)
    predicted_positive = (y_pred == 1).sum()
    predicted_negative = (y_pred == 0).sum()

    # Create prediction statistics table
    st.subheader("Prediction Statistics")
    stats_data = {
        "Metric": [
            "Total Test Samples",
            "Predicted Positive",
            "Predicted Negative",
            "True Positives",
            "True Negatives",
            "False Positives",
            "False Negatives"
        ],
        "Count": [
            total_samples,
            predicted_positive,
            predicted_negative,
            tp,
            tn,
            fp,
            fn
        ]
    }
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns([1,2])
    with col1:
        st.metric("Accuracy", f"{acc:.4f}")
        st.metric("AUC", f"{auc:.4f}" if not pd.isna(auc) else "N/A")
        st.metric("Precision", f"{prec:.4f}")
        st.metric("Recall", f"{rec:.4f}")
        st.metric("F1 Score", f"{f1:.4f}")
        st.metric("MCC", f"{mcc:.4f}")

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    st.subheader("Detailed Classification Report")
    # Get classification report as dictionary
    report_dict = classification_report(y_test, y_pred, output_dict=True, target_names=["No Disease", "Disease"])
    
    # Extract metrics for each class and overall metrics
    report_data = {
        "Metric": ["No Disease", "Disease", "accuracy", "macro avg", "weighted avg"],
        "precision": [
            f"{report_dict['No Disease']['precision']:.6f}",
            f"{report_dict['Disease']['precision']:.6f}",
            f"{report_dict['accuracy']:.6f}",
            f"{report_dict['macro avg']['precision']:.6f}",
            f"{report_dict['weighted avg']['precision']:.6f}"
        ],
        "recall": [
            f"{report_dict['No Disease']['recall']:.6f}",
            f"{report_dict['Disease']['recall']:.6f}",
            f"{report_dict['accuracy']:.6f}",
            f"{report_dict['macro avg']['recall']:.6f}",
            f"{report_dict['weighted avg']['recall']:.6f}"
        ],
        "f1-score": [
            f"{report_dict['No Disease']['f1-score']:.6f}",
            f"{report_dict['Disease']['f1-score']:.6f}",
            f"{report_dict['accuracy']:.6f}",
            f"{report_dict['macro avg']['f1-score']:.6f}",
            f"{report_dict['weighted avg']['f1-score']:.6f}"
        ],
        "support": [
            int(report_dict['No Disease']['support']),
            int(report_dict['Disease']['support']),
            int(report_dict['macro avg']['support']),
            int(report_dict['macro avg']['support']),
            int(report_dict['weighted avg']['support'])
        ]
    }
    
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, use_container_width=True, hide_index=True)
else:
    if not models:
        st.error("No models available. Please ensure all model files are committed to the repository.")
        st.info("**Steps to fix:**\n\n1. Run locally: `python save_models.py` to generate model files\n\n2. Verify files exist in `model/` folder\n\n3. Commit and push to GitHub:\n```bash\ngit add model/\ngit commit -m 'Add trained models'\ngit push origin main\n```\n\n4. Redeploy on Streamlit Cloud")

st.markdown("---")
st.markdown("**Usage:** Run `python save_models.py` to (re)train and save models, then deploy this app to Streamlit Cloud using `app.py`.")

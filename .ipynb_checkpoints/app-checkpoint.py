import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "."          # current folder
DATASET_ID = "FD001"

PROCESSED_TRAIN = os.path.join(DATA_DIR, f"processed_{DATASET_ID}_train.csv")
SCALER_PATH = os.path.join(DATA_DIR, f"scaler_{DATASET_ID}.joblib")
MODEL_PATH = os.path.join(DATA_DIR, f"rf_model_{DATASET_ID}.joblib")


# -----------------------------
# LOAD ASSETS (with caching)
# -----------------------------
@st.cache_data
def load_processed_train():
    df = pd.read_csv(PROCESSED_TRAIN)
    return df

@st.cache_resource
def load_model_and_scaler():
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    return model, scaler


# -----------------------------
# HELPER: get feature columns
# -----------------------------
def get_feature_columns(df):
    # features = all 'os*' and 's*' columns used for training
    feature_cols = [c for c in df.columns if c.startswith("os") or c.startswith("s")]
    return feature_cols


def predict_rul_for_row(model, scaler, feature_cols, feature_values_dict):
    """
    feature_values_dict: {col_name: value, ...} for all feature_cols
    """
    # Ensure order of columns
    x = np.array([[feature_values_dict[c] for c in feature_cols]], dtype=float)
    # Scale using same scaler
    x_scaled = scaler.transform(x)
    # Predict
    y_pred = model.predict(x_scaled)[0]
    return y_pred


# -----------------------------
# STREAMLIT UI
# -----------------------------
def main():
    st.set_page_config(page_title="Turbofan RUL Predictor", layout="wide")

    st.title("🔧 Turbofan Engine RUL Prediction – FD001")
    st.write(
        "This app uses a trained **RandomForest Regressor** on the NASA CMAPSS FD001 dataset "
        "to predict the **Remaining Useful Life (RUL)** of a turbofan engine."
    )

    # Load data + model
    with st.spinner("Loading data and model..."):
        train_df = load_processed_train()
        model, scaler = load_model_and_scaler()
        feature_cols = get_feature_columns(train_df)

    st.sidebar.header("Mode")
    mode = st.sidebar.radio(
        "Choose input mode:",
        ["Use sample from dataset", "Enter custom sensor values"]
    )

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Number of training rows:** {len(train_df)}")
    st.sidebar.write(f"**Number of features:** {len(feature_cols)}")

    # -------------------------
    # MODE 1: SAMPLE FROM DATA
    # -------------------------
    if mode == "Use sample from dataset":
        st.subheader("📊 Predict RUL for a sample row from training data")

        idx = st.slider(
            "Select row index from processed train data",
            min_value=0,
            max_value=len(train_df) - 1,
            value=0,
            step=1
        )

        sample = train_df.iloc[idx]
        st.write("### Selected row (raw values)")
        st.write(sample[["unit", "cycle", "RUL"] + feature_cols].to_frame().T)

        # Build feature dict
        feature_values = {c: float(sample[c]) for c in feature_cols}
        true_rul = float(sample["RUL"])

        if st.button("Predict RUL for this row"):
            pred_rul = predict_rul_for_row(model, scaler, feature_cols, feature_values)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🔮 Predicted RUL (cycles)", f"{pred_rul:.2f}")
            with col2:
                st.metric("✅ True RUL (from dataset)", f"{true_rul:.2f}")

    # -------------------------
    # MODE 2: CUSTOM INPUT
    # -------------------------
    else:
        st.subheader("🧪 Predict RUL for custom sensor values")

        st.write(
            "Enter values for the operating settings and sensor measurements.\n\n"
            "Tip: Start with values close to dataset statistics (min/max) "
            "or copy values from a real row to understand behavior."
        )

        # Show basic stats so user has a sense of ranges
        with st.expander("Show basic statistics for each feature"):
            st.write(train_df[feature_cols].describe().T)

        st.markdown("### Enter feature values")

        # To keep UI manageable, we split into 3 columns
        cols_ui = st.columns(3)
        feature_values = {}

        for i, col_name in enumerate(feature_cols):
            col_ui = cols_ui[i % 3]
            col_min = float(train_df[col_name].min())
            col_max = float(train_df[col_name].max())
            col_mean = float(train_df[col_name].mean())

            # Using number_input with default = mean, min/max for guidance
            with col_ui:
                feature_values[col_name] = st.number_input(
                    col_name,
                    value=col_mean,
                    help=f"Range approx: [{col_min:.2f}, {col_max:.2f}]"
                )

        if st.button("🔮 Predict RUL"):
            pred_rul = predict_rul_for_row(model, scaler, feature_cols, feature_values)
            st.success(f"Predicted Remaining Useful Life (RUL): **{pred_rul:.2f} cycles**")


    # -------------------------
    # Sidebar – Info
    # -------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.write(
        "- Dataset: NASA CMAPSS FD001\n"
        "- Model: RandomForestRegressor\n"
        "- Features: 3 operating settings + selected sensors\n"
        "\nYou can mention this app in your weekly report as a deployment/demo step."
    )


if __name__ == "__main__":
    main()

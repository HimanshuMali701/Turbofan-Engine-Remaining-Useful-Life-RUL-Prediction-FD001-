import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import requests  # ✅ NEW

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "."          # current folder
DATASET_ID = "FD001"

PROCESSED_TRAIN = os.path.join(DATA_DIR, f"processed_{DATASET_ID}_train.csv")
SCALER_PATH = os.path.join(DATA_DIR, f"scaler_{DATASET_ID}.joblib")

# Name of model file on disk (after download)
MODEL_PATH = os.path.join(DATA_DIR, f"rf_model_{DATASET_ID}.joblib")

MODEL_URL = "https://huggingface.co/HimanshuMali/turbofan-rul-rf-model/blob/main/rf_model_FD001.joblib"


# -----------------------------
# HELPERS FOR DOWNLOADING MODEL
# -----------------------------
def download_file_if_missing(local_path: str, url: str):
    """
    Download file from `url` to `local_path` if it does not exist.
    """
    if os.path.exists(local_path):
        return

    # Show progress in Streamlit
    with st.spinner("Downloading model file... this is done only once."):
        resp = requests.get(url)
        resp.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(resp.content)


# -----------------------------
# LOAD ASSETS (with caching)
# -----------------------------
@st.cache_data
def load_processed_train():
    df = pd.read_csv(PROCESSED_TRAIN)
    return df

@st.cache_resource
def load_model_and_scaler():
    # Make sure model file exists (download if needed)
    download_file_if_missing(MODEL_PATH, MODEL_URL)

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
    st.set_page_config(
        page_title="Turbofan RUL Predictor",
        page_icon="🔧",
        layout="wide"
    )

    # --------- HEADER -----------
    st.markdown(
        "<h2 style='text-align: center;'>Turbofan Engine RUL Prediction — FD001</h2>",
        unsafe_allow_html=True
    )
    st.write(
        "This dashboard uses a trained **RandomForest Regressor** on the "
        "NASA CMAPSS FD001 dataset to estimate the **Remaining Useful Life (RUL)** "
        "of a turbofan engine."
    )

    # --------- LOAD DATA & MODEL -----------
    with st.spinner("Loading data and model..."):
        train_df = load_processed_train()
        model, scaler = load_model_and_scaler()
        feature_cols = get_feature_columns(train_df)

    # Top summary cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Training rows", f"{len(train_df):,}")
    with c2:
        st.metric("Features used", len(feature_cols))
    with c3:
        st.metric("Engines (units)", train_df["unit"].nunique())

    st.markdown("---")

    # ------------ TABS ------------
    tab_sample, tab_custom, tab_importance, tab_about = st.tabs([
        "🔎 Sample from dataset",
        "🧪 Custom input",
        "📊 Feature importance",
        "ℹ️ About model"
    ])

    # ============ TAB 1: SAMPLE ============
    with tab_sample:
        st.subheader("Predict RUL for a sample row from training data")

        idx = st.slider(
            "Select row index",
            min_value=0,
            max_value=len(train_df) - 1,
            value=0,
            step=1
        )

        sample = train_df.iloc[idx]
        st.write("#### Selected row (raw values)")
        st.dataframe(sample[["unit", "cycle", "RUL"] + feature_cols].to_frame().T)

        feature_values = {c: float(sample[c]) for c in feature_cols}
        true_rul = float(sample["RUL"])

        if st.button("🔮 Predict RUL for this row"):
            pred_rul = predict_rul_for_row(model, scaler, feature_cols, feature_values)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Predicted RUL (cycles)", f"{pred_rul:.2f}")
            with c2:
                st.metric("True RUL (dataset)", f"{true_rul:.2f}")

    # ============ TAB 2: CUSTOM ============
    with tab_custom:
        st.subheader("Enter custom operating conditions & sensor values")

        st.write(
            "Use this section to experiment with hypothetical sensor readings and "
            "see how the predicted RUL changes."
        )

        with st.expander("Show basic statistics for each feature"):
            st.dataframe(train_df[feature_cols].describe().T)

        cols_ui = st.columns(3)
        feature_values = {}

        for i, col_name in enumerate(feature_cols):
            col_ui = cols_ui[i % 3]
            col_min = float(train_df[col_name].min())
            col_max = float(train_df[col_name].max())
            col_mean = float(train_df[col_name].mean())

            with col_ui:
                feature_values[col_name] = st.number_input(
                    col_name,
                    value=col_mean,
                    help=f"Range approx: [{col_min:.2f}, {col_max:.2f}]"
                )

        if st.button("🔮 Predict RUL for custom input"):
            pred_rul = predict_rul_for_row(model, scaler, feature_cols, feature_values)
            st.success(f"Estimated Remaining Useful Life: **{pred_rul:.2f} cycles**")

    # ============ TAB 3: IMPORTANCE ============
    with tab_importance:
        st.subheader("Feature importance")

        feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

        st.write(
            "The plot below shows the top features that contribute most to the "
            "RandomForest model's RUL predictions."
        )

        fig, ax = plt.subplots(figsize=(4, 2.5), dpi=80)  # small & compact

        top_feats = feat_imp.head(8)
        sns.barplot(x=top_feats.values, y=top_feats.index, ax=ax)

        ax.set_title("")
        ax.set_xlabel("Importance", fontsize=8)
        ax.set_ylabel("Feature", fontsize=8)
        ax.tick_params(axis='both', labelsize=8)

        plt.tight_layout()

        _, col, _ = st.columns([1, 3, 1])
        with col:
            st.pyplot(fig)

        st.write("### Top features (numeric values)")
        st.dataframe(top_feats.round(4).to_frame("importance"))

    # ============ TAB 4: ABOUT ============
    with tab_about:
        st.subheader("About this model & dataset")
        st.markdown(
            """
            - **Dataset:** NASA CMAPSS FD001 (single operating condition, single fault mode)  
            - **Target:** Remaining Useful Life (RUL) in cycles  
            - **Model:** `RandomForestRegressor(n_estimators=200)`  
            - **Preprocessing:**  
              - Removed low-variance sensors: `s1, s5, s10, s16, s18, s19`  
              - Min-Max scaling on operating settings & remaining sensors  
            """
        )


if __name__ == "__main__":
    main()

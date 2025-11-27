# Turbofan Engine Remaining Useful Life (RUL) Prediction — FD001

This project implements a **Predictive Maintenance** pipeline for turbofan engines using the **NASA CMAPSS FD001** dataset.  
The objective is to predict the **Remaining Useful Life (RUL)**, in cycles, before engine failure and to expose the model via an interactive **Streamlit dashboard**.

---

## 🔧 Project Overview

- **Domain:** Predictive Maintenance – Remaining Useful Life (RUL) Prediction  
- **Dataset:** NASA CMAPSS – FD001 subset (single operating condition, single fault mode)  
- **Target:** Remaining Useful Life (RUL) of each engine at each time step  
- **Model:** RandomForestRegressor (scikit-learn)  
- **UI / Deployment:** Streamlit app (local + cloud / Hugging Face Spaces)

---

## 🗂 Repository Structure

```text
turbofan-rul-pdm/
├─ app.py                    
├─ Turbofan engine_CMAPSS_RUL_FD001.ipynb     
├─ data/
│  └─ RUL_FD001.txt
│  └─ test_FD001.txt
│  └─ train_FD001.txt
│  └─ processed/
│       └─ processed_FD001_train.csv
│       └─ rf_model_FD001.joblib
│       └─ scaler_FD001.joblib        
├─ requirements.txt
├─ .gitignore
└─ README.md

📊 Methodology

Data Loading

Loaded train_FD001.txt, test_FD001.txt, and corresponding RUL_FD001.txt.

Assigned meaningful column names: unit, cycle, os1–os3, s1–s21.

RUL Label Generation

For each engine (unit), computed

RUL
=
max
⁡
(
cycle
unit
)
−
cycle
RUL=max(cycle
unit
	​

)−cycle

For test set, reconstructed per-cycle RUL using final RUL_FD001 values.

Exploratory Data Analysis

Visualized RUL distribution and sensor statistics.

Identified low-variance / almost constant sensors.

Feature Engineering

Removed low-variance sensors:
s1, s5, s10, s16, s18, s19.

Applied MinMax scaling to operating settings and remaining sensors.

Model Training

Model: RandomForestRegressor(n_estimators=200, random_state=42).

Train/validation split: 80% / 20%.

Evaluation

Metrics used: MAE, RMSE, R².

Evaluated on both validation and test sets (per-cycle RUL).

Deployment

Saved artifacts: rf_model_FD001.joblib, scaler_FD001.joblib.

Built a Streamlit dashboard for:

Selecting sample rows from dataset

Providing custom sensor inputs

Visualizing feature importance

✅ Results

Validation:
- MAE   : 29.566
- RMSE  : 41.386
- R²    : 0.625

Test:
- MAE   : 34.998
- RMSE  : 46.391
- R²    : 0.381



Feature importance analysis showed that sensors such as s11, s4, s9, s12 and s7 contribute most to the RUL prediction.
```
⚠ Dataset is not included due to licensing and size. 
Download CMAPSS FD001 from NASA PCoE:
https://drive.google.com/file/d/1zfqvs8-mAO6E0JpgvhBdueNx8Th03pUp/view?usp=sharing 
and place files in /data before running the notebook.






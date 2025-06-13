import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

# ---------------- SETUP ---------------- #
data_dir = "data"
plot_dir = "plots"

st.set_page_config(page_title="Volatility Forecast Dashboard", layout="wide")
st.title("📊 Volatility Forecasting Dashboard")
st.sidebar.header("Navigation")

# -------- Sidebar Model Selection -------- #
model_options = [
    "GARCH",
    "LSTM",
    "LSTM-GARCH",
    "LSTM-GARCH-VIX",
    "EWMA",
    "VIX Overlay",
    "All Models Comparison"
]
selected_models = st.sidebar.multiselect("Choose models to display:", model_options, default=model_options[:3])

# ----------------- HELPERS ----------------- #
def load_csv(file):
    path = os.path.join(data_dir, file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'Date' not in df.columns:
            for col in ['timestamp', 'test_date']:
                if col in df.columns:
                    df['Date'] = pd.to_datetime(df[col])
                    break
        else:
            df['Date'] = pd.to_datetime(df['Date'])
        return df.dropna()
    else:
        st.warning(f"File not found: {file}")
        return pd.DataFrame()

def plot_model(df, title, pred_col="prediction"):
    if not df.empty:
        st.subheader(title)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Date'], df[pred_col], label="Predicted", color="blue")
        if "actual" in df.columns:
            ax.plot(df['Date'], df['actual'], label="Actual", color="red", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")
        ax.legend()
        st.pyplot(fig)

        mae = mean_absolute_error(df['actual'], df[pred_col]) if "actual" in df.columns else np.nan
        rmse = np.sqrt(mean_squared_error(df['actual'], df[pred_col])) if "actual" in df.columns else np.nan
        st.markdown(f"**MAE:** {mae:.6f} | **RMSE:** {rmse:.6f}")

# ----------------- TABS ----------------- #
tabs = st.tabs(["📉 GARCH", "🤖 LSTM", "📘 LSTM-GARCH", "📘 LSTM-GARCH-VIX", "🔁 EWMA", "📈 VIX", "📊 Comparison"])

# GARCH Tab
with tabs[0]:
    if "GARCH" in selected_models:
        garch_df = load_csv("results_garch_intraday.csv")
        plot_model(garch_df, "GARCH Predictions vs. Actual")

# LSTM Tab
with tabs[1]:
    if "LSTM" in selected_models:
        lstm_df = load_csv("results_lstm_intraday.csv")
        plot_model(lstm_df, "LSTM Predictions vs. Actual")

# LSTM-GARCH Tab
with tabs[2]:
    if "LSTM-GARCH" in selected_models:
        lstm_garch_df = load_csv("results_lstm_garch_intraday.csv")
        plot_model(lstm_garch_df, "LSTM-GARCH Predictions vs. Actual")

# LSTM-GARCH-VIX Tab
with tabs[3]:
    if "LSTM-GARCH-VIX" in selected_models:
        lstm_garch_vix_df = load_csv("results_lstm_garch_vix_intraday.csv")
        plot_model(lstm_garch_vix_df, "LSTM-GARCH-VIX Predictions vs. Actual")

# EWMA Tab
with tabs[4]:
    if "EWMA" in selected_models:
        ewma_df = load_csv("results_ewma.csv")
        if not ewma_df.empty and {"log_returns", "ewma_volatility"}.issubset(ewma_df.columns):
            ewma_df["rolling_std"] = ewma_df["log_returns"].rolling(26).std()
            plot_df = ewma_df.dropna()
            st.subheader("EWMA vs Rolling Std Dev (26 periods)")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(plot_df["Date"], plot_df["ewma_volatility"], label="EWMA Volatility", color="blue")
            ax.plot(plot_df["Date"], plot_df["rolling_std"], label="Rolling Std Dev", color="gray", linestyle="--")
            ax.set_title("EWMA vs Rolling Std Dev")
            ax.set_xlabel("Date")
            ax.set_ylabel("Volatility")
            ax.legend()
            st.pyplot(fig)

# VIX Tab
with tabs[5]:
    if "VIX Overlay" in selected_models:
        vix_df = load_csv("vix_15min.csv")
        ewma_df = load_csv("results_ewma.csv")
        if not vix_df.empty and not ewma_df.empty:
            merged = pd.merge(ewma_df, vix_df, on="Date", how="inner")
            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(merged['Date'], merged['ewma_volatility'], color='blue', label='EWMA')
            ax2 = ax1.twinx()
            ax2.plot(merged['Date'], merged['close'], color='green', label='VIX Close', alpha=0.5)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('EWMA Volatility')
            ax2.set_ylabel('VIX Close')
            fig.legend(loc="upper right")
            st.pyplot(fig)

# Comparison Tab
with tabs[6]:
    st.markdown("### Model Comparison Summary")
    model_metrics = []
    files = {
        "GARCH": "results_garch_intraday.csv",
        "LSTM": "results_lstm_intraday.csv",
        "LSTM-GARCH": "results_lstm_garch_intraday.csv",
        "LSTM-GARCH-VIX": "results_lstm_garch_vix_intraday.csv",
        "EWMA": "results_ewma.csv",
    }
    for model, filename in files.items():
        df = load_csv(filename)
        if model == "EWMA":
            if {"ewma_volatility", "log_returns"}.issubset(df.columns):
                actual = df["log_returns"].rolling(26).std().dropna()
                predicted = df["ewma_volatility"][25:]
        else:
            if not {"actual", "prediction"}.issubset(df.columns):
                continue
            actual = df["actual"]
            predicted = df["prediction"]
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        model_metrics.append({"Model": model, "MAE": mae, "RMSE": rmse})

    if model_metrics:
        st.dataframe(pd.DataFrame(model_metrics))
    else:
        st.info("No models available for comparison.")

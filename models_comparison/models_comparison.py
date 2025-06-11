import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Add project root to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ------------------ Define Directories ------------------ #
data_dir = os.path.join("data")
os.makedirs(data_dir, exist_ok=True)

# ------------------ Load Results ------------------ #
# Load GARCH
garch_path = os.path.join(data_dir, "results_garch_intraday.csv")
garch = pd.read_csv(garch_path)
garch = garch.loc[garch["Date"] >= "2015-02-13", :]
print(
    f"DataFrame: {'garch'} | MAE: {mean_absolute_error(garch.actual, garch.prediction)} | RMSE: {np.sqrt(mean_squared_error(garch.actual, garch.prediction))}"
)

# Load LSTM-related results
lstm_files = {
    "lstm": "results_lstm_intraday.csv",
    "lstm_garch": "results_lstm_garch_intraday.csv",
    "lstm_garch_vix": "results_lstm_garch_vix_intraday.csv",
    "lstm_garch_vix_pct_change": "results_lstm_garch_vix_pct_change.csv",
    "lstm_garch_vix_1_layer": "results_lstm_garch_vix_1_layer.csv",
    "lstm_garch_vix_3_layers": "results_lstm_garch_vix_3_layer_intraday.csv",
    "lstm_garch_vix_lookback_5": "results_lstm_garch_vix_lookback_5.csv",
    "lstm_garch_vix_lookback_66": "results_lstm_garch_vix_lookback_66.csv",
    "lstm_garch_vix_mae_loss": "results_lstm_garch_vix_mae_loss_intraday.csv",
    "lstm_garch_vix_relu": "results_lstm_garch_vix_relu.csv",
}

lstm_dfs = {}
for key, file in lstm_files.items():
    file_path = os.path.join(data_dir, file)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.dropna(axis=0, inplace=True)
        lstm_dfs[key] = df

        mae = mean_absolute_error(df.actual, df.prediction)
        rmse = np.sqrt(mean_squared_error(df.actual, df.prediction))
        print(f"DataFrame: {key} | MAE: {mae} | RMSE: {rmse}")
    else:
        print(f"Warning: File not found - {file_path}")


# ------------------ Plotting Function ------------------ #
def plot_data(dataframes, labels, colors, linestyles, title, x_label, y_label):
    plt.figure(figsize=(12, 6))
    for df, label, color, linestyle in zip(dataframes, labels, colors, linestyles):
        plt.plot(
            pd.to_datetime(df["Date"]),
            df[df.columns[1]],
            label=label,
            color=color,
            linestyle=linestyle,
        )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------ Plotting ------------------ #
# GARCH
plot_data(
    dataframes=[garch[["Date", "prediction"]], garch[["Date", "actual"]]],
    labels=["Predicted Volatility", "Actual Values"],
    colors=["royalblue", "red"],
    linestyles=["-", "--"],
    title="GARCH Model Rolling Window Predictions vs. Actual Values",
    x_label="Date",
    y_label="Values",
)

# LSTM
if "lstm" in lstm_dfs:
    plot_data(
        dataframes=[
            lstm_dfs["lstm"][["Date", "prediction"]],
            lstm_dfs["lstm"][["Date", "actual"]],
        ],
        labels=["LSTM", "Actual Values"],
        colors=["gold", "red"],
        linestyles=["-", "--"],
        title="LSTM vs. Actual Values",
        x_label="Date",
        y_label="Values",
    )

# LSTM-GARCH
if "lstm_garch" in lstm_dfs:
    plot_data(
        dataframes=[
            lstm_dfs["lstm_garch"][["Date", "prediction"]],
            lstm_dfs["lstm_garch"][["Date", "actual"]],
        ],
        labels=["LSTM-GARCH", "Actual Values"],
        colors=["deepskyblue", "red"],
        linestyles=["-", "--"],
        title="LSTM-GARCH vs. Actual Values",
        x_label="Date",
        y_label="Values",
    )

# LSTM-GARCH-VIX
if "lstm_garch_vix" in lstm_dfs:
    plot_data(
        dataframes=[
            lstm_dfs["lstm_garch_vix"][["Date", "prediction"]],
            lstm_dfs["lstm_garch_vix"][["Date", "actual"]],
        ],
        labels=["LSTM-GARCH with VIX Input", "Actual Values"],
        colors=["springgreen", "red"],
        linestyles=["-", "--"],
        title="LSTM-GARCH with VIX Input vs. Actual Values",
        x_label="Date",
        y_label="Values",
    )

# Sensitivity Comparisons
if "lstm_garch_vix_mae_loss" in lstm_dfs and "lstm_garch_vix" in lstm_dfs:
    plot_data(
        dataframes=[
            lstm_dfs["lstm_garch_vix_mae_loss"][["Date", "prediction"]],
            lstm_dfs["lstm_garch_vix"][["Date", "prediction"]],
            lstm_dfs["lstm_garch_vix"][["Date", "actual"]],
        ],
        labels=["MAE Loss", "MSE Loss", "Actual Values"],
        colors=["purple", "springgreen", "red"],
        linestyles=["-", "-", "--"],
        title="LSTM-GARCH-VIX (MAE vs. MSE) vs. Actual Values",
        x_label="Date",
        y_label="Values",
    )

if "lstm_garch_vix" in lstm_dfs and "lstm_garch_vix_pct_change" in lstm_dfs:
    plot_data(
        dataframes=[
            lstm_dfs["lstm_garch_vix"][["Date", "prediction"]],
            lstm_dfs["lstm_garch_vix_pct_change"][["Date", "prediction"]],
            lstm_dfs["lstm_garch_vix"][["Date", "actual"]],
        ],
        labels=["Log Returns Input", "Pct Change Input", "Actual Values"],
        colors=["springgreen", "gold", "red"],
        linestyles=["-", "-", "--"],
        title="LSTM-GARCH-VIX (Log Returns vs. Pct Change) vs. Actual Values",
        x_label="Date",
        y_label="Values",
    )

if all(
    k in lstm_dfs
    for k in [
        "lstm_garch_vix",
        "lstm_garch_vix_lookback_5",
        "lstm_garch_vix_lookback_66",
    ]
):
    plot_data(
        dataframes=[
            lstm_dfs["lstm_garch_vix_lookback_66"][["Date", "prediction"]],
            lstm_dfs["lstm_garch_vix"][["Date", "prediction"]],
            lstm_dfs["lstm_garch_vix_lookback_5"][["Date", "prediction"]],
            lstm_dfs["lstm_garch_vix"][["Date", "actual"]],
        ],
        labels=["Sequence of 66", "Sequence of 22", "Sequence of 5", "Actual Values"],
        colors=["purple", "springgreen", "gold", "red"],
        linestyles=["-", "-", "-", "--"],
        title="LSTM-GARCH-VIX (Lookbacks) vs. Actual Values",
        x_label="Date",
        y_label="Values",
    )

if all(
    k in lstm_dfs
    for k in ["lstm_garch_vix_3_layers", "lstm_garch_vix", "lstm_garch_vix_1_layer"]
):
    plot_data(
        dataframes=[
            lstm_dfs["lstm_garch_vix_3_layers"][["Date", "prediction"]],
            lstm_dfs["lstm_garch_vix"][["Date", "prediction"]],
            lstm_dfs["lstm_garch_vix_1_layer"][["Date", "prediction"]],
            lstm_dfs["lstm_garch_vix"][["Date", "actual"]],
        ],
        labels=["3 LSTM layers", "2 LSTM layers", "1 LSTM layer", "Actual Values"],
        colors=["purple", "springgreen", "gold", "red"],
        linestyles=["-", "-", "-", "--"],
        title="LSTM-GARCH-VIX (1 vs. 2 vs. 3 LSTM Layers) vs. Actual Values",
        x_label="Date",
        y_label="Values",
    )

if "lstm_garch_vix_relu" in lstm_dfs and "lstm_garch_vix" in lstm_dfs:
    plot_data(
        dataframes=[
            lstm_dfs["lstm_garch_vix_relu"][["Date", "prediction"]],
            lstm_dfs["lstm_garch_vix"][["Date", "prediction"]],
            lstm_dfs["lstm_garch_vix"][["Date", "actual"]],
        ],
        labels=["ReLU Activation", "Tanh Activation", "Actual Values"],
        colors=["gold", "springgreen", "red"],
        linestyles=["-", "-", "--"],
        title="LSTM-GARCH-VIX (ReLU vs. Tanh) vs. Actual Values",
        x_label="Date",
        y_label="Values",
    )

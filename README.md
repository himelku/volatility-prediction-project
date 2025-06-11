# 📊 Volatility Prediction Project

This project implements **intraday volatility forecasting** using **LSTM-GARCH** and **VIX** models on **15-minute SPY data**. The pipeline combines time-series preprocessing, GARCH volatility estimation, and deep learning (LSTM) architectures, exploring sensitivity analysis on lookback windows, loss functions, and model architectures.

## 🚀 Features

- **GARCH model** for initial volatility estimation
- **LSTM model** for capturing non-linear patterns
- Integration of **VIX** data for enhanced market context
- Sensitivity tests: lookback windows, loss functions (MAE vs MSE), LSTM layers, activation functions
- Evaluation metrics: MAE, RMSE, time-series plots

## 📁 Project Structure

```
volatility_prediction_project/
│
├── data/
│   ├── SPY_15min_intraday.csv
│   ├── SPY_15min_lstm.csv
│   ├── vix_15min.csv
│   ├── results_garch_intraday.csv
│   ├── results_lstm_garch_intraday.csv
│   └── ... (all results CSVs)
│
├── garch/
│   ├── garch_functions.py
│
├── lstm/
│   ├── LSTM.py
│
├── sensitivity/
│   ├── sensitivity_model_function.py
│
├── scripts/
│   ├── train_lstm_garch_vix.py
│   ├── evaluate_models.py
│
├── requirements.txt
└── README.md
```

## 🖥️ How to Run in Google Colab

1. Clone the repository:
   ```python
   !git clone https://github.com/yourusername/volatility-prediction-project.git
   %cd volatility-prediction-project
   ```

2. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```

3. Run the training script:
   ```python
   !python scripts/train_lstm_garch_vix.py
   ```

4. Run the evaluation script:
   ```python
   !python scripts/evaluate_models.py
   ```

## 📝 Notes

- All input datasets and results files should be in the `data/` folder.
- Adjust lookback windows, model architectures, and evaluation metrics via the `scripts/` folder.
- For visualization and metrics, refer to the **evaluate_models.py** script.

## 🤝 Contributing

Feel free to open issues, fork, and submit pull requests!

## 📜 License

MIT License (replace with your preferred license)

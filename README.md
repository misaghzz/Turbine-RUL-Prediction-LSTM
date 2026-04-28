# Gas Turbine RUL Prediction using LSTM 🚀

This project predicts the **Remaining Useful Life (RUL)** of gas turbines based on vibration sensors using Deep Learning (LSTM).

## 🛠 Project Overview
- **Algorithm:** LSTM (Long Short-Term Memory)
- **Framework:** TensorFlow / Keras
- **Metric:** Custom MAE in Days

## 📊 The Physics
The degradation model follows an exponential growth after a stable period:
$$Vibration = Constant + Noise + e^{\frac{t - t_{start}}{20}}$$

The RUL is calculated and clipped to provide a stable learning signal for the AI:
$$RUL_{clipped} = \min(RUL_{actual}, 70)$$

## 📈 Results
The model achieves high accuracy with an average error of less than 1 day.

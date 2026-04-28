import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow.keras.backend as K

# Data Generation
day = np.arange(1, 201)
vibration = np.zeros(200)
RUL = 200 - day

for i in range(200):
    current_day = day[i]
    if current_day <= 131:
        vibration[i] = 10 + np.random.normal(0, 0.5)
    else:
        vibration[i] = 10 + np.random.normal(0, 0.5) + np.exp((current_day - 131) / 20)

Data = pd.DataFrame({'Vibration': vibration, 'Day': day, 'Remaining Useful Life': RUL})

# Preprocessing
scaler = MinMaxScaler()
vibration_scaled = scaler.fit_transform(Data['Vibration'].values.reshape(-1, 1))

def create_sequences(data, rul_data, look_back):
    x_seq, y_seq = [], []
    for i in range(len(data) - look_back):
        x_seq.append(data[i:(i + look_back)])
        y_seq.append(rul_data[i + look_back])
    return np.array(x_seq), np.array(y_seq)

look_back = 10

# Custom Metric Fix: Reshape y_true to match y_pred dimensions
def mae_in_days(y_true, y_pred):
    y_true = K.reshape(y_true, (-1, 1))
    return K.mean(K.abs(y_pred - y_true)) * 70

Data['Remaining Useful Life'] = Data['Remaining Useful Life'].clip(upper=70) / 70
X_deep, Y_deep = create_sequences(vibration_scaled, Data['Remaining Useful Life'].values, look_back)

# Model Architecture
model = Sequential()
model.add(LSTM(units=50, input_shape=(look_back, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=[mae_in_days])

print("\n--- Model Architecture ---")
model.summary()

# Training
print('\nStarting Training Process')
history = model.fit(X_deep, Y_deep, epochs=100, batch_size=8, verbose=1)

# Plotting Training Error in Days
plt.figure(figsize=(10, 5))
plt.plot(history.history['mae_in_days'], color='green', linewidth=2, label='MAE in Days')
plt.title('Training Progress: Error in Days')
plt.xlabel('Epochs')
plt.ylabel('Error (Days)')
plt.grid(True)
plt.legend()
plt.show()

# Final Prediction and Visualization
predictions_days = model.predict(X_deep) * 70
actual_days = Y_deep * 70

plt.figure(figsize=(12, 6))
plt.plot(actual_days, color='blue', label='Actual RUL (Reality)', linewidth=2)
plt.plot(predictions_days, color='red', linestyle='--', label='Predicted RUL (AI)', linewidth=2)
plt.title('Final Validation: AI Prediction vs Reality')
plt.xlabel('Samples')
plt.ylabel('Remaining Useful Life (Days)')
plt.legend()
plt.grid(True)
plt.show()
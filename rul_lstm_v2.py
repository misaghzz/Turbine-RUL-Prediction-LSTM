import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K
import keras_tuner as kt
import tensorflow as tf

# v2: reproducible runs + separate validation machine (no test-set leakage)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def generate_machine_data(total_days, fault_start_day, noise_level, trend_factor):
    day = np.arange(1, total_days + 1)
    vibration = np.zeros(total_days)
    RUL = total_days - day
    for i in range(total_days):
        if day[i] <= fault_start_day:
            vibration[i] = 10 + np.random.normal(0, noise_level)
        else:
            vibration[i] = 10 + np.random.normal(0, noise_level) + \
                           np.exp((day[i] - fault_start_day) / trend_factor)
    return pd.DataFrame({'Vibration': vibration, 'RUL': RUL})

train_raw = generate_machine_data(200, 120, 0.4, 20)
val_raw = generate_machine_data(230, 140, 0.6, 22)   # used ONLY for tuning / early stopping
test_raw = generate_machine_data(250, 160, 0.8, 25)  # untouched until the final evaluation

scaler = MinMaxScaler()
train_vibration_scaled = scaler.fit_transform(train_raw['Vibration'].values.reshape(-1, 1))
val_vibration_scaled = scaler.transform(val_raw['Vibration'].values.reshape(-1, 1))
test_vibration_scaled = scaler.transform(test_raw['Vibration'].values.reshape(-1, 1))

def create_sequences(data, rul_data, look_back):
    x_seq, y_seq = [], []
    for i in range(len(data) - look_back):
        x_seq.append(data[i:(i + look_back)])
        y_seq.append(rul_data[i + look_back])
    return np.array(x_seq), np.array(y_seq)

look_back = 10
max_rul = 70

Y_train_raw = train_raw['RUL'].clip(upper=max_rul) / max_rul
Y_val_raw = val_raw['RUL'].clip(upper=max_rul) / max_rul
Y_test_raw = test_raw['RUL'].clip(upper=max_rul) / max_rul

X_train, Y_train = create_sequences(train_vibration_scaled, Y_train_raw.values, look_back)
X_val, Y_val = create_sequences(val_vibration_scaled, Y_val_raw.values, look_back)
X_test, Y_test = create_sequences(test_vibration_scaled, Y_test_raw.values, look_back)

Y_train = Y_train.reshape(-1, 1)
Y_val = Y_val.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

def mae_in_days(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true)) * max_rul

def build_model(hp):
    model = Sequential()
    
    hp_lstm_units = hp.Int('lstm_units', min_value=15, max_value=50, step=5)
    model.add(LSTM(units=hp_lstm_units, input_shape=(look_back, 1), activation='tanh'))
    
    hp_dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    model.add(Dense(units=hp_dense_units, activation='relu'))
    
    hp_dropout = hp.Choice('dropout_rate', values=[0.1, 0.2])
    model.add(Dropout(rate=hp_dropout))
    
    model.add(Dense(1))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), 
                  loss='mse', 
                  metrics=[mae_in_days])
    return model

tuner = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective("val_mae_in_days", direction="min"),
    max_trials=15,
    executions_per_trial=1,
    directory='tuning_dir',
    project_name='rul_optimization',
    overwrite=True,
    seed=SEED
)

tuner.search(X_train, Y_train, epochs=40, validation_data=(X_val, Y_val), verbose=0)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best LSTM Units: {best_hps.get('lstm_units')}")
print(f"Best Dense Units: {best_hps.get('dense_units')}")
print(f"Best Dropout Rate: {best_hps.get('dropout_rate')}")
print(f"Best Learning Rate: {best_hps.get('learning_rate')}")

final_model = tuner.hypermodel.build(best_hps)
early = tf.keras.callbacks.EarlyStopping(monitor='val_mae_in_days', mode='min', patience=10, restore_best_weights=True)
history = final_model.fit(X_train, Y_train, epochs=80, validation_data=(X_val, Y_val), callbacks=[early], verbose=0)

predictions = final_model.predict(X_test) * max_rul
actuals = Y_test * max_rul

# honest evaluation on the untouched test machine
abs_err = np.abs(predictions - actuals).ravel()
degrading = actuals.ravel() < max_rul
print(f"Test MAE (all samples): {abs_err.mean():.2f} days")
print(f"Test MAE (degradation phase, RUL < {max_rul}): {abs_err[degrading].mean():.2f} days")

# generalisation check: a machine that degrades faster than anything seen in training
other_raw = generate_machine_data(220, 110, 0.6, 15)
X_other, Y_other = create_sequences(
    scaler.transform(other_raw['Vibration'].values.reshape(-1, 1)),
    (other_raw['RUL'].clip(upper=max_rul) / max_rul).values, look_back)
pred_other = final_model.predict(X_other.reshape(-1, look_back, 1), verbose=0) * max_rul
act_other = Y_other.reshape(-1, 1) * max_rul
err_other = np.abs(pred_other - act_other).ravel()
print(f"Unseen degradation regime (trend_factor 15 vs 20): MAE = {err_other.mean():.2f} days")

plt.figure(figsize=(12, 6))
plt.plot(actuals, color='blue', label='Actual (Test Machine)')
plt.plot(predictions, color='red', linestyle='--', label='AI Prediction (Optimized Model)')
plt.title('Final Validation of the Best Found Architecture')
plt.xlabel('Time (Samples)')
plt.ylabel('Remaining Useful Life (Days)')
plt.legend()
plt.grid(True)
plt.savefig('rul_prediction_v2.png', dpi=150)
plt.show()
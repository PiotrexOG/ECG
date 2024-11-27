import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
)
from EcgData import EcgData
from config import *
from Finders.PanTompkinsFinder import PanTompkinsFinder
from Finders.UNetFinder import UNetFinder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from EcgPlotter import EcgPlotter
from Finders.CnnFinder import CnnFinder


def create_hrv_model(input_length):
    inputs = Input(shape=(input_length, 1))

    x = Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = LSTM(64, return_sequences=False)(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(2, activation="linear")(x)
    model = Model(inputs, outputs)

    return model


def get_patient_data_mitbih(path, loaded=True):
    data = EcgData(SAMPLING_RATE, finder)
    # EcgPlotter("halo", data)
    data.load_data_from_mitbih(path)
    data.check_detected_peaks()
    # plt.show()
    if loaded:
        X_train, y_train = data.extract_hrv_windows_with_loaded_peaks(input_length)
    else:
        X_train, y_train = data.extract_hrv_windows_with_detected_peaks(input_length)
    return X_train, y_train


def get_patients_data_mithbih(dir_path: str, patients: list):
    X_train = None
    y_train = None
    for patient in patients:
        x, y = get_patient_data_mitbih(dir_path + "\\" + patient, True)
        if X_train is None:
            X_train = x
            y_train = y
        else:
            X_train = np.vstack((X_train, x))
            y_train = np.vstack((y_train, y))

    return X_train, y_train


input_length = 50  # Długość sekwencji interwałów RR
# finder = UNetFinder(f"models/model_{WINDOW_SIZE}_{EPOCHS}_unet.keras", WINDOW_SIZE)
# finder = CnnFinder(f"models/model_{WINDOW_SIZE}_{EPOCHS}_cnn.keras", WINDOW_SIZE)
finder = PanTompkinsFinder()
model = create_hrv_model(input_length)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"]
)

# model.summary()

# data1 = EcgData(SAMPLING_RATE, finder)
# # data.load_csv_data(CSV_PATH)
# data1.load_data_from_mitbih("data\\mit-bih\\108")
# data1.check_detected_peaks()
# EcgPlotter("112", data1)
# plt.show()
# exit()

# val_data = EcgData(SAMPLING_RATE, finder)
# # # data.load_csv_data(CSV_PATH)
# val_data.load_data_from_mitbih("data\\mit-bih\\108")
# val_data.check_detected_peaks()
# EcgPlotter("109", val_data)
# plt.show()
# exit()


X_train, y_train = get_patients_data_mithbih(
    "data\\mit-bih",
    ["100", "101", "102", "103"],
    # ["200", "201", "202"],
)
# X_train, y_train = data.extract_hrv_windows(input_length)
# X_train1, y_train1 = data1.extract_hrv_windows(input_length)
# X_train = np.vstack((X_train, X_train1))
# y_train = np.vstack((y_train, y_train1))
# X_train, y_train = shuffle(X_train, y_train, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.15, random_state=42
# )


X_train, y_train = shuffle(X_train, y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)

model_path = "models\\test.keras"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_path, monitor="loss", verbose=1, save_best_only=True, mode="min"
)
# callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=6)


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=32,
    verbose=1,
    # callbacks=[checkpoint]
)

x, y = get_patient_data_mitbih("data\\mit-bih\\112")
# data = EcgData(SAMPLING_RATE, UNetFinder(f"models/model_{WINDOW_SIZE}_{EPOCHS}.keras", WINDOW_SIZE))
# data.load_csv_data(CSV_PATH)
# x, y = data.extract_hrv_windows_with_detected_peaks(input_length)

# model.load_weights(model_path)
result = model.predict(x)

mae = mean_absolute_error(y, result)

mse = mean_squared_error(y, result)

rmse = np.sqrt(mse)

r2 = r2_score(y, result)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.2f}")

plt.scatter(y[:, 0], result[:, 0], alpha=0.5)
plt.plot([min(y[:, 0]), max(y[:, 0])], [min(y[:, 0]), max(y[:, 0])], "r--")
plt.xlabel("Rzeczywiste SDNN")
plt.ylabel("Przewidywane SDNN")
plt.title("Porównanie rzeczywistych i przewidywanych wartości SDNN")
plt.show()

plt.plot(y[:, 0], label="Rzeczywiste SDNN")
plt.plot(result[:, 0], label="Przewidywane SDNN", linestyle="--")
plt.legend()
plt.title("Przebieg czasowy SDNN")
plt.show()

plt.scatter(y[:, 1], result[:, 1], alpha=0.5)
plt.plot([min(y[:, 1]), max(y[:, 1])], [min(y[:, 1]), max(y[:, 1])], "r--")
plt.xlabel("Rzeczywiste RMSSD")
plt.ylabel("Przewidywane RMSSD")
plt.title("Porównanie rzeczywistych i przewidywanych wartości RMSSD")
plt.show()

plt.plot(y[:, 1], label="Rzeczywiste RMSSD")
plt.plot(result[:, 1], label="Przewidywane RMSSD", linestyle="--")
plt.legend()
plt.title("Przebieg czasowy RMSSD")
plt.show()

pass

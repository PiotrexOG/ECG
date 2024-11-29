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
import os


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

    outputs = Dense(len(METRICS), activation="linear")(x)
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


def get_patient_data_csv(path):
    data = EcgData(SAMPLING_RATE, finder)
    data.load_csv_data(path)
    X_train, y_train = data.extract_hrv_windows_with_detected_peaks(
        input_length, METRICS
    )
    return X_train, y_train


def get_patients_data_csv(dir_path: str, patients: list):
    X_train = None
    y_train = None
    for patient in patients:
        x, y = get_patient_data_csv(dir_path + "\\" + patient)
        if X_train is None:
            X_train = x
            y_train = y
        else:
            X_train = np.vstack((X_train, x))
            y_train = np.vstack((y_train, y))

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


def create_scatter_plot(axs, metric: str, real, pred):
    axs.scatter(real, pred, alpha=0.5)
    axs.plot([min(real), max(real)], [min(real), max(real)], "r--")
    axs.set_xlabel(f"Rzeczywiste {metric}")
    axs.set_ylabel(f"Przewidywane {metric}")
    axs.set_title(f"Porównanie rzeczywistych i przewidywanych wartości {metric}")


def create_line_plot(axs, metric: str, real, pred):
    axs.plot(real, label=f"Rzeczywiste {metric}")
    axs.plot(pred, label=f"Przewidywane {metric}", linestyle="--")
    axs.set_xlabel("Numer sprawdzanego okna")
    axs.set_ylabel(f"Wartość {metric}")
    axs.legend()
    axs.set_title(f"Przebieg czasowy {metric}")


def create_scatter_line_plots(
    metric: str, real, pred, save_path: str = "plots\\SDNN_RMSSD_LF_HF2"
):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    create_scatter_plot(axs[0], metric, real, pred)
    create_line_plot(axs[1], metric, real, pred)

    mae = mean_absolute_error(real, pred)
    mse = mean_squared_error(real, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(real, pred)
    metrics_text = f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}"
    axs[0].text(
        0.05,
        0.95,
        metrics_text,
        transform=axs[0].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    fig.tight_layout()

    if save_path:
        metric = metric.replace("\\", " ")
        metric = metric.replace("/", " ")
        file_path = f"{save_path}\\{metric}.png"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format="png", dpi=300)


if __name__ == "__main__":

    # METRICS = ["SDNN", "RMSSD", "LF", "HF"]
    # METRICS = ["SDNN", "RMSSD", "LF/HF"]
    METRICS = ["SDNN", "RMSSD"]
    input_length = 300  # Długość sekwencji interwałów RR
    finder = UNetFinder(f"models/model_{WINDOW_SIZE}_{EPOCHS}_unet.keras", WINDOW_SIZE)
    # finder = CnnFinder(f"models/model_{WINDOW_SIZE}_{EPOCHS}_cnn.keras", WINDOW_SIZE)
    # finder = PanTompkinsFinder()
    model = create_hrv_model(input_length)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    # model.summary()

    # X_train, y_train = get_patients_data_mithbih(
    #     "data\\mit-bih",
    #     ["100", "101", "102", "103"],
    #     # ["200", "201", "202"],
    # )

    X, y = get_patients_data_csv("data", [csvs[2], csvs[3], csvs[4]])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # model_path = "models\\test.keras"
    model_path = f"models\\{"_".join(METRICS).replace("\\","").replace("/", "")}_{finder.__class__.__name__}.keras"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path, monitor="loss", verbose=1, save_best_only=True, mode="min"
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50)

    history = model.fit(
        X_train,
        y_train,
        # validation_data=(X_val, y_val),
        epochs=500,
        batch_size=32,
        verbose=1,
        shuffle=True,
        callbacks=[checkpoint, callback],
    )

    X_test = X_val
    y_test = y_val

  


    model.load_weights(model_path)
    pred_result = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred_result)

    mse = mean_squared_error(y_test, pred_result)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, pred_result)

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2: {r2:.2f}")

    save_path = "plots\\" + "_".join(METRICS).replace("\\", "").replace("/", "")
    save_path = None

    for metric in METRICS:
        create_scatter_line_plots(
            metric,
            y_test[:, METRICS.index(metric)],
            pred_result[:, METRICS.index(metric)],
            save_path,
        )

    # create_scatter_line_plots(
    #     "LF/HF", y_test[:, METRICS.index("LF")] / y_test[:, METRICS.index("HF")], pred_result[:, METRICS.index("LF")] / pred_result[:, METRICS.index("HF")]
    # )

    plt.show()

    pass




  # X_test, y_test = get_patients_data_csv(
    #     "data",
    #     [
    #         # csvs[11],
    #         csvs[12],
    #         csvs[13],
    #     ]
    # )
    
    # X_test, y_test = get_patients_data_mithbih(
    #     "data\\mit-bih",
    #     [
    #         "117",
    #         "118",
    #         "119",
    #         "121",
    #         "122",
    #         "123",
    #         "124"
    #     ]
    #     # ["100", "101", "102", "103"],
    #     # MITBIH_PATIENTS
    # )
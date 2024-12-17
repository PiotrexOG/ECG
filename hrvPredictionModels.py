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
    Normalization
)

from EcgData import EcgData
from config import *
from Finders.PanTompkinsFinder import PanTompkinsFinder
# from Finders.UNetFinder import UNetFinder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from EcgPlotter import EcgPlotter
# from Finders.CnnFinder import CnnFinder
import os
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Flatten, GlobalAveragePooling1D, MultiHeadAttention, Multiply
from tensorflow.keras.layers import GlobalMaxPooling1D, Concatenate
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from wfdb import processing
from functools import partial

def normalize_bound(data, lb=-1, ub=1):
    min_val = tf.reduce_min(data, axis=-1, keepdims=True)
    max_val = tf.reduce_max(data, axis=-1, keepdims=True)
    return lb + (data - min_val) * (ub - lb) / (max_val - min_val + 1e-7)

class NormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, lb=-1, ub=1, **kwargs):
        super(NormalizeLayer, self).__init__(**kwargs)
        self.lb = lb
        self.ub = ub

    def call(self, inputs):
        return normalize_bound(inputs, lb=self.lb, ub=self.ub)

#PLIK DO PRZEWIDYWANIA MIAR HRV DLA SIECI FEAYTURE BASED Z INTERWALOW RR I MODELU END2END

def create_hrv_model(input_length, output_len):
    input_signal = Input(shape=(input_length, 1))
    x = Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(input_signal)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)  # 1300 → 650

    x = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)  # 650 → 325

    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)  # 325 → 162

    # Warstwa rekurencyjna (LSTM)
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    
    # x = Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(input_signal)
    # x = MaxPooling1D(pool_size=2)(x)
    # x = BatchNormalization()(x)

    # x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
    # x = MaxPooling1D(pool_size=2)(x)
    # x = BatchNormalization()(x)

    # x = LSTM(64, return_sequences=False)(x)

    # x = Dense(128, activation="relu")(x)
    # x = Dropout(0.3)(x)

    # Wyjścia dla SDNN i RMSSD
    output = Dense(output_len, activation='linear')(x)  # Wyjścia: SDNN i RMSSD dla okna
    model = Model(inputs=input_signal, outputs=output)

    return model


# SE Block (Squeeze-and-Excitation Block)
def se_block(input_tensor, ratio=16):
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]
    se = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    return tf.keras.layers.Multiply()([input_tensor, se])


# Model definition
def new_model(input_length, output_len):
    input_signal = Input(shape=(input_length, 1))  # 1300 próbek, 1 kanał

    # CustomNormalization
    # normalized_signal = Normalization()(input_signal)
    # normalized_signal = NormalizeLayer()(input_signal)
    # x = Conv1D(filters=8, kernel_size=11, activation='relu', padding='same')(input_signal)
    # x = BatchNormalization()(x)
    # x = MaxPooling1D(pool_size=10)(x)  # 15000 → 3750

    # x = Conv1D(filters=8, kernel_size=11, activation='relu', padding='same')(input_signal)
    # x = BatchNormalization()(x)
    # x = MaxPooling1D(pool_size=10)(x)  # 15000 → 3750

    x = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(input_signal)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=6)(x)

    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=6)(x)

    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=6)(x)

    # Attention Layer
    x_attention = Dense(128, activation='sigmoid')(x)
    x = Multiply()([x, x_attention])

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Wyjścia: SDNN i RMSSD
    output = Dense(output_len, activation='linear')(x)
    # Model
    model = Model(inputs=input_signal, outputs=output)
    return model


def with_lstm(input_length, output_len):
    input_signal = Input(shape=(input_length, 1))
    x = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(input_signal)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=4)(x)  # 15000 → 3750

    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=4)(x)  # 3750 → 937

    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=4)(x)  # 937 → 234

    # Warstwa rekurencyjna
    x = LSTM(128, return_sequences=False)(x)  # Redukcja do jednej reprezentacji

    # Warstwy Dense
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    # Wyjścia: SDNN i RMSSD
    output = Dense(output_len, activation='linear')(x)

    # Model
    model = Model(inputs=input_signal, outputs=output)
    return model

def Gcreate_scatter_plot(axs, metric: str, real, pred):
    axs.scatter(real, pred, alpha=0.5)
    axs.plot([min(real), max(real)], [min(real), max(real)], "r--")
    axs.set_xlabel(f"Rzeczywiste {metric}")
    axs.set_ylabel(f"Przewidywane {metric}")
    axs.set_title(f"Porównanie rzeczywistych i przewidywanych wartości {metric}")


def Gcreate_line_plot(axs, metric: str, real, pred):
    axs.plot(real, label=f"Rzeczywiste {metric}")
    axs.plot(pred, label=f"Przewidywane {metric}", linestyle="--")
    axs.set_xlabel("Numer sprawdzanego okna")
    axs.set_ylabel(f"Wartość {metric}")
    axs.legend()
    axs.set_title(f"Przebieg czasowy {metric}")


# Główna funkcja do rysowania wykresów
def Gcreate_plots(metric: str, real, pred, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))  # Dwa wykresy obok siebie

    # Wykres punktowy (scatter)
    create_scatter_plot(axs[0], metric, real, pred)

    # Wykres liniowy (line)
    create_line_plot(axs[1], metric, real, pred)

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{metric}_comparison.png"))
    plt.show()





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


def get_patient_data_csv(path, finder, stride):
    data = EcgData(SAMPLING_RATE, finder)
    data.load_csv_data(path)
    X_train, y_train = data.extract_hrv_windows_with_detected_peaks(
        input_length, METRICS
    )
    # X_train, y_train = data.extract_piotr(input_length, METRICS, stride)#, int(0.8 * input_length))
    return X_train, y_train


def get_patients_data_csv(dir_path: str, patients: list, finder, stride = None):
    X_train = None
    y_train = None
    for patient in patients:
        x, y = get_patient_data_csv(dir_path + "\\" + patient, finder, stride)
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

def create_scatter_line_plotsc(metric_name, y_true, y_pred, save_path=None):
    """
    Tworzy dwa rodzaje wykresów: scatter plot (rzeczywiste vs przewidywane)
    i line plot (rzeczywiste i przewidywane jako przebiegi czasowe).
    """
    # Scatter Plot: True vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_true)), y_true, alpha=0.5, label=f"True {metric_name}")
    plt.scatter(range(len(y_pred)), y_pred, alpha=0.5, label=f"Predicted {metric_name}", marker='x')
    plt.title(f"{metric_name}: True vs Predicted (Scatter)")
    plt.xlabel("Sample Index")
    plt.ylabel(f"{metric_name} Value")
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{metric_name}_scatter.png"))
    #plt.show(block = False)

    # # Line Plot: Comparison Over Time
    # plt.figure(figsize=(10, 6))
    # plt.plot(y_true, label=f"True {metric_name}")
    # plt.plot(y_pred, label=f"Predicted {metric_name}", linestyle='--')
    # plt.title(f"{metric_name}: True vs Predicted (Line)")
    # plt.xlabel("Sample Index")
    # plt.ylabel(f"{metric_name} Value")
    # plt.legend()
    # if save_path:
    #     plt.savefig(os.path.join(save_path, f"{metric_name}_line.png"))
    # plt.show()

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

# Custom loss function
def custom_loss(y_true, y_pred):
    """
    Custom loss function:
    - MSE for LF (first output)
    - 3 * MSE for HF (second output)
    """
    # LF: First column (index 0)
    mse_lf = K.mean(K.square(y_true[:, 0] - y_pred[:, 0]))
    # HF: Second column (index 1)
    mse_hf = K.mean(K.square(y_true[:, 1] - y_pred[:, 1]))
    # Weighted loss: MSE_LF + 3 * MSE_HF
    return mse_lf + 3 * mse_hf

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
    real_mean = np.mean(real)
    pred_mean = np.mean(pred)
    std_dev = np.std(real)
    std_dev_pred = np.std(pred)
    z_scores = (real - real_mean) / std_dev
    z_scores_pred = (pred- pred_mean)/  std_dev_pred
    threshold = 3
    filtered_data = real[np.abs(z_scores) <= threshold]
    filtered_predicts =  pred[np.abs(z_scores_pred) <= threshold]
    filt_real_pred = real[np.abs(z_scores_pred) <= threshold]
    real_mean_without_outliers = np.mean(filtered_data)
    r2_without_outliers = r2_score(filt_real_pred, filtered_predicts)
    
    mse_without_outliers = mean_squared_error(filt_real_pred, filtered_predicts)
    # metrics_text = f"Real mean value: {real_mean:.2f}\nReal mean without outliers: {real_mean_without_outliers:.2f}\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}\nR² without outliers: {r2_without_outliers:.2f}\nMSE without outliers {mse_without_outliers:.2f}"
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
    METRICS = ["LF", "HF"]#, "RMSSD"]
    input_length = 5*60*130  # Długość sekwencji interwałów RR
    # finder = UNetFinder(f"models/model_{WINDOW_SIZE}_{EPOCHS}_unet.keras", WINDOW_SIZE)
    finder = PanTompkinsFinder()
    # model = create_hrv_model(input_length, len(METRICS))
    # model = new_model(input_length, len(METRICS))
    # model = new_model(input_length, len(METRICS))

    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #     loss="mse",
    #     metrics=["mae"],
    # )

    kf = KFold(n_splits=5, shuffle=True)  # 5-krotna walidacja krzyżowa
    
    mae_per_metric_list= [[] for i in range(len(METRICS))]
    mse_per_metric_list= [[] for i in range(len(METRICS))]
    rmse_per_metric_list= [[] for i in range(len(METRICS))]
    r2_per_metric_list = [[] for i in range(len(METRICS))]
    
    lfhf_mae, lfhf_mse, lfhf_rmse, lfhf_r2 =[],[],[],[]
    X, y = get_patients_data_csv("data", [csvs[2], csvs[3], csvs[4], csvs[24]], finder)  # , int(0.8*input_length))
    # X, y = get_patients_data_mithbih(MITBIH_PATH, load_all_patient_indexes(f"{MITBIH_PATH}\\RECORDS"))  # , int(0.8*input_length))
    # X, y = get_patients_data_mithbih(MITBIH_PATH, ["101", "102", "103", "104", "105", "106", "107", "108", "109"])  # , int(0.8*input_length))
    # X, y = get_patients_data_csv("data", [csvs[24]], finder)  # , int(0.8*input_length))
    # random_seed = 42
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n### Fold {fold + 1} ###")

        # X_train, X_val, y_train, y_val = train_test_split(
        #     X, y, test_size=0.3, random_state=random_seed
        # )
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        ##################################################
        # input_length = X.shape[1]
        # model = new_model(input_length, len(METRICS))
        # model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        #     loss='mse',
        #     metrics=['mae']
        # )

        # model_path = f"models\\{fold}{"_".join(METRICS).replace("\\","").replace("/", "")}_{input_length}_{None}_{finder.__class__.__name__}.keras"

        ###########################################
        model = create_hrv_model(input_length, len(METRICS))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        model_path = f"models\\RR_{fold}{"_".join(METRICS).replace("\\","").replace("/", "")}_{input_length}_{None}_{finder.__class__.__name__}.keras"
        ########################
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="loss", verbose=1, save_best_only=True, mode="min"
        )
        callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=65)

        epochs= 500

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            verbose=1,
            shuffle=True,
            callbacks=[checkpoint, callback],
        )

        model.load_weights(model_path)

        pred_result = model.predict(X_val)
        
        for i in range(len(METRICS)):
            real = y_val[:, i]
            pred = pred_result[:, i]
            mae = mean_absolute_error(real, pred)
            mse = mean_squared_error(real, pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(real, pred)
            mae_per_metric_list[i].append(mae)
            mse_per_metric_list[i].append(mse)
            rmse_per_metric_list[i].append(rmse)
            r2_per_metric_list[i].append(r2)
            create_scatter_line_plots(f"{METRICS[i]} - Fold {fold +1}", real, pred, f"hrvPlots\\FOLD{fold+1}")
        
        # lfhf_real = y_val[:, 0]/y_val[:, 1]
        # lfhf_pred = pred_result[:, 0]/pred_result[:, 1]
        # mae = mean_absolute_error(lfhf_real, lfhf_pred)
        # mse = mean_squared_error(lfhf_real, lfhf_pred)
        # rmse = np.sqrt(rmse)
        # r2 = r2_score(lfhf_real, lfhf_pred)
        # lfhf_mae.append(mae)
        # lfhf_mse.append(mse)
        # lfhf_rmse.append(rmse)
        # lfhf_r2.append(r2)
        # create_scatter_line_plots(f"LF/HF - Fold {fold +1}", lfhf_real, lfhf_pred, f"hrvPlots\\FOLD{fold+1}")
        # break

    
    # Wyświetlanie średnich wyników
    print("\n### Wyniki Średnie ###")
    for i in range(len(METRICS)):
        print(f"### {METRICS[i]}")
        print(f"Średnie MAE: {np.mean(mae_per_metric_list[i]):.2f} ± {np.std(mae_per_metric_list[i]):.2f}")
        print(f"Średnie MSE: {np.mean(mse_per_metric_list[i]):.2f} ± {np.std(mse_per_metric_list[i]):.2f}")
        print(f"Średnie RMSE: {np.mean(rmse_per_metric_list[i]):.2f} ± {np.std(rmse_per_metric_list[i]):.2f}")
        print(f"Średnie R^2: {np.mean(r2_per_metric_list[i]):.2f} ± {np.std(r2_per_metric_list[i]):.2f}")


    # print(f"### LF/HF")
    # print(f"Średnie MAE: {np.mean(lfhf_mae[i]):.2f} ± {np.std(lfhf_mae[i]):.2f}")
    # print(f"Średnie MSE: {np.mean(lfhf_mse[i]):.2f} ± {np.std(lfhf_mse[i]):.2f}")
    # print(f"Średnie RMSE: {np.mean(lfhf_rmse[i]):.2f} ± {np.std(lfhf_rmse[i]):.2f}")
    # print(f"Średnie R^2: {np.mean(lfhf_r2[i]):.2f} ± {np.std(lfhf_r2[i]):.2f}")
    plt.show()

    pass




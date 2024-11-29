import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

from EcgData import EcgData
from Finders.PanTompkinsFinder import PanTompkinsFinder

# === PARAMETRY ===
WINDOW_SIZE = 500  # Liczba próbek w jednym oknie
OVERLAP = 0.5  # Nakładanie okien (50% zachodzenia)
EPOCHS = 50
BATCH_SIZE = 32
CSV_FILE = 'data//popoludnie_merged2.csv'  # Ścieżka do pliku CSV
SAMPLING_RATE = 130  # Częstotliwość próbkowania w Hz


# === FUNKCJE POMOCNICZE ===
def split_into_windows(signal, window_size, overlap):
    """Dzielenie sygnału na okna."""
    step = int(window_size * (1 - overlap))  # Skok między oknami
    windows = [
        signal[i:i + window_size]
        for i in range(0, len(signal) - window_size + 1, step)
    ]
    return np.array(windows)


def pan_tompkins_peak_detection(ecg_signal, sampling_rate=1000):
    """
    Wykrywanie szczytów R za pomocą algorytmu Pan-Tompkins.
    """
    # 1. Filtracja pasmowo-przepustowa (0.5–40 Hz)
    lowcut = 0.5
    highcut = 40.0
    nyquist = 0.5 * sampling_rate
    b, a = butter(1, [lowcut / nyquist, highcut / nyquist], btype="band")
    filtered_signal = filtfilt(b, a, ecg_signal)

    # 2. Różniczkowanie
    diff_signal = np.diff(filtered_signal)

    # 3. Potęgowanie (wzmacnia szczyty)
    squared_signal = diff_signal ** 2

    # 4. Uśrednianie ruchome (okno o długości 150 ms)
    window_size = int(0.150 * sampling_rate)  # 150 ms
    integrated_signal = np.convolve(squared_signal, np.ones(window_size) / window_size, mode='same')

    # 5. Wykrywanie szczytów
    threshold = 0.6 * np.max(integrated_signal)  # Dynamiczny próg
    peaks, _ = find_peaks(integrated_signal, height=threshold, distance=int(0.6 * sampling_rate))

    return peaks


def calculate_rr_intervals(window, sampling_rate=1000):
    """
    Oblicza odstępy RR na podstawie danych okna EKG, wykorzystując Pan-Tompkins.
    """
    r_peaks = pan_tompkins_peak_detection(window, sampling_rate)

    if len(r_peaks) > 1:  # Upewnij się, że są przynajmniej dwa szczyty
        rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # Konwersja na ms
    else:
        rr_intervals = []  # Brak wystarczających danych

    return rr_intervals


def generate_labels(windows, sampling_rate):
    """
    Oblicza rzeczywiste wartości SDNN i RMSSD dla każdego okna.
    """
    sdnn_values = []
    rmssd_values = []

    for window in windows:
        rr_intervals = calculate_rr_intervals(window, sampling_rate)

        if len(rr_intervals) > 1:  # Upewnij się, że są odstępy RR do analizy
            sdnn = np.std(rr_intervals)
            rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        else:
            sdnn = 0
            rmssd = 0

        sdnn_values.append(sdnn)
        rmssd_values.append(rmssd)


    return np.stack([sdnn_values, rmssd_values], axis=1)


def build_end_to_end_model(input_shape):
    """Budowa modelu end-to-end."""
    inputs = layers.Input(shape=input_shape)

    # Blok konwolucyjny do ekstrakcji cech
    x = layers.Conv1D(16, kernel_size=5, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Warstwa detekcji szczytów R
    r_peaks = layers.Conv1D(1, kernel_size=5, padding='same', activation='sigmoid', name="r_peak_detector")(x)

    # Symulacja obliczania odstępów RR
    rr_features = layers.GlobalAveragePooling1D()(r_peaks)

    # Warstwy gęste do przewidywania SDNN i RMSSD
    x = layers.Dense(128, activation='relu')(rr_features)
    outputs = layers.Dense(2, activation='linear', name="hrv_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# === Wczytywanie danych ===
data = pd.read_csv(CSV_FILE, header=None, names=['timestamp', 'ekg_value'])
ekg_signal = data['ekg_value'].values

# Dzielenie sygnału na okna
windows = split_into_windows(ekg_signal, WINDOW_SIZE, OVERLAP)

# Generowanie etykiet (SDNN, RMSSD)
labels = generate_labels(windows, sampling_rate=SAMPLING_RATE)

# Podzielenie danych na zbiór treningowy i walidacyjny
split = int(0.8 * len(windows))
train_signals = windows[:split]
train_labels = labels[:split]
val_signals = windows[split:]
val_labels = labels[split:]

# Normalizacja sygnału
train_signals = train_signals / np.max(np.abs(train_signals), axis=1, keepdims=True)
val_signals = val_signals / np.max(np.abs(val_signals), axis=1, keepdims=True)

# Dodanie wymiaru kanału
train_signals = train_signals[..., np.newaxis]
val_signals = val_signals[..., np.newaxis]

# === Budowa i trenowanie modelu ===
input_shape = (WINDOW_SIZE, 1)
model = build_end_to_end_model(input_shape)

# Kompilacja modelu
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Trenowanie modelu
history = model.fit(
    train_signals, train_labels,
    validation_data=(val_signals, val_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Ewaluacja modelu
val_loss, val_mae = model.evaluate(val_signals, val_labels)
print(f"Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")

# Przewidywanie dla zbioru walidacyjnego
predictions = model.predict(val_signals)

# === Wizualizacja wyników ===
plt.figure(figsize=(12, 6))

# SDNN
plt.subplot(2, 1, 1)
plt.plot(val_labels[:, 0], label='Rzeczywiste SDNN', color='blue')
plt.plot(predictions[:, 0], label='Przewidywane SDNN', color='orange', linestyle='dashed')
plt.title('Porównanie SDNN')
plt.xlabel('Próbka')
plt.ylabel('SDNN [ms]')
plt.legend()

# RMSSD
plt.subplot(2, 1, 2)
plt.plot(val_labels[:, 1], label='Rzeczywiste RMSSD', color='blue')
plt.plot(predictions[:, 1], label='Przewidywane RMSSD', color='orange', linestyle='dashed')
plt.title('Porównanie RMSSD')
plt.xlabel('Próbka')
plt.ylabel('RMSSD [ms]')
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
import scipy.signal as signal


def bandpass_filter(sig, frequency, lowcut: float = 5, highcut: float = 18):
    nyquist = 0.5 * frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(1, [low, high], btype="band")
    filtered_signal = signal.filtfilt(b, a, sig)
    return filtered_signal


def derivative_filter(sig):
    return np.diff(sig, prepend=0)


def square(sig):
    return sig**2


def moving_window_integration(sig, window_size):
    return np.convolve(sig, np.ones(window_size) / window_size, mode="same")

def refine_peak_positions(ecg_signal, detected_peaks, search_window=10):
    refined_peaks = []
    
    for peak in detected_peaks:
        start = max(peak - search_window, 0)  # Ensure the window doesn't go out of bounds
        end = min(peak + search_window, len(ecg_signal) - 1)
        
        refined_peak = np.argmax(ecg_signal[start:end]) + start
        refined_peaks.append(refined_peak)
    
    return np.array(refined_peaks)

def find_r_peaks(ecg_signal, frequency: float):
    filtered_signal = bandpass_filter(ecg_signal[:, 1], frequency)
    diff_signal = derivative_filter(filtered_signal)
    squared_signal = square(diff_signal)
    
    window_size = int(0.150 * frequency)  # 150 ms window
    integrated_signal = moving_window_integration(squared_signal, window_size)
    
    #threshold = 0.4 * np.max(integrated_signal)
    threshold = np.mean(integrated_signal) + 0.5 * np.std(integrated_signal)  # Mean + 0.5*std
    
    peaks, _ = signal.find_peaks(
        integrated_signal, height=threshold, distance=int(0.6 * frequency) # 600 ms
    )
    
    refined_peaks = refine_peak_positions(ecg_signal[:, 1], peaks)
    
    peak_values = ecg_signal[refined_peaks, 1]
    peak_timestamps = ecg_signal[refined_peaks, 0]
    
    peaks_with_timestamps = np.column_stack((peak_timestamps, peak_values))
    
    return peaks_with_timestamps


def find_r_peaks_piotr(raw_data):
    r_peaks = []  # Lista do przechowywania timestampów i wartości R-peaków
    threshold = 600  # Próg dla wartości sygnału (w mV)
    last_peak_time = -1  # Czas ostatniego wykrytego załamka R (inicjalnie brak)

    for index, (timestamp, value) in enumerate(raw_data):
        # Sprawdzamy, czy dane są nowsze od ostatnio wykrytego R-peaku
        if timestamp > last_peak_time:
            # Sprawdź, czy wartość sygnału przekracza próg
            if value > threshold:
                # Jeżeli to pierwsza wartość, ustaw ją jako początkową
                if index > 2:
                    # Oblicz pochodną (różnica wartości / różnica czasów)
                    derivative = (value - raw_data[index - 1][1])
                    prev_derivate = raw_data[index - 1][1] - raw_data[index - 2][1]
                    # Sprawdź, czy pochodna zmienia znak lub jest równa zero (szczyt załamka R)
                    if derivative <= 0 and prev_derivate >= 0:
                        # Zidentyfikowano szczyt, zapisujemy ten punkt jako załamek R
                        r_peaks.append(
                            (raw_data[index - 1][0], raw_data[index - 1][1]))  # Dodaj timestamp i wartość załamka R
                        last_peak_time = raw_data[index - 1][0]  # Zaktualizuj czas ostatniego załamka R

    # Zamień listę krotek na dwuwymiarową tablicę za pomocą np.column_stack()
    if r_peaks:
        peak_timestamps, peak_values = zip(*r_peaks)
        peaks_with_timestamps = np.column_stack((peak_timestamps, peak_values))
    else:
        peaks_with_timestamps = np.empty((0, 2))  # Zwróć pustą 2D tablicę, jeśli brak załamków R

    return peaks_with_timestamps

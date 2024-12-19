import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d

from config import RR_FREQ

#PLIK DO ROZPOZNAWANIA SZCZYTOW W SYGNALE

def bandpass_filter(sig, frequency, lowcut: float = 5, highcut: float = 18):
    nyquist = 0.5 * frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype="band")
    filtered_signal = signal.filtfilt(b, a, sig)
    return filtered_signal


def filter_ecg_with_timestamps(ecg_signal, frequency: float):
    timestamps = ecg_signal[:, 0]
    ecg_values = ecg_signal[:, 1]

    filtered_signal = bandpass_filter(ecg_values, frequency, lowcut=0.5, highcut=40)
    diff_signal = derivative_filter(filtered_signal)
    squared_signal = square(diff_signal)

    window_size = int(0.150 * frequency)  # 150 ms window
    integrated_signal = moving_window_integration(squared_signal, window_size)

    # Combine timestamps with filtered signal values
    filtered_signal_with_timestamps = np.column_stack((timestamps, integrated_signal))

    return filtered_signal_with_timestamps


def derivative_filter(sig):
    return np.diff(sig, prepend=0)


def square(sig):
    return sig**2


def moving_window_integration(sig, window_size):
    return np.convolve(sig, np.ones(window_size), mode="same")

def refine_peak_positions(ecg_signal, detected_peaks, search_window=10):
    refined_peaks = []
    
    for peak in detected_peaks:
        start = max(peak - search_window, 0)  # Ensure the window doesn't go out of bounds
        end = min(peak + search_window, len(ecg_signal) - 1)

        refined_peak = np.argmax(ecg_signal[start:end]) + start
        refined_peaks.append(refined_peak)
    
    return np.array(refined_peaks)

def find_r_peaks_values(ecg_signal, frequency: float):
    peaks = find_r_peaks_ind(ecg_signal, frequency)
    
    return ecg_signal[peaks, 1]


def find_r_peaks_values_with_timestamps(ecg_signal, frequency: float):
    peaks = find_r_peaks_ind(ecg_signal[:, 1], frequency)
    
    peak_values = ecg_signal[peaks, 1]
    peak_timestamps = ecg_signal[peaks, 0]
    
    peaks_with_timestamps = np.column_stack((peak_timestamps, peak_values))
    
    return peaks_with_timestamps


def find_r_peaks_ind(ecg_signal, frequency: float):
    # filtered_signal = bandpass_filter(ecg_signal[:, 1], frequency)
    filtered_signal = bandpass_filter(ecg_signal, frequency)
    diff_signal = derivative_filter(filtered_signal)
    squared_signal = square(diff_signal)
    
    window_size = int(0.050 * frequency)  # 50 ms window
    integrated_signal = moving_window_integration(squared_signal, window_size)
    
    clipped_signal = np.clip(integrated_signal, 0, np.percentile(integrated_signal, 99))
    threshold = np.mean(clipped_signal) + 0.6 * np.std(clipped_signal)
    peaks, _ = signal.find_peaks(
        integrated_signal, height=threshold, distance=int(0.4 * frequency) # 400 ms
    )

    
    refined_peaks = refine_peak_positions(ecg_signal, peaks, round(10/130*frequency))
    # refined_peaks = processing.correct_peaks(
    #     sig=ecg_signal,
    #     peak_inds=peaks,
    #     search_radius=30,
    #     smooth_window_size=30,
    #     # peak_dir="up",
    # )
    # refined_peaks = peaks
    
    return refined_peaks

def find_hr_peaks(hr_signal, frequency: float, lowcut: float = 5, highcut: float = 18, size: int = 7, isUpright:int = 1):
    timestamps = hr_signal[:, 0]
    hr_values = hr_signal[:, 1]




    # Generowanie nowych timestampów z częstotliwością 4 Hz
    new_timestamps = np.arange(timestamps[0], timestamps[-1], 1 / frequency)

    # Interpolacja (używamy metody liniowej)
    interpolator = interp1d(timestamps, hr_values, kind='linear', fill_value="extrapolate")
    new_hr_values = interpolator(new_timestamps)


    filtered_signal = bandpass_filter(new_hr_values, frequency, lowcut, highcut)

    filtered_signal = isUpright*filtered_signal

    threshold = np.mean(filtered_signal) + isUpright * 0.1 * np.std(filtered_signal)  # Mean + 0.6*std
    # threshold = 0.4 * np.max(integrated_signal)


    peaks, _ = signal.find_peaks(
        filtered_signal, height=threshold, distance=int(size * RR_FREQ)  # 400 ms
    )

    signal_to_filtered_ratio = (len(new_hr_values)/len(hr_values))

    # Podzielenie indeksów na pół i zaokrąglenie do najbliższej liczby całkowitej
    peaks = np.floor(peaks / signal_to_filtered_ratio).astype(int)
    if isUpright == -1:
        refined_peaks = refine_peak_positions(-hr_signal[:, 1], peaks, search_window=size)
    else:
        refined_peaks = refine_peak_positions(hr_signal[:, 1], peaks, search_window=size)


    peak_values = hr_signal[refined_peaks, 1]
    peak_timestamps = hr_signal[refined_peaks, 0]

    peaks_with_timestamps = np.column_stack((peak_timestamps, peak_values))

    return peaks_with_timestamps


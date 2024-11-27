import numpy as np
import scipy.signal as signal


def bandpass_filter(sig, frequency, lowcut: float = 5, highcut: float = 18):
    nyquist = 0.5 * frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype="band")
    filtered_signal = signal.filtfilt(b, a, sig)
    return filtered_signal


def filter_ecg_with_timestamps(ecg_signal, frequency: float):
    # Extract timestamps and ECG values from the input
    timestamps = ecg_signal[:, 0]
    ecg_values = ecg_signal[:, 1]

    # Apply bandpass filter to the ECG values
    filtered_signal = bandpass_filter(ecg_values, frequency, lowcut=0.5, highcut=40)
    #filtered_signal = ecg_values
    diff_signal = derivative_filter(filtered_signal)
    squared_signal = square(diff_signal)

    window_size = int(0.050 * frequency)  # 50 ms window
    integrated_signal = moving_window_integration(squared_signal, window_size)

    # Combine timestamps with filtered signal values
    filtered_signal_with_timestamps = np.column_stack((timestamps, integrated_signal))

    return filtered_signal_with_timestamps


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

def find_r_peaks_values(ecg_signal, frequency: float):
    peaks = find_r_peaks_ind(ecg_signal, frequency)
    
    peak_values = ecg_signal[peaks, 1]
    # peak_timestamps = ecg_signal[peaks, 0]
    
    # peaks_with_timestamps = np.column_stack((peak_timestamps, peak_values))
    
    # return peaks_with_timestamps
    return peak_values


def find_r_peaks_values_with_timestamps(ecg_signal, frequency: float):
    peaks = find_r_peaks_ind(ecg_signal[:, 1], frequency)
    
    peak_values = ecg_signal[peaks, 1]
    peak_timestamps = ecg_signal[peaks, 0]
    
    peaks_with_timestamps = np.column_stack((peak_timestamps, peak_values))
    
    return peaks_with_timestamps

# def find_r_peaks_ind(ecg_signal, frequency: float):
#     # filtered_signal = bandpass_filter(ecg_signal[:, 1], frequency)
#     filtered_signal = bandpass_filter(ecg_signal, frequency)
#     diff_signal = derivative_filter(filtered_signal)
#     squared_signal = square(diff_signal)
    
#     window_size = int(0.05 * frequency)  # 50 ms window
#     integrated_signal = moving_window_integration(squared_signal, window_size)
    
#   #  threshold = 0.4 * np.max(integrated_signal)
#     threshold = np.mean(integrated_signal) + 0.6 * np.std(integrated_signal)  # Mean + 0.6*std

#     peaks, _ = signal.find_peaks(
#         integrated_signal, height=threshold, distance=int(size * frequency) # 400 ms
#     )
    
#     refined_peaks = refine_peak_positions(ecg_signal[:, 1], peaks)
    
#     peak_values = ecg_signal[refined_peaks, 1]
#     peak_timestamps = ecg_signal[refined_peaks, 0]
    
#     peaks_with_timestamps = np.column_stack((peak_timestamps, peak_values))
    
#     return refined_peaks

def find_r_peaks_ind(ecg_signal, frequency: float):
    # filtered_signal = bandpass_filter(ecg_signal[:, 1], frequency)
    filtered_signal = bandpass_filter(ecg_signal, frequency)
    diff_signal = derivative_filter(filtered_signal)
    squared_signal = square(diff_signal)
    
    window_size = int(0.050 * frequency)  # 50 ms window
    integrated_signal = moving_window_integration(squared_signal, window_size)
    
    #threshold = 0.4 * np.max(integrated_signal)
    threshold = np.mean(integrated_signal) + 0.6 * np.std(integrated_signal)  # Mean + 0.6*std
    
    peaks, _ = signal.find_peaks(
        integrated_signal, height=threshold, distance=int(0.4 * frequency) # 400 ms
    )
    
    # refined_peaks = refine_peak_positions(ecg_signal[:, 1], peaks)
    refined_peaks = refine_peak_positions(ecg_signal, peaks)
    
    return refined_peaks

def find_hr_peaks(ecg_signal, frequency: float, lowcut: float = 5, highcut: float = 18, size: int = 7, isUpright:int = 1):

    filtered_signal = bandpass_filter(ecg_signal[:, 1], frequency, lowcut, highcut)

    filtered_signal = isUpright*filtered_signal

    threshold = np.mean(filtered_signal) + isUpright * 0.1 * np.std(filtered_signal)  # Mean + 0.6*std
    # threshold = 0.4 * np.max(integrated_signal)


    peaks, _ = signal.find_peaks(
        filtered_signal, height=threshold, distance=int(size * 1.22)  # 400 ms
    )


    if isUpright == -1:
        refined_peaks = refine_peak_positions(-ecg_signal[:, 1], peaks, search_window=size)
    else:
        refined_peaks = refine_peak_positions(ecg_signal[:, 1], peaks, search_window=size)

    peak_values = ecg_signal[refined_peaks, 1]
    peak_timestamps = ecg_signal[refined_peaks, 0]

    peaks_with_timestamps = np.column_stack((peak_timestamps, peak_values))

    return peaks_with_timestamps


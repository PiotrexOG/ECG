import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

import EcgData

#PLIK DO ANALIZOWANIA MIAR HRV Z CALEJ DOBY

def analyze(self_raw_data, self_rr_intervals):

    sampling_rate = 130  # Hz
    window_duration = 100  # seconds (5 minutes)
    window_samples = window_duration * sampling_rate

    # Calculate RMSSD for each window
    rmssd_values = []
    time_windows = []
    mean_nn_values = []
    sdnn_values = []

    for i in range(0, len(self_raw_data), window_samples):
        window_start = self_raw_data[i][0]
        window_end = self_raw_data[min(i + window_samples, len(self_raw_data) - 1)][0]

        # Filter RR intervals within the current window
        rr_in_window = self_rr_intervals[
            (self_rr_intervals[:, 0] >= window_start) & (self_rr_intervals[:, 0] < window_end), 1
        ]

        if len(rr_in_window) > 1:
            rr_diffs = np.diff(rr_in_window)
            rmssd = np.sqrt(np.mean(rr_diffs**2)) * 1e3
            mean_nn = np.mean(rr_in_window) * 1e3
            sdnn = np.std(rr_in_window) * 1e3

            mean_nn_values.append(mean_nn)
            sdnn_values.append(sdnn)
            rmssd_values.append(rmssd)
            time_windows.append((window_start + window_end) / 2)
        else:
            rmssd_values.append(None)
            time_windows.append((window_start + window_end) / 2)

    sdann = np.std(mean_nn_values)
    sdnn_index = np.mean(sdnn_values)

    return sdann, sdnn_index



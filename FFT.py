import numpy as np
import matplotlib.pyplot as plt

#PLIK DO LICZENIA LF I HF

def calc_lf_hf(intervals):
    rr_intervals = intervals  # Zastąp swoimi wartościami

    # Krok 1: Interpolacja sygnału
    fs = 4  # Przykładowa częstotliwość próbkowania (4 Hz)
    time = np.cumsum(rr_intervals)  # Czas narastający
    interp_time = np.linspace(time[0], time[-1], int((time[-1] - time[0]) * fs))
    interp_rr = np.interp(interp_time, time, rr_intervals)

    window = np.hanning(len(interp_rr))
    fft_result = np.fft.fft(interp_rr * window)
    freqs = np.fft.fftfreq(len(fft_result), 1 / fs)
    
    # Filtracja do pasma HF (0.15-0.4 Hz)
    hf_band = (freqs >= 0.15) & (freqs <= 0.4)
    hf_power = np.sum(np.abs(fft_result[hf_band]) ** 2)  # Moc w paśmie HF
    
    # Filtracja do pasma HF (0.15-0.4 Hz)
    lf_band = (freqs >= 0.04) & (freqs <= 0.15)
    lf_power = np.sum(np.abs(fft_result[lf_band]) ** 2)  # Moc w paśmie HF
    
    return lf_power, hf_power





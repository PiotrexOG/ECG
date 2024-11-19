import numpy as np
import matplotlib.pyplot as plt


def fft(intervals):
    # Przykladowe wartości RR (odstępy czasowe w sekundach)
    rr_intervals = intervals  # Zastąp swoimi wartościami

    # Krok 1: Interpolacja sygnału
    fs = 4  # Przykładowa częstotliwość próbkowania (4 Hz)
    time = np.cumsum(rr_intervals)  # Czas narastający
    interp_time = np.linspace(time[0], time[-1], int((time[-1] - time[0]) * fs))
    interp_rr = np.interp(interp_time, time, rr_intervals)

    # Krok 2: Przeprowadzenie FFT
    fft_result = np.fft.fft(interp_rr)
    freqs = np.fft.fftfreq(len(fft_result), 1 / fs)

    # Filtracja do pasma HF (0.15-0.4 Hz)
    hf_band = (freqs >= 0.15) & (freqs <= 0.4)
    hf_power = np.sum(np.abs(fft_result[hf_band]) ** 2)  # Moc w paśmie HF

    # Filtracja do pasma HF (0.15-0.4 Hz)
    lf_band = (freqs >= 0.04) & (freqs <= 0.15)
    lf_power = np.sum(np.abs(fft_result[lf_band]) ** 2)  # Moc w paśmie HF

    # Wykres widma mocy
    positive_freqs = freqs[freqs >= 0]
    positive_fft = np.abs(fft_result[freqs >= 0])

    print (f"moc w hf: {hf_power}")
    print (f"moc w lf: {lf_power}")

    plt.figure()
    plt.plot(positive_freqs, positive_fft)
    plt.title("Widmo mocy HRV - Analiza HF")
    plt.xlabel("Częstotliwość (Hz)")
    plt.ylabel("Amplituda")
    plt.xlim(0, 0.5)  # Ograniczenie widoku do 0.5 Hz (możesz zmodyfikować)
    plt.axvspan(0.15, 0.4, color='lightgray', alpha=0.5, label="Pasmo HF (0.15-0.4 Hz)")
    plt.legend()
    plt.show()



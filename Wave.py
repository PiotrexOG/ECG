import numpy as np
from scipy.interpolate import interp1d
import pywt
import matplotlib.pyplot as plt


def calculate_scales(f_min, f_max, sampling_rate, f_c):
    delta_t = 1 / sampling_rate
    frequencies = np.arange(f_min, f_max, 0.01)  # Dla precyzji co 0.01 Hz
    scales = f_c / (frequencies * delta_t)
    return scales

def analyze(intervals):
    # Wstępna analiza sygnału RR
    rr_intervals = intervals[:, 1]
    time = intervals[:, 0] - intervals[0, 0]
    sampling_rate = 4  # Większa częstotliwość próbkowania
    new_time = np.arange(time[0], time[-1], 1 / sampling_rate)

    # Interpolacja sygnału RR
    interpolated_rr = interp1d(time, rr_intervals, kind='cubic')(new_time)
    B = 1.5
    C = 1.5
    wavelet = f'cmor{B}-{C}'

    # Przykład dla f_min = 0.03 Hz, f_max = 0.5 Hz, sampling_rate = 10 Hz
    scales = calculate_scales(0.03, 0.5, sampling_rate=sampling_rate, f_c=C)

    fs = sampling_rate
    sampling_period = 1 / fs

    # Transformacja falek
    coefficients, frequencies = pywt.cwt(interpolated_rr, scales, wavelet, sampling_period=sampling_period)

    # Analiza mocy
    power = np.abs(coefficients)**2
    #power[power > 0.08] = 0
    power = np.log10(power + 1e-6)  # Logarytmowanie dla poprawy widoczności


    # Wizualizacja skalogramu
    plt.figure(figsize=(12, 8))
    plt.contourf(new_time, frequencies, power, levels=100, cmap='jet')#, vmin=0, vmax=0.25)


    plt.colorbar(label="Log Power")
    plt.title("Scalogram (Wavelet Transform)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    #plt.ylim(0.04, 0.4)  # Zakres częstotliwości bliski RSA
    plt.legend()
    plt.grid(True)
    plt.show(block=False)  # Nie blokuje programu

# # Dynamiczny sygnał testowy z harmonicznymi i zmienną amplitudą
# t_long = np.linspace(0, 80, 100)  # Więcej punktów czasowych
# rr_intervals_long = (
#     0.8
#     + (0.08 * np.sin(2 * np.pi * 0.02 * t_long))
# )
#
# plt.figure(figsize=(10, 5))
# plt.plot(t_long, rr_intervals_long, label='RR Intervals')
# plt.title("RR Intervals over Time")
# plt.xlabel("Time (s)")
# plt.ylabel("RR Interval (s)")
# plt.grid(True)
# plt.show()

# test = np.column_stack((t_long, rr_intervals_long))
#
# #Analiza z wyświetleniem skalogramu
# analyze_rr_intervals_with_scalogram(test)


import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from EcgData import *
from config import *

#PLIK DO ANALIZOWANIA FFT

class FFTPlotter:
    def __init__(self, title: str, ecg_data: EcgData):
        self.ecg_data = ecg_data
        self.fig, self.ax_fft = plt.subplots(1, 1, figsize=(6, 4))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

        self.ax_fft.set_title(title + " Analiza spektralna")
        self.ax_fft.set_ylabel("PSD [msec^2/Hz]")
        self.ax_fft.set_xlabel("Częstotliwość (Hz)")
        self.ax_fft.tick_params(axis="x")
        self.ax_fft.tick_params(axis="y")
        self.ax_fft.set_xlim(0, 0.5)
        self.ax_fft.axvspan(0.15, 0.4, color='lightblue' , label="Pasmo HF (0.15-0.4 Hz)")
        self.ax_fft.axvspan(0.04, 0.15, color='lightgray', label="Pasmo LF (0.04-0.15 Hz)")

        self.line_fft, = self.ax_fft.plot([], [], color="green", label="FFT")
        self.ax_fft.legend(loc="upper right")

        # Placeholdery na wartości LF, HF i LF/HF
        self.lf_hf_text = self.ax_fft.text(
            0.01, 0.95, "", transform=self.ax_fft.transAxes, fontsize=10, verticalalignment="top"
        )

        self.timer = self.fig.canvas.new_timer(interval=1000)
        self.timer.add_callback(self.check_for_data)
        self.timer.start()
        self.data_handled = False  # Flaga, aby wywołać update_plot tylko raz

    def check_for_data(self):
        if not self.data_handled and len(self.ecg_data.r_peaks) > 0:
            self.update_plot()
            if APP_MODE == AppModeEnum.LOAD_CSV:
                self.data_handled = True  # Zapobiega kolejnemu wywołaniu

    def compute_fft(self, rr_intervals):
        if len(rr_intervals) < 2:
            return np.array([]), np.array([])

        fs = 4  # Przykładowa częstotliwość próbkowania (4 Hz)
        time = np.cumsum(rr_intervals)
        interp_time = np.linspace(time[0], time[-1], int((time[-1] - time[0]) * fs))
        interp_rr = np.interp(interp_time, time, rr_intervals)

        window = np.hanning(len(interp_rr))
        fft_result = np.fft.fft(interp_rr * window)
        freqs = np.fft.fftfreq(len(fft_result), 1 / fs)

        positive_freqs = freqs[freqs >= 0]
        positive_fft = np.abs(fft_result[freqs >= 0])

        return positive_freqs, positive_fft

    def compute_lf_hf(self, freqs, fft_amplitudes):
        # Pasmo LF (0.04–0.15 Hz) i HF (0.15–0.4 Hz)
        lf_band = (freqs >= 0.04) & (freqs <= 0.15)
        hf_band = (freqs >= 0.15) & (freqs <= 0.4)

        lf_power = np.sum(fft_amplitudes[lf_band] ** 2)
        hf_power = np.sum(fft_amplitudes[hf_band] ** 2)

        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.inf
        return lf_power, hf_power, lf_hf_ratio

    def update_plot(self):
        if self.ecg_data:
            if self.ecg_data.rr_intervals.any():
                freqs, fft_amplitudes = self.compute_fft(self.ecg_data.rr_intervals[:, 1])

                if len(freqs) > 0:
                    self.line_fft.set_data(freqs, fft_amplitudes)
                    self.ax_fft.relim()
                    self.ax_fft.autoscale_view()

                    # Oblicz LF, HF i LF/HF
                    lf_power, hf_power, lf_hf_ratio = self.compute_lf_hf(freqs, fft_amplitudes)

                    # Aktualizacja tekstu
                    self.lf_hf_text.set_text(
                        f"LF Power: {lf_power:.2f}\nHF Power: {hf_power:.2f}\nLF/HF: {lf_hf_ratio:.2f}"
                    )

                self.fig.canvas.draw_idle()



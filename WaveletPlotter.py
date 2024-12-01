import numpy as np
from scipy.interpolate import interp1d
import pywt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class WaveletPlotter:
    def __init__(self, title: str, ecg_data):
        self.ecg_data = ecg_data

        # Tworzenie wykresów: 2 wiersze, 2 kolumny
        self.fig, (self.ax_scalogram, self.ax_power, self.ax_ratio) = plt.subplots(
            3, 1, figsize=(7, 5)
        )
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)

        # Skalogram
        self.ax_scalogram.set_title(title + " - Scalogram (Wavelet Transform)")
        self.ax_scalogram.set_xlabel("Time (s)")
        self.ax_scalogram.set_ylabel("Frequency (Hz)")

        # Kolorbar dla mocy
        self.norm = Normalize(vmin=-6, vmax=0)  # Zakres dla logarytmowanej mocy
        self.cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=self.norm, cmap="jet"), ax=self.ax_scalogram
        )
        self.cbar.set_label("Log Power")

        # Wykres mocy HF i LF
        self.ax_power.set_title("Power in HF and LF Bands")
        self.ax_power.set_xlabel("Time (s)")
        self.ax_power.set_ylabel("Power")
        self.line_hf, = self.ax_power.plot([], [], color="blue", label="HF Power")
        self.line_lf, = self.ax_power.plot([], [], color="red", label="LF Power")
        self.ax_power.legend(loc="upper right")

        # Wykres stosunku LF/HF
        self.ax_ratio.set_title("LF/HF Ratio Over Time")
        self.ax_ratio.set_xlabel("Time (s)")
        self.ax_ratio.set_ylabel("LF/HF Ratio")
        self.line_ratio, = self.ax_ratio.plot([], [], color="green", label="LF/HF Ratio")
        self.ax_ratio.legend(loc="upper right")

        # Bufory na moc w HF i LF oraz stosunek LF/HF
        self.hf_power = []
        self.lf_power = []
        self.ratio = []
        self.time_buffer = []

        self.timer = self.fig.canvas.new_timer(interval=5000)
        self.timer.add_callback(self.update_plot)
        self.timer.start()

    def calculate_scales(self, f_min, f_max, sampling_rate, f_c):
        delta_t = 1 / sampling_rate
        frequencies = np.arange(f_min, f_max, 0.01)
        scales = f_c / (frequencies * delta_t)
        return scales, frequencies

    def analyze_rr_intervals(self, intervals):
        if len(intervals) < 2:
            return np.array([]), np.array([]), np.array([]), np.array([])

        rr_intervals = intervals[:, 1]
        time = intervals[:, 0] - intervals[0, 0]
        sampling_rate = 4

        if len(rr_intervals) < 4:
            interp_method = "linear"
        else:
            interp_method = "cubic"

        try:
            new_time = np.arange(time[0], time[-1], 1 / sampling_rate)
            interpolated_rr = interp1d(time, rr_intervals, kind=interp_method)(new_time)
        except ValueError:
            return np.array([]), np.array([]), np.array([]), np.array([])

        B, C = 1.5, 1.5
        wavelet = f"cmor{B}-{C}"

        # Skale i częstotliwości
        scales, frequencies = self.calculate_scales(0.03, 0.5, sampling_rate, f_c=C)

        sampling_period = 1 / sampling_rate
        coefficients, _ = pywt.cwt(interpolated_rr, scales, wavelet, sampling_period=sampling_period)

        # Logarytmowana moc
        power = np.log10(np.abs(coefficients) ** 2 + 1e-6)
        return new_time, frequencies, power, coefficients

    def compute_band_power_from_coef(self, frequencies, coefficients, band_min, band_max):
        band = (frequencies >= band_min) & (frequencies <= band_max)
        band_power = np.sum(np.abs(coefficients[band, :]) ** 2, axis=0)  # Moc jako suma kwadratów amplitud
        return band_power

    def update_plot(self):
        if self.ecg_data and len(self.ecg_data.rr_intervals) >= 2:
            rr_intervals = self.ecg_data.rr_intervals
            new_time, frequencies, power, coefs = self.analyze_rr_intervals(rr_intervals)

            if len(new_time) > 0:
                # Aktualizacja danych skalogramu
                if not hasattr(self, 'scalogram_quadmesh'):
                    self.scalogram_quadmesh = self.ax_scalogram.contourf(
                        new_time, frequencies, power, levels=100, cmap="jet", norm=self.norm
                    )
                    self.ax_scalogram.set_ylim(0.03, 0.5)
                else:
                    for coll in self.scalogram_quadmesh.collections:
                        coll.remove()
                    self.scalogram_quadmesh = self.ax_scalogram.contourf(
                        new_time, frequencies, power, levels=100, cmap="jet", norm=self.norm
                    )

                # Moc w pasmach HF i LF
                hf_power = self.compute_band_power_from_coef(frequencies, coefs, 0.15, 0.4)
                lf_power = self.compute_band_power_from_coef(frequencies, coefs, 0.04, 0.15)

                # Buforowanie mocy i stosunku LF/HF
                self.hf_power = hf_power
                self.lf_power = lf_power
                self.time_buffer = new_time
                self.ratio = np.divide(self.lf_power, self.hf_power, out=np.zeros_like(self.lf_power), where=self.hf_power > 0)

                # Aktualizacja wykresu mocy
                self.line_hf.set_data(self.time_buffer, self.hf_power)
                self.line_lf.set_data(self.time_buffer, self.lf_power)

                self.ax_power.relim()
                self.ax_power.autoscale_view()

                # Aktualizacja wykresu stosunku LF/HF
                self.line_ratio.set_data(self.time_buffer, self.ratio)
                self.ax_ratio.relim()
                self.ax_ratio.autoscale_view()

                self.fig.canvas.draw_idle()



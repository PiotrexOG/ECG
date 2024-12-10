import numpy as np
from scipy.interpolate import interp1d
import pywt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import gridspec, ticker

from config import*

#PLIK DO ANALIZY CZASOWO-CZESTOTLIWOSCIOWEJ

class WaveletPlotter:
    def __init__(self, title: str, ecg_data):
        self.ecg_data = ecg_data

        # Tworzenie układu z GridSpec
        self.fig = plt.figure(figsize=(9, 6))
        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.05], height_ratios=[1, 1, 1])
        plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)

        # Skalogram
        self.ax_scalogram = self.fig.add_subplot(gs[0, 0])
        self.ax_scalogram.set_title("Analiza falkowa")
        self.ax_scalogram.set_xlabel("Godzina dnia")
        self.ax_scalogram.set_ylabel("Częstotliwość (Hz)")

        # Kolorbar dla mocy
        self.norm = Normalize(vmin=-6, vmax=0)  # Zakres dla logarytmowanej mocy
        cbar_ax = self.fig.add_subplot(gs[0, 1])  # Osobny subplot dla kolorbara
        self.cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=self.norm, cmap="jet"), cax=cbar_ax
        )
        self.cbar.set_label("Zlogarytmowana moc")

        # Wykres mocy HF i LF
        self.ax_power = self.fig.add_subplot(gs[1, 0])
        self.ax_power.set_title("Moc w pasmach HF i LF")
        self.ax_power.set_xlabel("Godzina dnia")
        self.ax_power.set_ylabel("PSD [msec^2/Hz]")
        self.create_colored_background(self.ax_power)
        #self.line_hf, = self.ax_power.plot([], [],  color="blue", label="Moc w HF")
        # self.line_hfU, = self.ax_power.plot([], [], color="green", label="Moc w HF U")
        # self.line_hfL, = self.ax_power.plot([], [], color="blue", label="Moc w HF L")
        self.line_lfG, = self.ax_power.plot([], [], color="red",     label="Moc w LF, f. Gaussa")
        self.line_hfG, = self.ax_power.plot([], [], color="blue", label="Moc w HF, f Gaussa")

        #self.line_lf, = self.ax_power.plot([], [],  color="red",         alpha = 0.33,    label="Moc w LF")
        #self.line_lfU, = self.ax_power.plot([], [], color="blue",   label="Moc w LF, f. Jednorodny")
        #self.line_lfL, = self.ax_power.plot([], [], color="green",     label="Moc w LF, f. Liniowy")

        #self.line_lfE, = self.ax_power.plot([], [], color="brown",     label="Moc w LF E")

        #self.line_lf, = self.ax_power.plot([], [], color="red", label="Moc w LF")
        self.ax_power.legend(loc="upper left")

        # Wykres stosunku LF/HF
        self.ax_ratio = self.fig.add_subplot(gs[2, 0])
        self.ax_ratio.set_title("Stosunek LF/HF w czasie")
        self.ax_ratio.set_xlabel("Godzina dnia")
        self.create_colored_background(self.ax_ratio)

        for ax in [self.ax_scalogram, self.ax_power, self.ax_ratio]:
            # Zakres osi X - dostosuj, jeśli masz konkretny przedział czasowy
            ax.xaxis.set_major_locator(ticker.MultipleLocator(3600))  # Znaczniki co 3600 sekund (1 godzina)
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(
                    ticker.FuncFormatter(lambda x, _: f'{int(x // 3600 + 19) % 24:02d}:{int((x % 3600) // 60):02d}')))



        self.ax_ratio.set_ylabel("Stosunek LF/HF")
        self.line_ratio, = self.ax_ratio.plot([], [], color="green", label="GF.LF/ GF.HF")
        self.line_ratio_raw, = self.ax_ratio.plot([], [], color="orange", label="GF. (LF/HF)")
        self.ax_ratio.legend(loc="upper left")

        # Bufory na moc w HF i LF oraz stosunek LF/HF
        self.hf_power = []
        self.lf_power = []
        self.ratio = []
        self.time_buffer = []

        # start_time = 0
        # end_time = 1080  # Możesz ustawić końcowy czas, jeśli jest znany
        # interval = 120
        # vertical_lines_time = [32940, 62460]
        #
        # for t in vertical_lines_time:
        #     self.ax_scalogram.axvline(x=t, color="purple", linestyle="--", label=f"t = {t}")
        #     self.ax_power.axvline(x=t, color="purple", linestyle="--", label=f"t = {t}")
        #     self.ax_ratio.axvline(x=t, color="purple", linestyle="--", label=f"t = {t}")

        # # Dodanie linii poziomych tylko pomiędzy kolejnymi pionowymi liniami
        # horizontal_lines_values = [1 / (2 * x) for x in range(2, 11)]  # Lista poziomych linii [0.25, ..., 0.05]
        #
        # # Iteracja po parach sąsiadujących czasów
        # for i in range(len(vertical_lines_time) - 1):
        #     t_start = vertical_lines_time[i]
        #     t_end = vertical_lines_time[i + 1]
        #
        #     height = horizontal_lines_values[i]
        #
        #     self.ax_scalogram.plot([t_start, t_end], [height,height], color="brown", linestyle="--")

        self.timer = self.fig.canvas.new_timer(interval=300)
        self.timer.add_callback(self.check_for_data)
        self.timer.start()
        self.data_handled = False  # Flaga, aby wywołać update_plot tylko raz

    def create_colored_background(self, ax):
        color_sequence = [1, 0, 1, 3, 1, 0, 3, 1, 0, 1, 2, 3, 2, 1, 3, 1, 3, 1, 3, 1, 0, 3, 0, 1, 0, 1, 0, 1, 2, 3, 2,
                          1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 3, 1, 3, 1, 2, 3, 2, 1, 2, 3, 2, 1,
                          2, 3, 1, 2, 3]
        durations = [46, 17, 2, 5, 13, 55, 5, 53, 83, 30, 14, 9, 5, 9, 4, 15, 6, 25, 7, 44, 2, 1, 2, 47, 5, 34, 18, 10,
                     5, 49, 3, 96, 5, 7, 3, 10, 6, 13, 4, 6, 58, 8, 2, 69, 31, 2, 2, 14, 2, 8, 4, 4, 3, 13, 16, 4, 1,
                     22, 4, 8, 5, 14, 8, 4, 3, 13, 14]

        total_time = 29520  # Match x-axis range

        custom_colors = [
            (58, 71, 228),  # Navy (ciemny niebieski)
            (101, 120, 232),  # Blue (niebieski)
            (117, 187, 249),  # Turquoise (turkusowy)
            (238, 128, 76)  # Orange (pomarańczowy)
        ]
        colors = [(r / 255, g / 255, b / 255) for r, g, b in custom_colors]

        # Normalize durations to the total time
        total_duration = sum(durations)
        normalized_durations = [d / total_duration * total_time for d in durations]

        # Plot colored background rectangles
        start_time = 32940
        for idx, duration in zip(color_sequence, normalized_durations):
            ax.axvspan(start_time, start_time + duration, color=colors[idx], alpha=0.6, lw = 0)
            start_time += duration

    def check_for_data(self):
        if not self.data_handled and len(self.ecg_data.r_peaks) > 0:
            self.update_plot()
            if APP_MODE == AppModeEnum.LOAD_CSV:
                self.data_handled = True  # Zapobiega kolejnemu wywołaniu

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

        B, C = 15, 0.22
        wavelet = f"cmor{B}-{C}"

        # Skale i częstotliwości
        scales, frequencies = self.calculate_scales(0.04, 0.41, sampling_rate, f_c=C)

        sampling_period = 1 / sampling_rate
        coefficients, _ = pywt.cwt(interpolated_rr, scales, wavelet, sampling_period=sampling_period)

        # Logarytmowana moc
        power = np.log10(np.abs(coefficients) ** 2)
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
                    self.ax_scalogram.set_ylim(0.04, 0.4)
                else:
                    for coll in self.scalogram_quadmesh.collections:
                        coll.remove()
                    self.scalogram_quadmesh = self.ax_scalogram.contourf(
                        new_time, frequencies, power, levels=100, cmap="jet", norm=self.norm
                    )

                # Moc w pasmach HF i LF
                hf_power = self.compute_band_power_from_coef(frequencies, coefs, 0.15, 0.4)
                lf_power = self.compute_band_power_from_coef(frequencies, coefs, 0.04, 0.15)


                # Jeśli globalna zmienna `use_average` jest ustawiona na True, oblicz średnią
                if GAUSS_WINDOW_SIZE != 0  and len(hf_power) > GAUSS_WINDOW_SIZE:

                    x2 = np.arange(1, GAUSS_WINDOW_SIZE // 2 + 2)
                    linear_weights = np.concatenate((x2, x2[::-1][1:])) / np.sum(np.concatenate((x2, x2[::-1][1:])))
                    x = np.arange(GAUSS_WINDOW_SIZE)
                    gaussian_weights = np.exp(-((x - GAUSS_WINDOW_SIZE / 2) ** 2) / (2 * (GAUSS_WINDOW_SIZE / 4) ** 2))
                    gaussian_weights /= np.sum(gaussian_weights)  # Normalizacja

                    hf_power_avgG = np.convolve(hf_power, gaussian_weights, mode='same')

                    lf_power_avgG = np.convolve(lf_power, gaussian_weights, mode='same')

                    time_buffer_avg = new_time[len(new_time) - len(hf_power_avgG):]


                    ratio_avg = np.divide(lf_power_avgG, hf_power_avgG, out=np.zeros_like(lf_power_avgG),
                                          where=hf_power_avgG > 0)

                    ratio_raw = np.divide(lf_power, hf_power, out=np.zeros_like(lf_power),
                                          where=hf_power > 0)

                    ratio_avg_raw = np.convolve(ratio_raw, gaussian_weights, mode='same')

                else:
                    time_buffer_avg = new_time
                    hf_power_avg = hf_power
                    lf_power_avg = lf_power
                    ratio_avg = np.divide(lf_power, hf_power, out=np.zeros_like(lf_power), where=hf_power > 0)

                # Buforowanie wyników
                self.hf_power = hf_power

                self.lf_power = lf_power
                self.time_buffer = time_buffer_avg

                self.line_hfG.set_data(time_buffer_avg, hf_power_avgG)
                self.line_lfG.set_data(time_buffer_avg, lf_power_avgG)
                #self.line_lf.set_data(new_time, self.lf_power)


                #self.line_hf.set_data(new_time, self.hf_power)

                self.ax_power.relim()
                self.ax_power.autoscale_view()

                # Aktualizacja wykresu stosunku LF/HF
                self.ratio = ratio_avg
                self.line_ratio.set_data(time_buffer_avg, self.ratio)
                #self.line_ratio_raw.set_data(new_time, ratio_avg_raw)
                self.ax_ratio.relim()
                self.ax_ratio.autoscale_view()

                self.fig.canvas.draw_idle()



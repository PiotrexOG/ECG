import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from EcgData import *
from config import *


class HRPlotter:

    def __init__(self, title: str, ecg_data: EcgData):
        self.ecg_data = ecg_data
        self.PEAKS_TO_PLOT = 160
        self._hr_plot_data = deque(maxlen=self.PEAKS_TO_PLOT)  # Queue for HR data
        self._r_peaks_plot_data = deque(maxlen=self.PEAKS_TO_PLOT)  # Queue for R-peak data

        self.fig, (self.ax_rr, self.ax_hr, self.ax_rr_fil) = plt.subplots(3, 1, figsize=(5, 6))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)

        # R-peak plot as a continuous line using deque data
        self.ax_rr.set_title(title + " - R-peaks")
        self.ax_rr.set_ylabel("uV")
        self.ax_rr.set_facecolor("black")
        self.ax_rr.spines["bottom"].set_color("green")
        self.ax_rr.spines["top"].set_color("green")
        self.ax_rr.spines["right"].set_color("green")
        self.ax_rr.spines["left"].set_color("green")
        self.ax_rr.tick_params(axis="x")
        self.ax_rr.tick_params(axis="y")
        self.ax_rr.set_xlabel("Time (s)")

        x = np.empty(0)
        y = np.empty(0)  # Przykład sygnału (użyj tutaj swoich danych ECG)

        self.ax_r_peaks = self.ax_rr.scatter(
            x, y, color="red", label="R-peaks", marker=".", s=100
        )
        # Line plot for R-peaks using deque data
        (self.line_r_peaks,) = self.ax_rr.plot(x, y, color="green", label="R-peaks")

        # Heart rate plot
        (self.line_hr,) = self.ax_hr.plot([], [], color="blue")
        self.ax_hr.set_title("Heart Rate (HR)")
        self.ax_hr.set_ylabel("HR (bpm)")
        self.ax_hr.set_xlabel("Time (s)")

        self.text_box = self.fig.text(0.87, 0.5, '', fontsize=14, color='white',
                                      bbox=dict(facecolor='black', alpha=0.5))

        self.timer = self.fig.canvas.new_timer(interval=500)
        self.timer.add_callback(self.update_plot)
        self.timer.start()

    def _update_r_peaks_plot_data(self) -> None:
        if len(self.ecg_data.r_peaks) == 0:
            return

        if len(self.ecg_data.r_peaks) < self.PEAKS_TO_PLOT:
            recent_peaks = self.ecg_data.r_peaks[1:]  # Wszystkie elementy oprócz pierwszego
        else:
            recent_peaks = self.ecg_data.r_peaks[-self.PEAKS_TO_PLOT:]

        self._r_peaks_plot_data.clear()
        self._r_peaks_plot_data.extend(recent_peaks)

    def _update_hr_plot_data(self) -> None:
        if len(self.ecg_data.rr_intervals) == 0:
            return

        rr_intervals = np.array(self.ecg_data.rr_intervals[-self.PEAKS_TO_PLOT:])  # Ostatnie 10 wartości RR

        # Aktualizacja kolejki z najnowszymi wartościami
        self._hr_plot_data.clear()
        self._hr_plot_data.extend(rr_intervals)

    def update_plot(self) -> None:
        self._update_r_peaks_plot_data()  # Update R-peak plot data
        self._update_hr_plot_data()  # Update HR plot data

        r_peaks = self._r_peaks_plot_data
        if r_peaks:
            r_peak_times, r_peak_values = zip(*r_peaks)
            r_peak_times_normalized = np.array(r_peak_times) - r_peak_times[0]

            self.line_r_peaks.set_data(r_peak_times_normalized, r_peak_values)
            self.ax_r_peaks.set_offsets(
                np.array([r_peak_times_normalized, r_peak_values]).T
            )

            self.ax_rr.relim()
            self.ax_rr.autoscale_view()
        else:
            self.ax_r_peaks.set_offsets(np.empty((0, 2)))  # Wyczyszczenie wykresu, jeśli brak danych

        r_intervals = self._hr_plot_data

        hr_values = 60 / np.array(r_intervals)  # Calculate HR from RR intervals
        hr_times = np.cumsum(r_intervals)
        if r_intervals:
            hr_times_normalized = np.array(hr_times) - hr_times[0]

            self.line_hr.set_data(hr_times_normalized, hr_values)
            self.ax_hr.relim()
            self.ax_hr.autoscale_view()

            current_hr = hr_values[-1]
            current_interval = r_intervals[-1]
            self.text_box.set_text(
                f"Interval: {current_interval:.2f}s\nHR: {current_hr:.2f}"
            )

        if PRINT_ECG_DATA:
            self.ecg_data.print_data()

        self.fig.canvas.draw_idle()

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from EcgData import *

from config import *


class EcgPlotter:
    @property
    def _is_plot_up_to_date(self) -> bool:
        if (
            0 == len(self._plot_data)
            or self.ecg_data.raw_data[-1][0] > self._plot_data[-1][0]
        ):
            return False
        return True

    def __init__(self, title: str, ecg_data: EcgData):
        self.ecg_data = ecg_data
        self.SECONDS_TO_PLOT = 5
        self._plot_data = deque(maxlen=ecg_data.frequency * self.SECONDS_TO_PLOT)

        self.fig, (self.ax_ecg) = plt.subplots(figsize=(16, 8))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)

        # ECG plot
        (self.line_ecg,) = self.ax_ecg.plot([], [], color="green")
        self.ax_ecg.set_title(title + " - ECG Signal")
        self.ax_ecg.set_ylabel("uV")

        self.ax_ecg.set_facecolor("black")
        self.ax_ecg.spines["bottom"].set_color("green")
        self.ax_ecg.spines["top"].set_color("green")
        self.ax_ecg.spines["right"].set_color("green")
        self.ax_ecg.spines["left"].set_color("green")
        self.ax_ecg.tick_params(
            axis="x",
        )
        self.ax_ecg.tick_params(
            axis="y",
        )
        self.ax_ecg.set_title("ECG Waveform")
        self.ax_ecg.set_xlabel("Time (s)")

        self.ax_r_peaks = self.ax_ecg.scatter(
            [], [], color="red", label="R-peaks", marker="x", s=100
        )

        self.timer = self.fig.canvas.new_timer(interval=500)
        self.timer.add_callback(self.update_plot)

        self.timer.start()

    # def send_single_sample(self, timestamp, voltage):
    #     self._plot_data.append((timestamp, voltage))

    def _update_plot_data(self) -> None:
        if 0 == len(self.ecg_data.raw_data) or self._is_plot_up_to_date:
            return
        else:
            plot_data_count = len(self._plot_data)
            ecg_data_count = len(self.ecg_data.raw_data)

            if 0 == plot_data_count:
                number_of_rows = (
                    plot_data_count
                    if len(self.ecg_data.raw_data) >= plot_data_count
                    else ecg_data_count
                )
                for row in self.ecg_data.raw_data[-number_of_rows:]:
                    self._plot_data.append(row)
                return
            else:
                last_index = np.where(
                    np.all(self.ecg_data.raw_data == self._plot_data[-1], axis=1)
                )[0][0]

                for row in self.ecg_data.raw_data[last_index + 1:]:
                    self._plot_data.append(row)
                return

    def update_plot(self) -> None:
        self._update_plot_data()
        if len(self._plot_data) > 0:
            timestamps, ecg_values = zip(*self._plot_data)
            x = np.array(timestamps)
            x_normalized = x - x[0]  # Normalize time to start from 0

            # Update ECG plot
            self.line_ecg.set_data(x_normalized, ecg_values)  # Use normalized time
            self.ax_ecg.relim()  # Recalculate limits
            self.ax_ecg.autoscale_view()  # Auto scale the view

            r_peaks = self.ecg_data.r_peaks
            # r_peaks = self.ecg_data.r_peaks_piotr
            if r_peaks.any():  # Check if any R-peaks were found
                r_peak_times, r_peak_values = zip(*r_peaks)
                # Normalize the R-peak timestamps
                r_peak_times_normalized = np.array(r_peak_times) - x[0]
                # Update scatter plot with R-peaks
                self.ax_r_peaks.set_offsets(
                    np.array([r_peak_times_normalized, r_peak_values]).T
                )
            else:
                self.ax_r_peaks.set_offsets(np.empty((0, 2)))
        if PRINT_ECG_DATA:
            pass#self.ecg_data.print_data()

        self.fig.canvas.draw_idle()

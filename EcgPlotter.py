import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from matplotlib import ticker

from EcgData import *
from matplotlib.font_manager import FontProperties
from config import *

#KLASA DO WYSIETLANIA WYKRESU EKG

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
        self._plot_data = deque(maxlen=ecg_data.frequency * SECONDS_TO_PLOT)

        self.fig, (self.ax_ecg) = plt.subplots(figsize=(12, 6))
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
        self.ax_ecg.set_xlabel("Time (s)")
        self.fig.gca().xaxis.set_major_locator(ticker.MultipleLocator(3600))  # co 3600 sekund (1 godzina)
        self.fig.gca().xaxis.set_major_formatter(
            ticker.FuncFormatter(
                ticker.FuncFormatter(lambda x, _: f'{int(x // 3600 + 19)%24:02d}:{int((x % 3600) // 60):02d}')))
        #self.ax_ecg.set_ylim([-100, 100])

        self.ax_r_peaks = self.ax_ecg.scatter(
            [], [], color="red", label="R-peaks", marker="x", s=100
        )

        self.ax_loaded_r_peaks = self.ax_ecg.scatter(
            [], [], color="yellow", label="Loaded R-peaks", marker="x", s=100
        )

        self.ax_detected_r_peaks = self.ax_ecg.scatter(
            [], [], color="blue", label="Detected R-peaks", marker="x", s=100
        )

        self.ax_within_spec_r_peaks = self.ax_ecg.scatter(
            [], [], color="white", label="Within spec R-peaks", marker="x", s=100
        )
        
        handles, labels = self.ax_ecg.get_legend_handles_labels()
        
        # Add the legend
        if self.ecg_data.loaded_r_peak_ind.size != np.empty(0).size:
            legend = self.ax_ecg.legend(
                loc="lower right",
                fontsize=12,
                facecolor="black",
                edgecolor="green"
            )
            # Change text color for the legend
            for text in legend.get_texts():
                text.set_color("white")

        
        
        self.stats_text = self.ax_ecg.text(
            0.01,
            0.01,
            "",
            transform=self.ax_ecg.transAxes,
            color="white",
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(facecolor="black", alpha=0.5),
        )

        # self.timer = self.fig.canvas.new_timer(interval=500)
        # self.timer.add_callback(self.update_plot)

        # self.timer.start()
        self.ecg_data.add_listener(self.update_plot)
        self.update_plot()

    # def send_single_sample(self, timestamp, voltage):
    #     self._plot_data.append((timestamp, voltage))

    def _update_plot_data(self) -> None:
        #self._plot_data = self.ecg_data.filtered_data
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
                # Timestamp z ostatniego elementu w _plot_data
                target_timestamp = self._plot_data[-1][0]

                # Szukanie indeksu z tym samym timestampem w ecg_data.filtered_data
                last_index = np.where(
                    np.array([row[0] == target_timestamp for row in self.ecg_data.raw_data])
                )[0][0]

                for row in self.ecg_data.raw_data[last_index + 1 :]:
                    self._plot_data.append(row)
                return

    def update_plot(self) -> None:

        # if np.count_nonzero(self.ecg_data.data_buffer) < 1000:
        #     return
        # self.ecg_data.refresh_if_dirty()
        self._update_plot_data()

        if len(self._plot_data) > 0:
            timestamps, ecg_values = zip(*self._plot_data)
            x = np.array(timestamps)
            x_normalized = x - x[0]  # Normalize time to start from 0
            
            if DISPLAY_X_INDEXES:
                x_ind = np.arange(0, len(x_normalized))
                self.line_ecg.set_data(x_ind, ecg_values)
            else:
                self.line_ecg.set_data(x_normalized, ecg_values)  # Use normalized time
            self.ax_ecg.relim()  # Recalculate limits
            self.ax_ecg.autoscale_view()  # Auto scale the view

            # r_peaks_ind = self.ecg_data.r_peaks_ind
            loaded_r_peaks_ind = self.ecg_data.loaded_r_peak_ind
            detected_r_peaks_ind = self.ecg_data.r_peaks_ind

            if loaded_r_peaks_ind.size != np.empty(0).size:
                r_peaks_ind = np.intersect1d(loaded_r_peaks_ind, detected_r_peaks_ind)
                within_spec_ind = np.intersect1d(self.ecg_data.refined_loaded_peaks_ind, detected_r_peaks_ind)
                # new_arr = np.empty(0)
                # for i in range(within_spec_ind.size):
                #     if within_spec_ind[i] != loaded_r_peaks_ind[i]
                    
                within_spec_ind = np.setdiff1d(within_spec_ind, r_peaks_ind)
                # r_peaks_ind = np.union1d(within_spec_ind, r_peaks_ind)
                loaded_r_peaks_ind = np.setdiff1d(loaded_r_peaks_ind, r_peaks_ind)
                detected_r_peaks_ind = np.setdiff1d(detected_r_peaks_ind, r_peaks_ind)

                r_peaks = self.ecg_data.raw_data[r_peaks_ind]
                loaded_r_peaks = self.ecg_data.raw_data[loaded_r_peaks_ind]
                detected_r_peaks = self.ecg_data.raw_data[detected_r_peaks_ind]
                within_spec_r_peaks = self.ecg_data.raw_data[within_spec_ind]
                for i in range(len(loaded_r_peaks) - 1, -1, -1):
                    if loaded_r_peaks[i, 0] < self._plot_data[0][0]:
                        loaded_r_peaks = loaded_r_peaks[i + 1 :, :]
                        break

                for i in range(len(detected_r_peaks) - 1, -1, -1):
                    if detected_r_peaks[i, 0] < self._plot_data[0][0]:
                        detected_r_peaks = detected_r_peaks[i + 1 :, :]
                        break
                    
                for i in range(len(within_spec_r_peaks) - 1, -1, -1):
                    if within_spec_r_peaks[i, 0] < self._plot_data[0][0]:
                        within_spec_r_peaks = within_spec_r_peaks[i + 1 :, :]
                        break
                    
                if loaded_r_peaks.any():  # Check if any R-peaks were found
                    r_peak_times, r_peak_values = zip(*loaded_r_peaks)
                    # Normalize the R-peak timestamps
                    r_peak_times_normalized = np.array(r_peak_times) - x[0]
                    # Update scatter plot with R-peaks
                    self.ax_loaded_r_peaks.set_offsets(
                        np.array([r_peak_times_normalized, r_peak_values]).T
                    )
                else:
                    self.ax_loaded_r_peaks.set_offsets(np.empty((0, 2)))

                if detected_r_peaks.any():  # Check if any R-peaks were found
                    r_peak_times, r_peak_values = zip(*detected_r_peaks)
                    # Normalize the R-peak timestamps
                    r_peak_times_normalized = np.array(r_peak_times) - x[0]
                    # Update scatter plot with R-peaks
                    self.ax_detected_r_peaks.set_offsets(
                        np.array([r_peak_times_normalized, r_peak_values]).T
                    )
                else:
                    self.ax_detected_r_peaks.set_offsets(np.empty((0, 2)))
                    
                if within_spec_r_peaks.any():  # Check if any R-peaks were found
                    r_peak_times, r_peak_values = zip(*within_spec_r_peaks)
                    # Normalize the R-peak timestamps
                    r_peak_times_normalized = np.array(r_peak_times) - x[0]
                    # Update scatter plot with R-peaks
                    self.ax_within_spec_r_peaks.set_offsets(
                        np.array([r_peak_times_normalized, r_peak_values]).T
                    )
                else:
                    self.ax_within_spec_r_peaks.set_offsets(np.empty((0, 2)))
            else:
                r_peaks = self.ecg_data.r_peaks

            for i in range(len(r_peaks) - 1, -1, -1):
                if r_peaks[i, 0] < self._plot_data[0][0]:
                    r_peaks = r_peaks[i + 1 :, :]
                    break

            if r_peaks.any():  # Check if any R-peaks were found
                r_peak_times, r_peak_values = zip(*r_peaks)
                # Normalize the R-peak timestamps
                r_peak_times_normalized = np.array(r_peak_times) - x[0]
                if DISPLAY_X_INDEXES:
                    r_peak_ind = np.squeeze(np.where(np.isin(x_normalized, r_peak_times_normalized)))
                    r_peak_times_normalized = r_peak_ind
                # Update scatter plot with R-peaks
                self.ax_r_peaks.set_offsets(
                    np.array([r_peak_times_normalized, r_peak_values]).T
                )
            else:
                self.ax_r_peaks.set_offsets(np.empty((0, 2)))

            # Update dynamic text (e.g., current voltage)
            current_voltage = ecg_values[-1]  # Get latest voltage value
            current_time = timestamps[-1]  # Get latest timestamp
            if self.ecg_data.rr_intervals.any():
                current_interval = self.ecg_data.rr_intervals[:,1][-1]
                current_hr = 60 / current_interval
                current_r_peak = self.ecg_data.r_peaks[-1][1]
                # self.text_box.set_text(
                #     f"Time: {current_time:.2f}s\nVoltage: {current_voltage:.2f}uV\nInterval: {current_interval:.2f}s\nR peak volt: {current_r_peak:.2f}uV\nHR: {current_hr:.2f}")
            # self.text_box.set_text(f"Time: {current_time:.2f}s\nVoltage: {current_voltage:.2f}uV\nInterval: {current_interval:.2f}s\nR peak volt: {current_r_peak:.2f}uV")


        if PRINT_ECG_DATA:
            self.stats_text.set_text(self.ecg_data.print_data_string())
            # self.ecg_data.print_data()

        self.fig.canvas.draw_idle()

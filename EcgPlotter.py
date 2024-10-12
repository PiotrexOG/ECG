import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class EcgPlotter:
    def __init__(self, title, sampling_frequency):
        self.SECONDS_TO_PLOT = 5
        self.plot_data = deque(maxlen=sampling_frequency * self.SECONDS_TO_PLOT)
        self.r_peaks = []  # List to store R-peaks timestamps

        # Create subplots: one for ECG, one for moving average signal
        self.fig, (self.ax_ecg) = plt.subplots()

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

        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.update_plot)
        self.timer.start()

        # self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)

    def send_single_sample(self, timestamp, voltage):
        self.plot_data.append((timestamp, voltage))

    def update_plot(self):
        if len(self.plot_data) > 0:
            timestamps, ecg_values = zip(*self.plot_data)
            x = np.array(timestamps)
            x_normalized = x - x[0]  # Normalize time to start from 0

            # Update ECG plot
            self.line_ecg.set_data(x_normalized, ecg_values)  # Use normalized time
            self.ax_ecg.relim()  # Recalculate limits
            self.ax_ecg.autoscale_view()  # Auto scale the view

        self.fig.canvas.draw_idle()
        # return self.line_ecg,  # Return the updated line object for FuncAnimation

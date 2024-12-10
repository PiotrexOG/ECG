import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from matplotlib import ticker

from EcgData import *
from config import *
import datetime

#PLIK DO ANALIZOWANIA TETNA

class HRPlotter:

    def __init__(self, title: str, ecg_data: EcgData):

        self.ecg_data = ecg_data
        self.PEAKS_TO_PLOT = int(SECONDS_TO_PLOT/0.75)
        self._hr_plot_data = deque(maxlen=self.PEAKS_TO_PLOT)  # Queue for HR data
        self._hr_fil_plot_data = deque(maxlen=self.PEAKS_TO_PLOT)  # Queue for HR data
        self._r_peaks_plot_data = deque(maxlen=self.PEAKS_TO_PLOT)  # Queue for R-peak data
        self._r_peaks_fil_plot_data = deque(maxlen=self.PEAKS_TO_PLOT)  # Queue for R-peak data

        self.fig, (self.ax_rr, self.ax_rr_fil, self.ax_hr, self.ax_hr_fil) = plt.subplots(4, 1, figsize=(5, 6))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)

        # R-peak plot as a continuous line using deque data
        self.ax_rr.set_title("Szczyty R")
        self.ax_rr.set_ylabel("Napięcie (uV)")
        self.ax_rr.set_facecolor("black")
        self.ax_rr.spines["bottom"].set_color("green")
        self.ax_rr.spines["top"].set_color("green")
        self.ax_rr.spines["right"].set_color("green")
        self.ax_rr.spines["left"].set_color("green")
        self.ax_rr.tick_params(axis="x")
        self.ax_rr.tick_params(axis="y")
        self.ax_rr.set_xlabel("Czas (s)")

        for ax in [self.ax_rr, self.ax_rr_fil, self.ax_hr, self.ax_hr_fil]:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(3600))  # Znaczniki co 3600 sekund (1 godzina)
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(
                    ticker.FuncFormatter(lambda x, _: f'{int(x // 3600 + 19) % 24:02d}:{int((x % 3600) // 60):02d}')))

        x = np.empty(0)
        y = np.empty(0)  # Przykład sygnału (użyj tutaj swoich danych ECG)

        self.ax_r_peaks = self.ax_rr.scatter(
            x, y, color="red", label="R-peaks", marker=".", s=100
        )


        # Line plot for R-peaks using deque data
        (self.line_r_peaks,) = self.ax_rr.plot(x, y, color="green", label="R-peaks")

        # Heart rate plot
        (self.line_r_peaks_fil,) = self.ax_rr_fil.plot([], [], color="blue")
        (self.line_r_peaks_filCON,) = self.ax_rr_fil.plot([], [], color="red")
        self.ax_rr_fil.set_title("Interwały RR")
        self.ax_rr_fil.set_ylabel("Różnica pomiędzy\n kolejnymi interwałami (s)")
        self.ax_rr_fil.set_xlabel("Czas (s)")

        # Heart rate plot
        (self.line_hr,) = self.ax_hr.plot([], [], alpha = 0.5, color="blue")
        (self.line_hrCON,) = self.ax_hr.plot([], [], color="red")
        self.ax_hr.set_title("Tętno (HR)")
        self.ax_hr.set_ylabel("HR (bpm)")
        self.ax_hr.set_xlabel("Czas (s)")
        self.create_colored_background(self.ax_hr)

        # Heart rate plot
        (self.line_hr_fil,) = self.ax_hr_fil.plot([], [], color="blue")
        self.ax_hr_fil.set_title("Tętno przefiltrowane (HR)")
        self.ax_hr_fil.set_ylabel("HR (n.u.)")
        self.ax_hr_fil.set_xlabel("Czas (s)")

        self.ax_hr_ups = self.ax_hr.scatter(
            x, y, color="red", label="R-peaks", marker=".", s=100
        )

        self.ax_hr_downs = self.ax_hr.scatter(
            x, y, color="green", label="R-peaks", marker=".", s=100
        )

        self.text_box = self.fig.text(0.87, 0.5, '', fontsize=14, color='white',
                                      bbox=dict(facecolor='black', alpha=0.5))



        self.stats_text = self.ax_hr.text(
            0.01,
            0.01,
            "",
            transform=self.ax_hr.transAxes,
            color="white",
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(facecolor="black", alpha=0.5),
        )
        # t_seconds = np.linspace(0, 21600, 1000)  # Symulowany czas w sekundach (0 - 86400)
        # # Ustalanie czasu początkowego (t = 0 oznacza 19:36:00)
        # start_time = datetime.datetime(2024, 12, 6, 19, 36, 0)
        #
        # # Tworzenie osi czasu w formacie daty i godziny
        # time_labels = [start_time + datetime.timedelta(seconds=int(t)) for t in t_seconds]
        #
        # # Konfigurowanie osi X - pełne godziny
        # # Znajdujemy najbliższą pełną godzinę >= start_time
        # first_full_hour = (start_time + datetime.timedelta(minutes=60 - start_time.minute)).replace(minute=0, second=0,
        #                                                                                             microsecond=0)
        #
        # # Generujemy pełne godziny od 20:00 (lub najbliższej pełnej godziny) do 19:00 następnego dnia
        # hour_ticks = [first_full_hour + datetime.timedelta(hours=i) for i in range(-1, 24)]
        # hour_labels = [tick.strftime('%H:%M') for tick in hour_ticks]
        # # Ustawianie etykiet na osi X tylko dla pełnych godzin
        # plt.xticks(hour_ticks, hour_labels, rotation=45)

        # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(3600))  # co 3600 sekund (1 godzina)
        # plt.gca().xaxis.set_major_formatter(
        #     ticker.FuncFormatter(lambda x,: f'{int(x // 3600):02d}:{int((x % 3600) // 60):02d}'))


        interv = 1500
        # if(APP_MODE.LOAD_CSV):
        #     interv = 50000
        self.timer = self.fig.canvas.new_timer(interval=1000)
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

       # self.ecg_data.add_listener(self.update_plot)

    def _update_r_peaks_plot_data(self) -> None:
        if len(self.ecg_data.r_peaks) == 0:
            return

        if len(self.ecg_data.r_peaks) <= self.PEAKS_TO_PLOT:
            recent_peaks = self.ecg_data.r_peaks[1:]  # Wszystkie elementy oprócz pierwszego
        else:
            recent_peaks = self.ecg_data.r_peaks[-self.PEAKS_TO_PLOT:]

        self._r_peaks_plot_data.clear()
        self._r_peaks_plot_data.extend(recent_peaks)

    def _update_r_peaks_plot_data1(self) -> None:
        if len(self.ecg_data.rr_intervals) == 0:
            return

        recent_peaks = self.ecg_data.rr_intervals[-self.PEAKS_TO_PLOT:]

        self._r_peaks_fil_plot_data.clear()
        self._r_peaks_fil_plot_data.extend(recent_peaks)

    def _update_hr_plot_data(self) -> None:
        if len(self.ecg_data.hr) == 0:
            return

        hrs = np.array(self.ecg_data.hr[-self.PEAKS_TO_PLOT:])  # Ostatnie 10 wartości RR

        # Aktualizacja kolejki z najnowszymi wartościami
        self._hr_plot_data.clear()
        self._hr_plot_data.extend(hrs)

    def _update_hr_fil_plot_data(self) -> None:
        if len(self.ecg_data.hr_filtered) == 0:
            return

        hrs = np.array(self.ecg_data.hr_filtered[-self.PEAKS_TO_PLOT:])  # Ostatnie 10 wartości RR

        # Aktualizacja kolejki z najnowszymi wartościami
        self._hr_fil_plot_data.clear()
        self._hr_fil_plot_data.extend(hrs)

    def update_plot(self) -> None:
        self._update_r_peaks_plot_data()  # Update R-peak plot data
        self._update_hr_plot_data()  # Update HR plot data
        self._update_hr_fil_plot_data()  # Update HR plot data
        self._update_r_peaks_plot_data1()

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

        rr_intervals = self._r_peaks_fil_plot_data
        if rr_intervals:
            r_peak_times, r_peak_values = zip(*rr_intervals)
            r_peak_times_normalized = np.array(r_peak_times) - r_peak_times[0]

            # Uśrednianie interwałów RR (analogicznie do uśredniania mocy HF/LF)
            if GAUSS_WINDOW_SIZEHR != 0  and len(r_peak_values) > GAUSS_WINDOW_SIZEHR:
                rr_intervals_avg = np.convolve(r_peak_values, np.ones(GAUSS_WINDOW_SIZEHR) / GAUSS_WINDOW_SIZEHR, mode='valid')
                avg_times = r_peak_times_normalized[len(r_peak_times_normalized) - len(rr_intervals_avg):]
                self.line_r_peaks_filCON.set_data(avg_times, rr_intervals_avg)


            self.line_r_peaks_fil.set_data(r_peak_times_normalized, r_peak_values)

            self.ax_rr_fil.relim()
            self.ax_rr_fil.autoscale_view()


        hrs = self._hr_plot_data

        if hrs:
            r_peak_times, r_peak_values = zip(*hrs)
            x = np.array(r_peak_times)
            r_peak_times_normalized = np.array(r_peak_times) - r_peak_times[0]

            # Uśrednianie interwałów RR (analogicznie do uśredniania mocy HF/LF)
            if GAUSS_WINDOW_SIZEHR != 0 and len(r_peak_values) > GAUSS_WINDOW_SIZEHR:

                x = np.arange(GAUSS_WINDOW_SIZEHR)
                gaussian_weights = np.exp(-((x - GAUSS_WINDOW_SIZEHR / 2) ** 2) / (2 * (GAUSS_WINDOW_SIZEHR / 4) ** 2))
                gaussian_weights /= np.sum(gaussian_weights)  # Normalizacja

                rr_intervals_avg = np.convolve(r_peak_values, gaussian_weights, mode='same')
                avg_times = r_peak_times_normalized[len(r_peak_times_normalized) - len(rr_intervals_avg):]
                self.line_hrCON.set_data(avg_times, rr_intervals_avg)

            self.line_hr.set_data(r_peak_times_normalized, r_peak_values)


            ups = self.ecg_data.hr_ups
            if ups.any():  # Check if any R-peaks were found
                t, v = zip(*ups)
                # Normalize the R-peak timestamps
                tn = np.array(t) - x[0]
                # Update scatter plot with R-peaks
                self.ax_hr_ups.set_offsets(
                    np.array([tn, v]).T
                )
            else:
                self.ax_hr_ups.set_offsets(np.empty((0, 2)))

            downs = self.ecg_data.hr_downs

            if downs.any():  # Check if any R-peaks were found
                t, v = zip(*downs)
                # Normalize the R-peak timestamps
                tn = np.array(t) - x[0]
                # Update scatter plot with R-peaks
                self.ax_hr_downs.set_offsets(
                    np.array([tn, v]).T
                )
            else:
                self.ax_hr_downs.set_offsets(np.empty((0, 2)))

            self.ax_hr.relim()
            self.ax_hr.autoscale_view()

        hrs_fil = self._hr_fil_plot_data
        if hrs_fil:
            t, r = zip(*hrs_fil)
            r_peak_times, r_peak_values = zip(*hrs_fil)
        #    x = t[16] - t[0]
            r_peak_times_normalized = np.array(r_peak_times) - t[0]

            # Dodanie początkowych pustych danych do czasu 14 sekund

            self.line_hr_fil.set_data(r_peak_times_normalized, r_peak_values)

            self.ax_hr_fil.relim()
            self.ax_hr_fil.autoscale_view()


        # hr_values = 60 / np.array(r_intervals)  # Calculate HR from RR intervals
        # hr_times = np.cumsum(r_intervals)
        # if r_intervals:
        #     hr_times_normalized = np.array(hr_times) - hr_times[0]
        #
        #     self.line_hr.set_data(hr_times_normalized, hr_values)
        #     self.ax_hr.relim()
        #     self.ax_hr.autoscale_view()
        #
        #     current_hr = hr_values[-1]
        #     current_interval = r_intervals[-1]
        #     self.text_box.set_text(
        #         f"Interval: {current_interval:.2f}s\nHR: {current_hr:.2f}"
        #     )

        if PRINT_ECG_DATA:
            self.stats_text.set_text(self.ecg_data.print_data_string())
            print(self.ecg_data.print_data_string())

        self.fig.canvas.draw_idle()

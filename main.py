import socket
import struct
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import threading
import queue
import datetime

import numpy as np
from scipy.signal import lfilter, find_peaks

class PanTompkins:
    def __init__(self, fs):
        self.fs = fs  # Sampling frequency (e.g., 130 Hz)

    def bandpass_filter(self, signal):
        # Simple bandpass filter between 5-15 Hz for QRS detection
        b = np.array([1, -2, 1])
        a = np.array([1, -1.8, 0.81])
        filtered_signal = lfilter(b, a, signal)
        return filtered_signal

    def derivative(self, signal):
        # Derivative emphasizes slope of QRS complex
        b = np.array([1, 2, 0, -2, -1]) * (self.fs / 8.0)
        derivative_signal = lfilter(b, 1, signal)
        return derivative_signal

    def squaring(self, signal):
        # Square each point in the signal
        squared_signal = signal ** 2
        return squared_signal

    def moving_average(self, signal):
        # Moving average window size (approx 150ms window)
        window_size = int(0.15 * self.fs)
        ma_signal = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
        return ma_signal

    def detect_r_peaks(self, ecg_signal):
        # Apply the Pan-Tompkins algorithm step by step
        filtered_signal = self.bandpass_filter(ecg_signal)
        derivative_signal = self.derivative(filtered_signal)
        squared_signal = self.squaring(derivative_signal)
        ma_signal = self.moving_average(squared_signal)

        # Detect peaks in the moving average signal
        peaks, _ = find_peaks(ma_signal, distance=int(0.6 * self.fs))  # Peaks with minimum distance of 600 ms
        return peaks


class EcgRPeakDetector:
    def __init__(self):
        self.r_peaks = []  # Lista do przechowywania timestampów i wartości R-peaków
        self.threshold = 700  # Próg dla wartości sygnału (w mV)
        self.last_peak_time = -1  # Czas ostatniego wykrytego załamka R (inicjalnie brak)

    def detect_r_peaks(self, timestamps, ecg_values):
        # Przechodzimy po parach timestamp-value
        for index, (timestamp, value) in enumerate(zip(timestamps, ecg_values)):
            # Sprawdzamy, czy dane są nowsze od ostatnio wykrytego R-peaku
            if timestamp > self.last_peak_time:
                # Sprawdź, czy wartość sygnału przekracza próg
                if value > self.threshold:
                    # Jeżeli to pierwsza wartość, ustaw ją jako początkową
                    if index > 2:
                        # Oblicz pochodną (różnica wartości / różnica czasów)
                        derivative = (value - ecg_values[index-1])
                        prev_derivate = ecg_values[index-1] - ecg_values[index-2]
                        # Sprawdź, czy pochodna zmienia znak lub jest równa zero (szczyt załamka R)
                        if derivative <= 0 and prev_derivate >= 0:
                            # Zidentyfikowano szczyt, zapisujemy ten punkt jako załamek R
                            self.r_peaks.append(
                                (timestamps[index-1], ecg_values[index-1]))  # Dodaj timestamp i wartość załamka R
                            self.last_peak_time = timestamps[index-1]  # Zaktualizuj czas ostatniego załamka R

                            # Po wykryciu załamka przechodzimy do kolejnej iteracji



    def get_r_peaks(self):
        return self.r_peaks

class EcgPlotter:
    def __init__(self, title, sampling_frequency):
        self.SECONDS_TO_PLOT = 5
        #self.plot_data = deque(maxlen=130 * self.SECONDS_TO_PLOT)  # Assuming 130 Hz max frequency
        self.plot_data = []
        self.detector = EcgRPeakDetector()  # Tworzenie instancji klasy wykrywającej R-peaks
        self.r_peaks = []  # List to store R-peaks timestamps
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.scatter_r_peaks = None  # To store scatter points for R-peaks
        plt.title(title)
        plt.xlabel('Time (ms)')
        plt.ylabel('mV')
        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.update_plot)
        self.timer.start()

        # Initialize the Pan-Tompkins algorithm
        self.pan_tompkins = PanTompkins(sampling_frequency)

    def send_single_sample(self, timestamp, mV):
        #if timestamp < 599616068803.2223:
        if timestamp < 599616073201739904:
            return

        self.plot_data.append((timestamp, mV))
        timestamps, ecg_values = zip(*self.plot_data)
        self.detector.detect_r_peaks(timestamps, ecg_values)  # Wykrywanie R-peaks
        #if len(self.plot_data) >= 130:  # Process when we have enough data
            #self.detect_r_peak()
            #self.detect_r_peak_simple()

    def detect_r_peak(self):
        timestamps, ecg_values = zip(*self.plot_data)
        ecg_array = np.array(ecg_values)

        # Use Pan-Tompkins to detect R-peaks
        r_peak_indices = self.pan_tompkins.detect_r_peaks(ecg_array)

        for peak_index in r_peak_indices:
            peak_timestamp = timestamps[peak_index]
            if len(self.r_peaks) == 0 or (peak_timestamp - self.r_peaks[-1]) > 300:
                self.r_peaks.append(peak_timestamp)
                print(f"R-peak detected at {peak_timestamp} ms")

                # Calculate R-R intervals and HRV
                self.calculate_rr_intervals()

    def detect_r_peak_simple(self):
        timestamps, ecg_values = zip(*self.plot_data)
        ecg_array = np.array(ecg_values)
        window_size = int(0.8 * self.pan_tompkins.fs)  # Zakładamy okno cyklu serca ok. 800 ms

        # Lista do przechowywania miejsc, gdzie będą pionowe linie rozdzielające cykle
        cycle_separators = []

        r_peaks_simple = []
        for i in range(0, len(ecg_array), window_size):
            segment = ecg_array[i:i + window_size]
            if len(segment) > 0:
                # Znajdź maksimum w tym segmencie (potencjalny R-peak)
                peak_idx = np.argmax(segment)
                peak_timestamp = timestamps[i + peak_idx]

                # Sprawdź, czy R-peak nie jest zbyt blisko poprzedniego
                if len(self.r_peaks) == 0 or (peak_timestamp - self.r_peaks[-1]) > 300:
                    self.r_peaks.append(peak_timestamp)
                    r_peaks_simple.append(peak_timestamp)
                    print(f"R-peak (simple) detected at {peak_timestamp} ms")
                    cycle_separators.append(peak_timestamp)

                    # Obliczamy R-R interwały tylko wtedy, gdy wykryjemy nowy R-peak
                    self.calculate_rr_intervals()

        # Rysujemy linie oddzielające cykle serca
        self.plot_cycle_separators(cycle_separators)

    def detect_r_peaks_own(self, timestamp, value):
        # Sprawdź, czy wartość sygnału przekracza próg
        if value > self.threshold:
            # Jeżeli to pierwsza wartość, ustaw ją jako początkową
            if self.previous_value is not None and self.previous_time is not None:
                # Oblicz pochodną (różnica wartości / różnica czasów)
                derivative = (value - self.previous_value) / (timestamp - self.previous_time)

                # Sprawdź, czy pochodna zmienia znak lub jest równa zero (szczyt załamka R)
                if derivative <= 0:
                    # Zidentyfikowano potencjalny szczyt - wybierz maksymalną wartość z kandydatów
                    if len(self.candidate_peaks) > 0:
                        max_peak = max(self.candidate_peaks, key=lambda x: x[1])  # Znajdź maksimum
                        self.r_peaks.append(max_peak)  # Dodaj timestamp i wartość załamka R
                        self.candidate_peaks.clear()  # Wyczyść kandydatów po detekcji

            # Zapisz obecnego kandydata na załamek R
            self.candidate_peaks.append((timestamp, value))

        # Aktualizuj poprzednie wartości
        self.previous_value = value
        self.previous_time = timestamp

    def get_r_peaks(self):
        return self.r_peaks

    def plot_cycle_separators(self, cycle_separators):
        # Dodaj linie na wykresie w miejscach cykli serca (skalujemy z nanosekund na milisekundy)
        for separator in cycle_separators:
            self.ax.axvline(x=separator / 1e6 - self.plot_data[0][0] / 1e6, color='green', linestyle='--')
        self.fig.canvas.draw_idle()

    def calculate_rr_intervals(self):
        if len(self.r_peaks) > 1:
            rr_intervals = np.diff(self.r_peaks)  # Calculate differences between consecutive R-peaks
            print(f"R-R Intervals: {rr_intervals}")

            # Calculate basic HRV metrics like SDNN (Standard deviation of RR intervals)
            sdnn = np.std(rr_intervals)
            print(f"SDNN: {sdnn} ms")

    def update_plot(self):
        if len(self.plot_data) > 0:
            timestamps, values = zip(*self.plot_data)
            # Konwertuj timestampy z nanosekund na milisekundy (lub sekundy)

            # Rysowanie sygnału EKG
            timestamps_ms = np.array(timestamps) / 1e6  # Konwersja na milisekundy
            self.line.set_data(timestamps_ms - timestamps_ms[0], values)

            # Rysowanie R-peaks
            if self.scatter_r_peaks:
                self.scatter_r_peaks.remove()  # Usuwanie poprzedniego scattera
            self.r_peaks = self.detector.get_r_peaks()
            if len(self.r_peaks) > 0:
                r_peak_times, r_peak_values = zip(*self.r_peaks)
                r_peak_times_ms = np.array(r_peak_times) / 1e6  # Konwersja na milisekundy
                self.scatter_r_peaks = self.ax.scatter(
                    r_peak_times_ms - timestamps_ms[0], r_peak_values, color='red'
                )

            # Aktualizacja osi i wykresu
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw_idle()  # Bezpieczna aktualizacja wykresu




HOST = '127.0.0.1'
PORT = 12345
ecg_plotter = EcgPlotter("ECG", 130)
data_queue = queue.Queue()


def handle_client_connection(client_socket):
    try:
        while True:
            raw_data = receive_packet(client_socket)
            process_packet(raw_data)
    except (ConnectionResetError, ValueError):
        print("Client disconnected.")
    finally:
        client_socket.close()


def receive_packet(client_socket):
    raw_data = client_socket.recv(73 * (4 + 8))
    if len(raw_data) < 73 * (4 + 8):
        raise ValueError("Brak danych lub błąd połączenia.")
    return raw_data


def process_packet(raw_data):
    siema = 0
    for i in range(73):
        offset = i * (4 + 8)
        float_value = struct.unpack('!f', raw_data[offset:offset + 4])[0]  # Wyciągnięcie floata
        long_value = struct.unpack('!q', raw_data[offset + 4:offset + 12])[0]  # Wyciągnięcie long
       # if siema == 0:
           # print("First value:" + "%f" % float_value)
            # date = datetime.datetime.fromtimestamp((long_value / 1e9))
           # print("First value timestamp:" + str(long_value))
        siema = 1
        data_queue.put((long_value, -float_value))  # Dodanie wartości do kolejki


def plot_data():
    while True:
        try:
            timestamp, mV = data_queue.get(timeout=1)
            ecg_plotter.send_single_sample(timestamp, mV)
        except queue.Empty:
            continue


def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Bind the socket to the address and port
        server_socket.bind((HOST, PORT))

        # Start listening for incoming connections
        server_socket.listen(1)
        print("Server listening on port", HOST, PORT)

        plot_thread = threading.Thread(target=plot_data)
        plot_thread.daemon = True
        plot_thread.start()

        while True:
            # Accept incoming connections
            client_socket, _ = server_socket.accept()
            print("Client connected.")

            # Handle the client connection
            client_thread = threading.Thread(target=handle_client_connection, args=(client_socket,))
            client_thread.daemon = True
            client_thread.start()


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    plt.show()  # Upewnij się, że główny wątek pozostaje w pętli wyświetlania wykresu

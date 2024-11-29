import numpy as np

import matplotlib.pyplot as plt

import FFT
import Wave
import PanTompkins
import NAME_THIS_MODULE_YOURSELF_PIOTER
from PanTompkins import derivative_filter
from config import *
from dataSenderEmulator import read_csv
import scipy.signal as signal
from Finders.RPeaksFinder import RPeaksFinder
import threading
from scipy.interpolate import interp1d
from filterpy.kalman import KalmanFilter
from scipy.signal import butter, filtfilt, detrend
from Finders.PanTompkinsFinder import PanTompkinsFinder

from functools import partial
from wfdb import processing
import wfdb
from tqdm import tqdm


def setup_kalman_filter():
    kf = KalmanFilter(dim_x=2, dim_z=1)

    # Model dynamiki sygnału
    kf.F = np.array([[1, 1], [0, 1]])  # Macierz przejścia
    kf.H = np.array([[1, 0]])  # Macierz obserwacji

    # Szumy
    kf.Q = np.array([[0.0001, 0], [0, 0.01]])  # Szum procesowy
    kf.R = np.array([[20]])  # Szum pomiarowy (dopasuj dla szumu)
    kf.P = np.eye(2) * 0.1  # Początkowa macierz kowariancji
    kf.x = np.array([[0], [0]])  # Początkowy stan
    return kf


# Filtrowanie sygnału
def apply_kalman_filter(signal):
    kf = setup_kalman_filter()
    filtered_signal = []

    for z in signal:
        kf.predict()
        kf.update([z])
        filtered_signal.append(kf.x[0, 0])

    return np.array(filtered_signal)


class EcgData:

    @property
    def raw_data(self):
        return self.__raw_data

    @raw_data.setter
    def raw_data(self, raw_data):
        self.__lock.acquire()
        self.__raw_data = raw_data
        self.__set_dirty()
        self.__lock.release()
        self.refresh_if_dirty()

    @property
    def r_peaks(self):
        # self.__refresh_if_dirty()
        return self.__r_peaks

    @property
    def r_peaks_ind(self):
        return self.__r_peaks_ind

    @property
    def loaded_r_peaks(self):
        return self.__raw_data[self.__loaded_r_peaks_ind]

    @property
    def loaded_r_peak_ind(self):
        return self.__loaded_r_peaks_ind

    @property
    def hr_ups(self):
        # self.__refresh_if_dirty()
        return self.__hr_ups

    @property
    def hr_downs(self):
        # self.__refresh_if_dirty()
        return self.__hr_downs

    @property
    def hr(self):
        # self.__refresh_if_dirty()
        return self.__hr

    @property
    def inhalation_starts_moments(self):
        # self.__refresh_if_dirty()
        return self.__inhalation_starts_moments

    @property
    def exhalation_starts_moments(self):
        # self.__refresh_if_dirty()
        return self.__exhalation_starts_moments

    @property
    def r_peaks_filtered(self):
        # self.__refresh_if_dirty()
        return self.__r_peaks_filtered

    @property
    def r_peaks_piotr(self):
        # self.__refresh_if_dirty()
        return self.__r_peaks_piotr

    @property
    def rr_intervals(self):
        # self.__refresh_if_dirty()
        return self.__rr_intervals

    @property
    def mean_rr(self):
        # self.__refresh_if_dirty()
        return self.__mean_rr

    @property
    def sdnn(self):
        # self.__refresh_if_dirty()
        return self.__sdnn

    @property
    def rmssd(self):
        # self.__refresh_if_dirty()
        return self.__rmssd

    @property
    def pnn50(self):
        # self.__refresh_if_dirty()
        return self.__pnn50

    def __init__(
        self, data_frequency, r_peaks_finder: RPeaksFinder, target_frequency: int = -1
    ):
        self.frequency = data_frequency
        self.__r_peaks_finder = r_peaks_finder
        self.__raw_data = np.empty((0, 2))
        self.__lock = threading.Lock()
        self.loaded___raw_data = np.empty(0)
        self.filtered_data = np.empty((0, 2))
        self.__r_peaks = np.empty(0)
        self.__r_peaks_ind = np.empty(0)
        self.__loaded_r_peaks_ind = np.empty(0)
        self.__hr_ups = np.empty(0)
        self.__hr_downs = np.empty(0)
        self.__r_peaks_filtered = np.empty(0)
        self.__r_peaks_piotr = np.empty(0)
        self.__rr_intervals = np.empty((0, 2))
        self.__hr = np.empty((0, 2))
        self.hr_filtered = np.empty((0, 2))

        self.__callbacks = []

        self.__mean_rr = -1
        self.__sdnn = -1
        self.__rmssd = -1
        self.__pnn50 = -1
        self.__is_dirty = False

        if target_frequency < 0:
            self.target_frequency = None
        else:
            self.target_frequency = target_frequency

        self.lowcut = 5
        self.highcut = 18
        self.buffer_size = 4000
        self.data_buffer = np.zeros(
            self.buffer_size
        )  # Inicjalizacja bufora o stałej wielkości
        self.b, self.a = self.create_bandpass_filter(self.lowcut, self.highcut, 4)
        self.b1, self.a1 = self.create_bandpass_filter(0.5, 40, 4)
        self.zi = signal.lfilter_zi(self.b, self.a)  # Stan filtra dla lfilter

        self.lowcutHR = LOW
        self.highcutHR = HIGH
        self.buffer_sizeHR = 200
        self.data_bufferHR = np.zeros(
            self.buffer_sizeHR
        )  # Inicjalizacja bufora o stałej wielkości
        self.bHR, self.aHR = self.create_bandpass_filter1(
            self.lowcutHR, self.highcutHR, 4
        )
        self.ziHR = signal.lfilter_zi(self.bHR, self.aHR)  # Stan filtra dla lfilter

        self.__inhalation_starts_moments = np.empty((0, 2))
        self.__exhalation_starts_moments = np.empty((0, 2))

        self.wdechy = []  # Lista par (początek wdechu, koniec wdechu)
        self.wydechy = []  # Lista par (początek wydechu, koniec wydechu)

        # self.bHR, self.aHR = signal.butter(2, [0.1, 40], btype='bandpass', fs=100)  # Przykładowy filtr
        # self.ziHR = signal.lfilter_zi(self.bHR, self.aHR) * 0  # Inicjalizacja współczynników stanu filtra

        return

    def add_listener(self, callback):
        if callable(callback):
            self.__callbacks.append(callback)

    def remove_listener(self, callback):
        if callback in self._callbacks:
            self.__callbacks.remove(callback)

    def on_data_updated(self):
        for callback in self.__callbacks:
            callback()  # Wywołuje update_plot, który obsługuje przekierowanie do głównego wątku

    @staticmethod
    def highpass_filter(data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        cutoff_normalized = cutoff / nyquist
        b, a = butter(order, cutoff_normalized, btype="high")
        y = filtfilt(b, a, data)
        return y

    def update_and_filter(self, new_data):
        # Aktualizacja bufora danych
        self.data_buffer = np.concatenate((self.data_buffer[len(new_data) :], new_data))

        filtered_signal, self.zi = signal.lfilter(
            self.b, self.a, self.data_buffer, zi=self.zi * self.data_buffer[0]
        )
        return filtered_signal[-len(new_data) :], self.zi

    def update_hr(self, new_data):
        # Aktualizacja bufora poprzez usunięcie najstarszych elementów, jeśli bufor osiągnął maksymalny rozmiar
        if len(self.data_bufferHR) >= self.buffer_sizeHR:
            self.data_bufferHR = np.roll(self.data_bufferHR, -1)
            self.data_bufferHR[-1] = new_data

    def filter_hr(self):
        # Przefiltrowanie sygnału
        # filtered_signal, self.ziHR = signal.lfilter(self.bHR, self.aHR, self.data_bufferHR, zi=self.ziHR)
        filtered_signal = signal.filtfilt(self.bHR, self.aHR, self.__hr[:, 1])
        # return filtered_signal[-len(new_data):]  # Zwróć tylko najnowsze przefiltrowane wartości
        peaks_with_timestamps = np.column_stack((self.__hr[:, 0], filtered_signal))
        return peaks_with_timestamps  # Zwróć tylko najnowsze przefiltrowane wartości

    def derivative_filter(self, sig):
        return np.diff(sig, prepend=0)

    def square1(self, sig):
        return sig**2

    def moving_window_integration(self, sig, window_size):
        return np.convolve(sig, np.ones(window_size) / window_size, mode="same")

    def update_and_filter(self, new_data):
        # Aktualizacja bufora danych
        self.data_buffer = np.concatenate((self.data_buffer[len(new_data) :], new_data))

        filtered_signal, self.zi = signal.lfilter(
            self.b, self.a, self.data_buffer, zi=self.zi * self.data_buffer[0]
        )
        return filtered_signal[-len(new_data) :], self.zi

    def update_hr(self, new_data):
        # Aktualizacja bufora poprzez usunięcie najstarszych elementów, jeśli bufor osiągnął maksymalny rozmiar
        if len(self.data_bufferHR) >= self.buffer_sizeHR:
            self.data_bufferHR = np.roll(self.data_bufferHR, -1)
            self.data_bufferHR[-1] = new_data

    def filter_hr(self):
        # Przefiltrowanie sygnału
        # filtered_signal, self.ziHR = signal.lfilter(self.bHR, self.aHR, self.data_bufferHR, zi=self.ziHR)
        filtered_signal = signal.filtfilt(self.bHR, self.aHR, self.__hr[:, 1])
        # return filtered_signal[-len(new_data):]  # Zwróć tylko najnowsze przefiltrowane wartości
        peaks_with_timestamps = np.column_stack((self.__hr[:, 0], filtered_signal))
        return peaks_with_timestamps  # Zwróć tylko najnowsze przefiltrowane wartości

    def derivative_filter(self, sig):
        return np.diff(sig, prepend=0)

    def square1(self, sig):
        return sig**2

    def moving_window_integration(self, sig, window_size):
        return np.convolve(sig, np.ones(window_size) / window_size, mode="same")

    def load_csv_data(self, path):
        csv_data = np.array(read_csv(path))
        if NORMALIZED_TEST_DATA_TIME:
            csv_data[:, 0] -= csv_data[0, 0]
        if NEGATE_INCOMING_DATA:
            csv_data[:, 1] *= -1
        csv_data[:, 0] /= TIME_SCALE_FACTOR

        # filtered_signal = PanTompkins.bandpass_filter(csv_data[:, 1], self.frequency, 0.5, 40)
        # baseline_corrected_signal = EcgData.highpass_filter(filtered_signal, cutoff=0.5, fs=self.frequency)
        # csv_data = np.column_stack((csv_data[:, 0], baseline_corrected_signal))

        baseline_corrected_signal = EcgData.highpass_filter(
            csv_data[:, 1], cutoff=0.5, fs=self.frequency
        )
        filtered_signal = PanTompkins.bandpass_filter(
            baseline_corrected_signal, self.frequency, 0.5, 40
        )
        csv_data = np.column_stack((csv_data[:, 0], filtered_signal))

        if self.target_frequency is not None:
            interpolated_data = EcgData.interpolate_data(
                csv_data, self.target_frequency
            )
            self.frequency = self.target_frequency
            self.__raw_data = interpolated_data
        else:
            self.__raw_data = csv_data
            self.frequency = SAMPLING_RATE

        self.frequency = SAMPLING_RATE
        # self.__raw_data = csv_data
        # self.__is_dirty = True
        self.__refresh_data()

        # if self.__r_peaks_finder.__class__.__name__ == PanTompkinsFinder.__name__:
        #     np.save(f"{path}.rpeaks", self.__r_peaks_ind)
        # else:
        #     self.__loaded_r_peaks_ind = np.load(f"{path}.rpeaks.npy")

        # self.on_data_updated()

        #  filtered_signal = signal.filtfilt(self.b, self.a, csv_data[:, 1])

        # vals = signal.filtfilt(self.b1, self.a1, csv_data[:, 1])

        # # Zapisanie przefiltrowanych danych do raw_data
        # self.raw_data = np.column_stack((csv_data[:, 0], vals))

        # #self.raw_data = self.detect_and_remove_artifacts(self.raw_data)

        # print(len(self.raw_data))

        # self.filtered_data = np.column_stack((csv_data[:, 0], filtered_signal))
        # print(len(self.filtered_data))

        # self.__set_dirty()
        # self.__refresh_if_dirty()

        try:
            self.hr_filtered = self.filter_hr()
        except:
            pass
        # FFT.fft(self.__rr_intervals[:, 1])

        # Wave.analyze(self.__rr_intervals)
        # self.print_data()

        self.on_data_updated()

    def detect_and_remove_artifacts(
        self, signal, sampling_rate=130, window_size=5, threshold_multiplier=2
    ):

        times = signal[:, 0]
        values = signal[:, 1]

        # Długość okna w próbkach
        window_samples = int(window_size * sampling_rate)
        # Obliczenie średniej wartości sygnału
        global_mean = np.mean(np.abs(values))

        # Przechowywanie przetworzonego sygnału
        cleaned_signal = values.copy()

        # Iteracja przez okna
        for i in range(0, len(values), window_samples):
            window = values[i : i + window_samples]
            window_mean = np.mean(np.abs(window))

            # Detekcja artefaktów
            if window_mean > threshold_multiplier * global_mean:
                cleaned_signal[i : i + window_samples] = 0  # Lub np.nan

        return np.column_stack((times, cleaned_signal))

    @staticmethod
    def interpolate_data(data, target_frequency):
        timestamps = data[:, 0]
        values = data[:, 1]
        # value = apply_kalman_filter(values)

        start_time = timestamps[0]  # Początkowy timestamp
        end_time = timestamps[-1]  # Końcowy timestamp

        # Nowa siatka czasowa z 400 Hz
        new_timestamps = np.arange(0, end_time - start_time, 1 / target_frequency)
        new_timestamps = new_timestamps + start_time
        # Interpolacja
        interpolator = interp1d(
            timestamps, values, kind="linear"
        )  # , fill_value="extrapolate")
        # Można zmienić na 'cubic'
        new_values = interpolator(new_timestamps)
        # new_values = apply_kalman_filter(new_values)
        # new_values = PanTompkins.bandpass_filter(new_values, 130)
        interpolated_data = np.column_stack((new_timestamps, new_values))
        return interpolated_data

    def load_data_from_mitbih(self, path, record_num=0):
        record = wfdb.rdrecord(path)
        annotation = wfdb.rdann(path, "atr")
        ecg_signal = record.p_signal[:, record_num]
        sample_rate = record.fs
        channel_names = record.sig_name
        r_peak_locations = annotation.sample

        timestamps = np.array([i / sample_rate for i in range(len(ecg_signal))])

        self.frequency = sample_rate
        self.__loaded_r_peaks_ind = r_peak_locations
        self.raw_data = np.column_stack((timestamps, ecg_signal))
        self.__refresh_data()
        self.on_data_updated()

    def load_data_from_qt(self, path, record_num=0):
        record = wfdb.rdrecord(path)
        sample_rate = record.fs
        self.frequency = sample_rate
        ecg_signal = record.p_signal[:, record_num]
        # header = wfdb.rdann(path, "hea")
        # ann = wfdb.rdann(path, 'atr')
        # self.__loaded_r_peaks_ind = ann.sample
        annotation_q1c = wfdb.rdann(path, "q1c")
        annotation_qt1 = wfdb.rdann(path, "qt1")
        annotation_pu = wfdb.rdann(path, "pu")
        annotation_pu0 = wfdb.rdann(path, "pu0")

        self.test = annotation_pu.sample

        p_indexes = np.where(np.array(annotation_pu.symbol) == "p")[0]
        self.p = annotation_pu.sample[p_indexes]
        q_indexes = np.where(np.array(annotation_pu.symbol) == "N")[0]
        self.q = annotation_pu.sample[q_indexes]
        self.__loaded_r_peaks_ind = self.q
        # self.__loaded_r_peaks_ind = EcgData.remove_duplicates_and_adjacent(self.q, self.frequency)
        # self.__loaded_r_peaks_ind = EcgData.filter_similar_elements(self.q, self.frequency)

        timestamps = np.array([i / sample_rate for i in range(len(ecg_signal))])

        self.raw_data = np.column_stack((timestamps, ecg_signal))

        self.__refresh_data()

        self.on_data_updated()

    @staticmethod
    def remove_duplicates_and_adjacent(array, frequency):
        # Remove duplicates and sort
        array = np.unique(array)
        result = []

        # for num in array:
        #     if not result or num != result[-1] + 1:
        #         result.append(num)

        for i in range(len(array) - 1, -1, -1):
            # Add to result if it doesn't differ by 1 from the last added number
            if not result or array[i] < result[-1] - int(0.05 * frequency):
                result.append(array[i])
        result.sort()
        return np.array(result)

    @staticmethod
    def filter_similar_elements(arr, frequency):
        arr = np.unique(arr)
        sorted_arr = np.sort(arr)

        result = []

        group = [sorted_arr[0]]

        for i in range(1, len(sorted_arr)):
            if sorted_arr[i] - sorted_arr[i - 1] <= int(0.05 * frequency):
                group.append(sorted_arr[i])
            else:
                mean_value = np.mean(group)
                closest_element = min(group, key=lambda x: abs(x - mean_value))
                result.append(closest_element)
                group = [sorted_arr[i]]

        mean_value = np.mean(group)
        closest_element = min(group, key=lambda x: abs(x - mean_value))
        result.append(closest_element)

        return np.array(result)

    def print_data(self):
        print("ECG DATA---------------")
        print(self.print_data_string())
        return

    def print_data_string(self):
        mean_rr = round(self.mean_rr * 1e3, 2) if self.mean_rr is not None else None
        sdnn = round(self.sdnn, 2) if self.sdnn is not None else None
        rmssd = round(self.rmssd, 2) if self.rmssd is not None else None
        pnn50 = round(self.pnn50, 2) if self.rmssd is not None else None

        # print("poczatki wdechu: ")
        # print(self.__inhalation_starts_moments)
        # print("poczatki wydechu: ")
        # print(self.__exhalation_starts_moments)

        self.__find_inex_haletion_moments()

        # # Wyświetlenie wyników końcowych
        # print("\nLista wdechów (początek, koniec, czas trwania):")
        # for start, end, czas, roznicaHR, roznicaRR in self.wdechy:
        #     print(f"Od {start} do {end}, czas trwania: {czas}, roznica hr: {roznicaHR}, roznica rr: {roznicaRR}")
        #
        # print("\nLista wydechów (początek, koniec, czas trwania):")
        # for start, end, czas, roznicaHR, roznicaRR in self.wydechy:
        #     print(f"Od {start} do {end}, czas trwania: {czas}, roznica hr: {roznicaHR}, roznica rr: {roznicaRR}")

        # Obliczenie i wyświetlenie średnich czasów trwania wdechu i wydechu
        try:
            if self.wdechy:
                srednia_wdechu = sum(czas for _, _, czas, _, _ in self.wdechy) / len(
                    self.wdechy
                )
                print(f"\nŚredni czas trwania wdechu: {srednia_wdechu:.2f}")

            if self.wydechy:
                srednia_wydechu = sum(czas for _, _, czas, _, _ in self.wydechy) / len(
                    self.wydechy
                )
                print(f"Średni czas trwania wydechu: {srednia_wydechu:.2f}")

            print(f"rsa index wynosi: {self.calculate_rsa_index(self.__hr[:,1])}")

            # print(f"rsa index wzgledny wynosi: {self.calculate_relative_rsa_index_()}")

            print(
                f"mean heart rate diff wynosi: {self.calculate_mean_heart_rate_diff()}"
            )

            print("rr")

            print(
                f"RR rsa index wynosi: {self.calculate_rsa_index(self.__rr_intervals[:,1])}"
            )

            #        print(f"RR rsa index wzgledny wynosi: {self.calculate_relative_rsa_index_RR()}")

            print(
                f"RR mean heart rate diff wynosi: {self.calculate_mean_heart_rate_diffRR()}"
            )

            print(f"średnie tętno wynosi: {self.calculate_mean_heart_rate()}")
        except:
            pass
        output = str()
        # print("ECG peaks---------------")
        # print (self.r_peaks)
        # print("ECG intervals---------------")
        # print (self.rr_intervals)
        # print("ECG hr---------------")
        # print (self.hr)
        # print("\n*********************\n")
        print(len(self.__rr_intervals))

        if mean_rr is not None:
            output += f"Mean RR: {mean_rr} ms\n"
        if sdnn is not None:
            output += f"SDNN: {sdnn}\n"
        if rmssd is not None:
            output += f"RMSSD: {rmssd}\n"
        if pnn50 is not None:
            output += f"PNN50: {pnn50}%\n"
        if len(output) > 0:
            output = output[:-1]
        return output

        # if mean_rr is not None:
        #     print(f"Mean RR: {mean_rr} ms")
        # if sdnn is not None:
        #     output += f"SDNN: {sdnn}\n"
        # if rmssd is not None:
        #     output += f"RMSSD: {rmssd}\n"
        # if pnn50 is not None:
        # print(f"PNN50: {pnn50}%")
        return

    # def push_raw_data(self, x, y):
    #     if isinstance(x, list) and isinstance(y, list):
    #         self.raw_data = np.concatenate((self.raw_data, np.column_stack((x, y))))
    #     else:
    #         self.raw_data = np.append(self.raw_data, [[x, y]], axis=0)
    #
    #     self.__set_dirty()
    #     return

    def push_raw_data(self, x, y):
        self.__lock.acquire()
        try:
            if isinstance(x, list) and isinstance(y, list):
                new_data = np.column_stack((x, y))

                if self.target_frequency is not None:
                    if self.raw_data.size > 0:
                        new_data = np.vstack((self.__raw_data[-1], new_data))

                    new_data = EcgData.interpolate_data(new_data, self.target_frequency)
                    if self.raw_data.size > 0:
                        new_data = new_data[1:]

                filtered_data, self.zi = self.update_and_filter(
                    new_data[:, 1]
                )  # Filtrowanie z zachowaniem stanu
                self.filtered_data = np.concatenate(
                    (
                        self.filtered_data,
                        np.column_stack((new_data[:, 0], filtered_data)),
                    )
                )

                self.__raw_data = np.concatenate((self.__raw_data, new_data))
            else:
                new_data = np.array([y])  # Przekształć wartość sygnału na tablicę
                filtered_data = self.update_and_filter(new_data)
                self.filtered_data = np.append(
                    self.filtered_data, [[x, filtered_data[0]]], axis=0
                )
                self.__raw_data = np.append(self.__raw_data, [[x, y]], axis=0)
        finally:
            self.__lock.release()

        self.__set_dirty()
        self.refresh_if_dirty()
        self.on_data_updated()
        return

    def create_bandpass_filter(self, lowcut, highcut, order):
        nyquist = 0.5 * self.frequency
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype="band")
        return b, a

    def calculate_rsa_index(self, values):
        hr_values = values
        return np.max(hr_values) - np.min(hr_values)

    def calculate_mean_heart_rate(self):
        hr_values = self.__hr[:, 1]
        return np.mean(hr_values) if len(hr_values) > 0 else 0

    def calculate_mean_heart_rate_diff(self):
        total_heart_rate_diff = 0
        number_of_cycles = len(self.wdechy) + len(self.wydechy)

        for _, _, _, diff, _ in self.wdechy:
            total_heart_rate_diff += abs(diff)

        for _, _, _, diff, _ in self.wydechy:
            total_heart_rate_diff += abs(diff)

        mean_absolute_difference = (
            total_heart_rate_diff / number_of_cycles if number_of_cycles > 0 else 0
        )

        return mean_absolute_difference

    def calculate_mean_heart_rate_diffRR(self):
        total_heart_rate_diff = 0
        number_of_cycles = len(self.wdechy) + len(self.wydechy)

        for _, _, _, _, diff in self.wdechy:
            total_heart_rate_diff += abs(diff)

        for _, _, _, _, diff in self.wydechy:
            total_heart_rate_diff += abs(diff)

        mean_absolute_difference = (
            total_heart_rate_diff / number_of_cycles if number_of_cycles > 0 else 0
        )

        return mean_absolute_difference

    def calculate_relative_rsa_index_(self):
        max_relative_rsa_diff = max(
            max(abs(diff) for _, _, _, diff, _ in self.wdechy),
            max(abs(diff) for _, _, _, diff, _ in self.wydechy),
        )
        return max_relative_rsa_diff

    def calculate_relative_rsa_index_RR(self):
        max_relative_rsa_diffRR = max(
            max(abs(diff) for _, _, _, _, diff in self.wdechy),
            max(abs(diff) for _, _, _, _, diff in self.wydechy),
        )
        return max_relative_rsa_diffRR

    def create_bandpass_filter1(self, lowcut, highcut, order):
        nyquist = 0.5 * self.frequency
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype="band")
        return b, a

    def __set_dirty(self):
        self.__is_dirty = True
        return

    def refresh_if_dirty(self):
        if self.__is_dirty == False:
            return

        self.__refresh_data()
        self.__is_dirty = False
        return

    def __refresh_data(self):
        # self.__find_r_peaks()

        self.__find_new_r_peaks()
        self.__find_new_r_peaks_filtered()
        self.__find_r_peaks_piotr()
        self.__calc_rr_intervals()

        # # Pobierz tylko interwały RR
        # rr_values = self.__rr_intervals[:, 1] * 1000

        # window_size = 1000
        # # Oblicz liczbę okien
        # num_windows = len(rr_values) // window_size

        # # Lista na SDNN dla każdego okna
        # sdnn_values = []

        # # Iteracja przez każde okno i obliczanie SDNN
        # for i in range(num_windows):
        #     window = rr_values[i * window_size:(i + 1) * window_size]
        #     sdnn = np.std(window, ddof=0)  # Odchylenie standardowe (ddof=1 to standardowe odchylenie)
        #     sdnn_values.append(sdnn)

        # # Rysowanie wykresu
        # plt.figure(figsize=(10, 6))
        # plt.plot(sdnn_values, marker='o')
        # plt.title('SDNN dla każdego okna z 300 interwałami')
        # plt.xlabel('Numer okna')
        # plt.ylabel('SDNN (ms)')
        # plt.grid(True)
        # plt.show()

        # median_rr = np.median(self.__rr_intervals[:,1])
        # anomalies = np.abs(self.__rr_intervals[:,1] - median_rr) > 0.10 * median_rr
        #
        # # Interpolacja odstających RR
        # corrected_rr = self.__rr_intervals[:,1].copy()
        # corrected_rr[anomalies] = np.mean(self.__rr_intervals[:,1][~anomalies])  # Zastępuje anomalie średnią
        #
        #
        # self.__rr_intervals[:,1] = corrected_rr

        self.__calc_mean_rr()
        self.__calc_sdnn()
        self.__calc_rmssd()
        self.__calc_pnn50()
        self.__calc_hr()
        if len(self.__hr) > 28:
            self.hr_filtered = self.filter_hr()
            ups = self.__find_hr_ups()
            self.__exhalation_starts_moments = np.column_stack(
                (
                    ups[:, 0]
                    - self.__hr[0][0],  # Znormalizowane czasy (pierwsza kolumna)
                    ups[:, 1],  # Oryginalne wartości (druga kolumna)
                )
            )

            downs = self.__find_hr_downs()
            self.__inhalation_starts_moments = np.column_stack(
                (
                    downs[:, 0]
                    - self.__hr[0][0],  # Znormalizowane czasy (pierwsza kolumna)
                    downs[:, 1],  # Oryginalne wartości (druga kolumna)
                )
            )

            # Zakładając posortowane listy minima (wdechy) i maksima (wydechy)

        return

    def __find_hr_downs(self):
        self.__hr_downs = self.find_hr_downs(
            self.__hr, self.frequency, self.lowcutHR, self.highcutHR
        )
        return self.__hr_downs

    def __find_hr_ups(self):
        self.__hr_ups = self.find_hr_ups(
            self.__hr, self.frequency, self.lowcutHR, self.highcutHR
        )
        return self.__hr_ups

    def __find_r_peaks(self):
        self.__r_peaks = self.__r_peaks_finder.find_r_peaks_values_with_timestamps(
            self.__raw_data, self.frequency
        )
        self.__r_peaks_ind = self.__r_peaks_finder.find_r_peaks_ind(
            self.__raw_data[:, 1], self.frequency
        )
        return self.__r_peaks

    def __find_r_peaks_filtered(self):
        self.__r_peaks_filtered = (
            self.__r_peaks_finder.find_r_peaks_values_with_timestamps(
                self.filtered_data, self.frequency
            )
        )
        return self.__r_peaks_filtered

    def __find_inex_haletion_moments(self):
        minima = self.__inhalation_starts_moments[:, 0]
        maksima = self.__exhalation_starts_moments[:, 0]
        minimaHR = self.__inhalation_starts_moments[:, 1]
        maksimaHR = self.__exhalation_starts_moments[:, 1]
        self.wdechy.clear()
        self.wydechy.clear()

        i, j = 0, 0  # Indeksy do list minimów i maksimów

        while i < len(minima) and j < len(maksima):
            if minima[i] < maksima[j]:
                if not (i + 1 < len(minima) and minima[i + 1] < maksima[j]):
                    czas = maksima[j] - minima[i]
                    roznica_hr = maksimaHR[j] - minimaHR[i]
                    roznica_rr = 60 / maksimaHR[j] - 60 / minimaHR[i]
                    self.wdechy.append(
                        (minima[i], maksima[j], czas, roznica_hr, roznica_rr)
                    )
                i += 1
            else:
                if not (j + 1 < len(maksima) and maksima[j + 1] < minima[i]):
                    czas = minima[i] - maksima[j]
                    roznica_hr = minimaHR[i] - maksimaHR[j]
                    roznica_rr = 60 / minimaHR[i] - 60 / maksimaHR[j]
                    self.wydechy.append(
                        (maksima[j], minima[i], czas, roznica_hr, roznica_rr)
                    )
                j += 1

    # @staticmethod
    # def find_r_peaks(data: np.ndarray, frequency: int) -> np.ndarray:
    #     return PanTompkins.find_r_peaks_values_with_timestamps(data, frequency)

    @staticmethod
    def find_hr_ups(data: np.ndarray, frequency: int, a: float, b: float) -> np.ndarray:
        return PanTompkins.find_hr_peaks(data, frequency, a, b, SIZE)

    @staticmethod
    def find_hr_downs(
        data: np.ndarray, frequency: int, a: float, b: float
    ) -> np.ndarray:

        return PanTompkins.find_hr_peaks(data, frequency, a, b, SIZE, -1)

    def __find_new_r_peaks(self):
        self.__lock.acquire()
        try:
            # thats probably a reference type, but somehow it fixes synchro bug
            data = self.__raw_data
        finally:
            self.__lock.release()

        if len(data) < self.frequency * 2:
            return
        if len(self.__r_peaks) < 4:
            self.__find_r_peaks()
            return

        index = -1
        for i in range(len(data) - 1, -1, -1):
            if data[i][0] < self.__r_peaks[-1][0]:
                index = i
                break
        # offset = int(self.__mean_rr/2*self.frequency)
        # new_peaks = PanTompkins.find_r_peaks(self.__raw_data[-(index+offset):], self.frequency)
        # new_peaks = self.__r_peaks_finder.find_r_peaks_values_with_timestamps(self.__raw_data[-index:], self.frequency)
        # new_peaks_ind = self.__r_peaks_finder.find_r_peaks_ind(
        #     self.__raw_data[-index:, 1], self.frequency
        # )

        new_peaks = self.__r_peaks_finder.find_r_peaks_values_with_timestamps(
            data[-index:], self.frequency
        )
        # new_peaks_ind = self.__r_peaks_finder.find_r_peaks_ind(
        #         data[-index:, 1], self.frequency
        #     )

        start_index = -1
        for i in range(len(new_peaks)):
            if self.__r_peaks[-1][0] < new_peaks[i][0]:
                start_index = i
                break

        if -1 == start_index:
            return

        rr_intervals = self.calc_rr_intervals(self.__r_peaks[-4:])
        last_3_mean_rr = self.calc_mean_rr(rr_intervals)

        if data[-1][0] - new_peaks[-1][0] > last_3_mean_rr / 2:
            end_index = len(new_peaks - 1)
        else:
            end_index = -1

        nowe_peaki = new_peaks[start_index - 1 : end_index]

        # if len(nowe_peaki) <= 1:
        #     if len(self.__r_peaks) >= 2:
        #         for i in range(len(self.__r_peaks) - 1):
        #             x0 = self.__r_peaks[i][0]
        #             x1 = self.__r_peaks[i+1][0]
        #             x = x1 - x0
        #             y = 60 / x
        #
        #             # Dodajemy nową wartość do bufora
        #             new_data = np.array([y])  # Przekształć wartość sygnału na tablicę
        #             filtered_data = self.update_and_filter_hr(new_data)  # Filtracja nowej wartości
        #
        #             # Dodaj przefiltrowany wynik do przechowywanej tablicy wyników
        #             self.hr_filtered = np.append(self.hr_filtered, [[x1, filtered_data[0]]], axis=0)

        if len(nowe_peaki) > 1:
            x0 = nowe_peaki[0][0]
            x1 = nowe_peaki[1][0]
            x = x1 - x0
            y = 60 / x

            # Dodajemy nową wartość do bufora
        #          new_data = np.array([y])  # Przekształć wartość sygnału na tablicę
        #        self.update_hr(new_data)  # Filtracja nowej wartości

        #        if (len(self.__hr) > 15):
        #        filtered_data = self.filter_hr(new_data)  # Filtracja nowej wartości
        # Dodaj przefiltrowany wynik do przechowywanej tablicy wyników
        # self.hr_filtered = np.append(self.hr_filtered, [[x1, filtered_data[0]]], axis=0)
        #          self.hr_filtered = filtered_data
        # else:
        #     self.hr_filtered = np.append(self.hr_filtered, [[x1, new_data[0]]], axis=0)
        self.__r_peaks = np.vstack((self.__r_peaks, new_peaks[start_index:end_index]))

        # self.__r_peaks_ind = np.hstack((self.__r_peaks_ind, new_peaks_ind[start_index:end_index]))

        if len(self.__r_peaks) != len(self.__r_peaks_ind):
            return
            raise Exception("The arrays are of different lengths!")

    def __find_new_r_peaks_filtered(self):
        if len(self.filtered_data) < self.frequency * 2:
            return
        if len(self.__r_peaks_filtered) == 0:
            self.__find_r_peaks_filtered()
            return

        index = -1
        for i in range(len(self.filtered_data) - 1, -1, -1):
            if self.filtered_data[i][0] < self.__r_peaks_filtered[-1][0]:
                index = i
                break
        # offset = int(self.__mean_rr/2*self.frequency)
        # new_peaks = PanTompkins.find_r_peaks(self.raw_data[-(index+offset):], self.frequency)
        new_peaks = self.__r_peaks_finder.find_r_peaks_values_with_timestamps(
            self.filtered_data[-index:], self.frequency
        )

        start_index = -1
        for i in range(len(new_peaks)):
            if self.__r_peaks_filtered[-1][0] < new_peaks[i][0]:
                start_index = i
                break

        if -1 == start_index:
            return

        rr_intervals = self.calc_rr_intervals(self.__r_peaks_filtered[-4:])
        last_3_mean_rr = self.calc_mean_rr(rr_intervals)

        if self.filtered_data[-1][0] - new_peaks[-1][0] > last_3_mean_rr:
            end_index = len(new_peaks - 1)
        else:
            end_index = -1

        self.__r_peaks_filtered = np.vstack(
            (self.__r_peaks_filtered, new_peaks[start_index:end_index])
        )

    def __find_r_peaks_piotr(self):
        self.__r_peaks_piotr = NAME_THIS_MODULE_YOURSELF_PIOTER.find_r_peaks_piotr(
            self.__raw_data
        )
        return self.__r_peaks_piotr

    def __calc_rr_intervals(self):
        self.__rr_intervals = self.calc_rr_intervals(self.__r_peaks)
        return self.__rr_intervals

    @staticmethod
    def calc_rr_intervals(r_peaks: np.ndarray) -> np.ndarray:
        # size = len(r_peaks)
        # # Dane testowe (RR w sekundach)
        # t_long = np.linspace(0, 80, 100)  # 150 RR intervals
        # intervals = 0.8 + 0.4 * np.sin(2 * np.pi * 0.25 * t_long)
        #
        # # Parameters for the sine wave
        # frequency = 0.2  # Frequency in Hz
        # sampling_rate = 1  # Sampling rate in Hz
        # duration = 80  # Duration of the signal in seconds
        #
        # # Time vector
        # t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        #
        # # Sine wave
        # sine_wave = 1.01 + np.sin(2 * np.pi * frequency * t)
        # minter = np.column_stack((t, sine_wave))
        #
        # inter = np.column_stack((t_long, intervals))
        # return inter

        if len(r_peaks) < 2:
            return np.empty((0, 2))

        rr_intervals = np.diff([peak[0] for peak in r_peaks])

        timestamps = r_peaks[1:, 0]

        # Połącz znaczniki czasowe i wartości tętna w kolumny
        result = np.column_stack((timestamps, rr_intervals))
        return result

    def __calc_mean_rr(self):
        self.__mean_rr = self.calc_mean_rr(self.__rr_intervals[:, 1])
        return self.__mean_rr

    @staticmethod
    def calc_mean_rr(rr_intervals: np.ndarray) -> np.ndarray:
        return np.mean(rr_intervals) if len(rr_intervals) > 0 else None

    def __calc_sdnn(self):
        self.__sdnn = self.calc_sdnn(self.__rr_intervals[:, 1])
        return self.__sdnn

    @staticmethod
    def calc_sdnn(rr_intervals: np.ndarray) -> float:
        return np.std(rr_intervals) * 1e3 if len(rr_intervals) > 0 else None

    def __calc_hr(self):
        self.__hr = self.calc_hr(self.__rr_intervals)
        return self.__hr

    @staticmethod
    def calc_hr(rr_intervals: np.ndarray) -> np.ndarray:

        # Czas znacznika przypisany do heart_rate to czas końca każdego odstępu RR
        timestamps = rr_intervals[:, 0]

        # Oblicz wartości tętna jako 60 / RR
        heart_rates = 60 / rr_intervals[:, 1]

        # Połącz znaczniki czasowe i wartości tętna w kolumny
        result = np.column_stack((timestamps, heart_rates))

        # if len(result) > 10:
        #     result2 = PanTompkins.filter_ecg_with_timestamps(result, 100)
        #     print(result2)
        #     return result2
        return result

    def __calc_rmssd(self):
        self.__rmssd = self.calc_rmssd(self.__rr_intervals[:, 1])
        return self.__rmssd

    @staticmethod
    def calc_rmssd(rr_intervals: np.ndarray) -> float:
        diff_rr_intervals = np.diff(rr_intervals)
        return (
            np.sqrt(np.mean(diff_rr_intervals**2)) * 1e3
            if len(diff_rr_intervals) > 0
            else None
        )

    def __calc_pnn50(self):
        self.__pnn50 = self.calc_pnn50(self.__rr_intervals[:, 1])
        return self.__pnn50

    @staticmethod
    def calc_pnn50(rr_intervals: np.ndarray) -> float:
        if len(rr_intervals) < 2:
            return None

        diff_rr_intervals = np.diff(rr_intervals)

        nn50_count = np.sum(np.abs(diff_rr_intervals) > 0.050)  # 50 ms
        return (
            (nn50_count / len(diff_rr_intervals)) * 100
            if len(diff_rr_intervals) > 0
            else None
        )

    def extract_windows(self, window_size):
        win_count = int(len(self.__raw_data) / window_size)

        X_train = np.zeros((win_count, window_size), dtype=np.float64)
        y_train = np.zeros((win_count, window_size))
        R_per_w = []

        normalize = partial(processing.normalize_bound, lb=-1, ub=1)

        for i in range(win_count):
            # for i in tqdm(range(win_count)):
            win_start = i * window_size
            end = win_start + window_size
            r_peaks_ind = np.where(
                (self.__r_peaks_ind >= win_start) & (self.__r_peaks_ind < end)
            )
            R_per_w.append(self.__r_peaks_ind[r_peaks_ind] - win_start)

            for j in self.__r_peaks_ind[r_peaks_ind]:
                r = int(j) - win_start
                y_train[i, r - 2 : r + 3] = 1

            if self.__raw_data[win_start:end][1].any():
                X_train[i:] = np.squeeze(
                    np.apply_along_axis(normalize, 0, self.__raw_data[win_start:end, 1])
                )
            else:
                X_train[i, :] = self.__raw_data[win_start:end, 1].T

        X_train = np.expand_dims(X_train, axis=2)

        y_train = np.expand_dims(y_train, axis=2)

        return X_train, y_train, R_per_w

    def extract_piotr(self, window_size):
        X, _, y = self.extract_windows(window_size)
        intervals = [np.diff(row)/130 for row in y]
        result_y = [[self.calc_sdnn(row), self.calc_rmssd(row)] for row in intervals]
        result_y = np.array(result_y)
        return X, result_y

    def extract_windows_loaded_peaks(self, window_size):
        win_count = int(len(self.__raw_data) / window_size)

        X_train = np.zeros((win_count, window_size), dtype=np.float64)
        y_train = np.zeros((win_count, window_size))
        R_per_w = []

        normalize = partial(processing.normalize_bound, lb=-1, ub=1)

        for i in range(win_count):
            # for i in tqdm(range(win_count)):
            win_start = i * window_size
            end = win_start + window_size
            r_peaks_ind = np.where(
                (self.__loaded_r_peaks_ind >= win_start)
                & (self.__loaded_r_peaks_ind < end)
            )
            R_per_w.append(self.__loaded_r_peaks_ind[r_peaks_ind] - win_start)

            for j in self.__loaded_r_peaks_ind[r_peaks_ind]:
                r = int(j) - win_start
                y_train[i, r - 2 : r + 3] = 1

            if self.__raw_data[win_start:end][1].any():
                X_train[i:] = np.squeeze(
                    np.apply_along_axis(normalize, 0, self.__raw_data[win_start:end, 1])
                )
            else:
                X_train[i, :] = self.__raw_data[win_start:end, 1].T

        X_train = np.expand_dims(X_train, axis=2)

        y_train = np.expand_dims(y_train, axis=2)

        return X_train, y_train, R_per_w

    def extract_test_windows(self, win_size, stride):
        normalize = partial(processing.normalize_bound, lb=-1, ub=1)

        signal = np.squeeze(self.__raw_data[:, 1])

        pad_sig = np.pad(signal, (win_size - stride, win_size), mode="edge")

        data_windows = []
        win_idx = []

        pad_id = np.arange(pad_sig.shape[0])

        for win_id in range(0, len(pad_sig), stride):
            if win_id + win_size < len(pad_sig):

                window = pad_sig[win_id : win_id + win_size]
                if window.any():
                    window = np.squeeze(np.apply_along_axis(normalize, 0, window))

                data_windows.append(window)
                win_idx.append(pad_id[win_id : win_id + win_size])

        data_windows = np.asarray(data_windows)
        data_windows = data_windows.reshape(
            data_windows.shape[0], data_windows.shape[1], 1
        )
        win_idx = np.asarray(win_idx)
        win_idx = win_idx.reshape(win_idx.shape[0] * win_idx.shape[1])

        return win_idx, data_windows

    def extract_hrv_windows_with_loaded_peaks(
        self, window_size, metrics: list = ["SDNN", "RMSSD"]
    ):
        metrics = [s.lower() for s in metrics]
        win_count = int(len(self.__rr_intervals) / window_size)
        r_peaks = self.raw_data[self.__loaded_r_peaks_ind]
        rr_intervals = np.diff([peak[0] for peak in r_peaks])
        X_train = np.zeros((win_count, window_size), dtype=np.float64)
        # y_train = np.zeros((win_count, 3))
        y_train = np.zeros((win_count, len(metrics)))

        for i in range(win_count):
            win_start = i * window_size
            win_end = win_start + window_size
            # X_train[i] = self.rr_intervals[win_start:win_end]
            X_train[i] = rr_intervals[win_start:win_end]
            # y_train[i] = (EcgData.calc_sdnn(X_train[i]), EcgData.calc_rmssd(X_train[i]), 0)

            if "sdnn" in metrics:
                y_train[i, metrics.index("sdnn")] = EcgData.calc_sdnn(X_train[i])

            if "rmssd" in metrics:
                y_train[i, metrics.index("rmssd")] = EcgData.calc_rmssd(X_train[i])

            if "lf" in metrics or "hf" in metrics or "lf/hf" in metrics:
                lf, hf = FFT.calc_lf_hf(X_train[i])
                if "lf" in metrics:
                    y_train[i, metrics.index("lf")] = lf
                if "hf" in metrics:
                    y_train[i, metrics.index("hf")] = hf
                if "lf/hf" in metrics:
                    y_train[i, metrics.index("lf/hf")] = lf / hf

            # y_train[i] = (EcgData.calc_sdnn(X_train[i]), EcgData.calc_rmssd(X_train[i]), lf, hf)

        return X_train, y_train

        """
        # win_count = int(len(self.__rr_intervals) / window_size)
        # r_peaks = self.raw_data[self.__loaded_r_peaks_ind]
        # rr_intervals = np.diff([peak[0] for peak in r_peaks])
        
        

        # X_train = np.zeros((win_count, window_size), dtype=np.float64)
        # # y_train = np.zeros((win_count, 3))
        # y_train = np.zeros((win_count, 1))

        # for i in range(win_count):
        #     win_start = i * window_size
        #     win_end = win_start + window_size
        #     # X_train[i] = self.rr_intervals[win_start:win_end]
        #     X_train[i] = rr_intervals[win_start:win_end]
        #     # y_train[i] = (EcgData.calc_sdnn(X_train[i]), EcgData.calc_rmssd(X_train[i]), 0)
        #     # y_train[i] = (EcgData.calc_sdnn(X_train[i]), EcgData.calc_rmssd(X_train[i]))
        #     y_train[i] = EcgData.calc_sdnn(X_train[i])

        # return X_train, y_train
        """

    def extract_hrv_windows_with_detected_peaks(
        self, window_size, metrics: list = ["SDNN", "RMSSD"]
    ):
        metrics = [s.lower() for s in metrics]
        win_count = int(len(self.__rr_intervals) / window_size)

        X_train = np.zeros((win_count, window_size), dtype=np.float64)
        # y_train = np.zeros((win_count, 3))
        y_train = np.zeros((win_count, len(metrics)))

        for i in range(win_count):
            win_start = i * window_size
            win_end = win_start + window_size
            # X_train[i] = self.rr_intervals[win_start:win_end]
            X_train[i] = self.__rr_intervals[win_start:win_end, 1]
            # y_train[i] = (EcgData.calc_sdnn(X_train[i]), EcgData.calc_rmssd(X_train[i]), 0)

            if "sdnn" in metrics:
                y_train[i, metrics.index("sdnn")] = EcgData.calc_sdnn(X_train[i])

            if "rmssd" in metrics:
                y_train[i, metrics.index("rmssd")] = EcgData.calc_rmssd(X_train[i])

            if "lf" in metrics or "hf" in metrics or "lf/hf" in metrics:
                lf, hf = FFT.calc_lf_hf(X_train[i])
                if "lf" in metrics:
                    y_train[i, metrics.index("lf")] = lf
                if "hf" in metrics:
                    y_train[i, metrics.index("hf")] = hf
                if "lf/hf" in metrics:
                    y_train[i, metrics.index("lf/hf")] = lf / hf

            # y_train[i] = (EcgData.calc_sdnn(X_train[i]), EcgData.calc_rmssd(X_train[i]), lf, hf)

        return X_train, y_train

    def check_detected_peaks(self):
        intersection1 = np.intersect1d(self.__r_peaks_ind, self.__loaded_r_peaks_ind)
        test = PanTompkins.refine_peak_positions(
            self.raw_data[:, 1], self.__loaded_r_peaks_ind, 20
        )
        test = self.refined_loaded_peaks_ind
        intersection2 = np.intersect1d(self.__r_peaks_ind, test)
        # print(
        #     f"Correctly classified: {round(intersection2.size / self.__loaded_r_peaks_ind.size * 100, 2)}%"
        # )
        TP = intersection2.size
        FP = np.setdiff1d(self.__r_peaks_ind, intersection2).size
        FN = self.__loaded_r_peaks_ind.size - intersection2.size
        # print(f"TP {TP}")
        # print(f"FP {FP}")
        # print(f"FN {FN}")
        sens = TP / (TP + FN)
        prec = TP / (TP + FP)
        f1 = 2 * prec * sens / (prec + sens)
        # print(f"Sensitivity {sens}")
        # print(f"Precision {prec}")
        print(f"F1-score {f1}")
        pass

    @property
    def refined_loaded_peaks_ind(self):
        max_distance = int(0.05 * self.frequency)
        distances = []

        # Refine R peaks indices
        result_indexes = np.array(
            [
                val
                for idx, val in enumerate(self.__r_peaks_ind)
                if any(abs(val - a1) <= max_distance for a1 in self.loaded_r_peak_ind)
            ]
        )

        # Calculate distances
        for val in result_indexes:
            closest_peak = min(self.loaded_r_peak_ind, key=lambda a1: abs(val - a1))
            distances.append(abs(val - closest_peak))
            # print(distances[-1])

        # Compute average distance
        average_distance = np.mean(distances) if distances else 0
        # print(average_distance/self.frequency*1000)

        return result_indexes  # , average_distance
        # return PanTompkins.refine_peak_positions(
        #     self.raw_data[:, 1], self.__loaded_r_peaks_ind, 25
        # )

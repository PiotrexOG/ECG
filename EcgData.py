from traceback import print_tb

import numpy as np

import matplotlib.pyplot as plt

import FFT
import PanTompkins
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
import doba_analysis
from matplotlib import ticker
import os


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

#GLOWNA KLASA ODPOWIADAJACA ZA PRZETWARZANIE DANYCH W PROGRAMIE

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
        self.rr_frequency = RR_FREQ
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
        self.__rr_intervals = np.empty((0, 2))
        self.__hr = np.empty((0, 2))
        self.hr_filtered = np.empty((0, 2))

        self.__callbacks = []

        self.__mean_rr = -1
        self.__sdnn = -1
        self.__rmssd = -1
        self.__pnn50 = -1
        self.sdann = -1
        self.sdnn_index = -1
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
        timestamps = self.__hr[:, 0]
        hr_values = self.__hr[:, 1]

        # Generowanie nowych timestampów z częstotliwością 4 Hz
        new_timestamps = np.arange(timestamps[0], timestamps[-1], 1/self.rr_frequency)

        # Interpolacja (używamy metody liniowej)
        interpolator = interp1d(timestamps, hr_values, kind='linear', fill_value="extrapolate")
        new_hr_values = interpolator(new_timestamps)

        filtered_signal = signal.filtfilt(self.bHR, self.aHR, new_hr_values)
        # return filtered_signal[-len(new_data):]  # Zwróć tylko najnowsze przefiltrowane wartości
        peaks_with_timestamps = np.column_stack((new_timestamps, filtered_signal))
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

    def derivative_filter(self, sig):
        return np.diff(sig, prepend=0)

    def square1(self, sig):
        return sig**2

    def moving_window_integration(self, sig, window_size):
        return np.convolve(sig, np.ones(window_size) / window_size, mode="same")

    def load_csv_data(self, path, save_peaks=False):
        csv_data = np.array(read_csv(path))
        if NORMALIZED_TEST_DATA_TIME:
            csv_data[:, 0] -= csv_data[0, 0]
        if NEGATE_INCOMING_DATA:
            csv_data[:, 1] *= -1
        csv_data[:, 0] /= TIMESTAMP_SCALE_FACTOR

        path_without_ext = os.path.splitext(path)[0]
        # if not save_peaks:
        #     try:
        #         if self.__r_peaks_finder.__class__.__name__ == PanTompkinsFinder.__name__:
        #             self.__loaded_r_peaks_ind = np.loadtxt(f"{path_without_ext}_r.txt")
        #         else:
        #             self.__loaded_r_peaks_ind = np.loadtxt(f"{path_without_ext}_r.txt")
        #         self.__loaded_r_peaks_ind = self.__loaded_r_peaks_ind.astype(int)
        #         self.__loaded_r_peaks_ind.sort()
        #     except:
        #         pass

        # indexes = np.arange(0, csv_data.shape[0])
        # csv_data = np.column_stack((indexes, csv_data[:, 1]))

        # filtered_signal = PanTompkins.bandpass_filter(csv_data[:, 1], self.frequency, 0.5, 40)
        # baseline_corrected_signal = EcgData.highpass_filter(filtered_signal, cutoff=0.5, fs=self.frequency)
        # csv_data = np.column_stack((csv_data[:, 0], baseline_corrected_signal))

        # baseline_corrected_signal = EcgData.highpass_filter(
        #     csv_data[:, 1], cutoff=0.5, fs=self.frequency
        # )
        # filtered_signal = PanTompkins.bandpass_filter(
        #     csv_data[:, 1], self.frequency, 5, 18
        # )
        # filtered_signal = PanTompkins.derivative_filter(filtered_signal)
        # filtered_signal = PanTompkins.square(filtered_signal)
        # window_size = int(0.175 * self.frequency)
        # filtered_signal = PanTompkins.moving_window_integration(filtered_signal, window_size)
        # csv_data = np.column_stack((csv_data[:, 0], filtered_signal))

        if self.target_frequency is not None:
            if len(self.__loaded_r_peaks_ind) > 0:
                csv_data[:, 0] -= csv_data[0, 0]
                start_time = csv_data[0, 0]
                end_time = csv_data[-1, 0]
                r_peak_timestamps = csv_data[self.__loaded_r_peaks_ind, 0]

                new_timestamps = np.arange(
                    0, end_time - start_time, 1 / self.target_frequency
                )

                mapped_indexes = [
                    np.argmin(np.abs(new_timestamps - ts)) for ts in r_peak_timestamps
                ]
                self.__loaded_r_peaks_ind = np.array(mapped_indexes)
            # self.__loaded_r_peaks_ind= np.floor(self.__loaded_r_peaks_ind*(self.target_frequency/self.frequency)).astype(int)
            interpolated_data = EcgData.interpolate_data(
                csv_data, self.target_frequency
            )
            self.frequency = self.target_frequency
            self.__raw_data = interpolated_data
        else:
            self.__raw_data = csv_data
            self.frequency = SAMPLING_RATE

        self.frequency = SAMPLING_RATE

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

        #self.raw_data = self.detect_and_remove_artifacts(self.raw_data)


        # print(len(self.raw_data))

        # self.filtered_data = np.column_stack((csv_data[:, 0], filtered_signal))
        # print(len(self.filtered_data))

        # self.__set_dirty()
        # self.__refresh_if_dirty()

        # try:
        #     self.hr_filtered = self.filter_hr()
        # except:
        #     pass
        # FFT.fft(self.__rr_intervals[:, 1])

        # Wave.analyze(self.__rr_intervals)
        # self.print_data()

        if save_peaks:
            # np.save(path_without_ext, self.__r_peaks_ind)
            np.savetxt(f"{path_without_ext}.txt", self.__r_peaks_ind, fmt="%d")

        self.on_data_updated()

    def clean_r_peaks(self, sampling_rate=130, threshold_multiplier=1.1):
        r_times = self.r_peaks[:, 0].tolist()  # Konwersja na listę dla łatwego usuwania
        r_values = self.r_peaks[:, 1].tolist()

        print(len(r_values))
        print(max(r_values))

        # Obliczenie globalnej średniej wartości absolutnej R-peaków
        global_mean = np.mean(np.abs(r_values))

        # Iteracja przez R-peaks od końca do początku (aby unikać przesunięć indeksów)
        idx = len(r_values) - 1
        while idx >= 0:
            r_time = r_times[idx]
            r_value = r_values[idx]

            # Sprawdzanie, czy wartość R-peaku przekracza próg
            if np.abs(r_value) > threshold_multiplier * global_mean:
                # Wyznaczanie zakresu czasowego
                start_time = r_times[
                    idx - 1] if idx > 0 else r_time  # Poprzedni R-peak lub bieżący, jeśli brak poprzedniego
                end_time = r_times[idx + 1] if idx < len(
                    r_times) - 1 else r_time  # Następny R-peak lub bieżący, jeśli brak następnego

                # Znajdowanie próbek w raw_data w tym zakresie
                raw_indices = np.where((self.raw_data[:, 0] >= start_time) & (self.raw_data[:, 0] <= end_time))

                # Ustawianie wartości na 0
                self.raw_data[raw_indices, 1] = 0  # Lub np.nan, jeśli preferujesz

                # Usuwanie R-peaku
                del r_times[idx]
                del r_values[idx]

            idx -= 1

        # Aktualizacja self.r_peaks po usunięciu
        self.__r_peaks = np.column_stack((r_times, r_values))


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

        timestamps = np.array([i / sample_rate for i in range(len(ecg_signal))])

        self.raw_data = np.column_stack((timestamps, ecg_signal))

        self.__refresh_data()

        self.on_data_updated()

    @staticmethod
    def remove_duplicates_and_adjacent(array, frequency):
        # Remove duplicates and sort
        array = np.unique(array)
        result = []

        for i in range(len(array) - 1, -1, -1):
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

        print("sdann=")

        print(self.sdann)
        print("sdnn_index=")
        print(self.sdnn_index)

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

            rsa_ids = self.calculate_rsa_indexes()
            # print(rsa_ids)


            # # Printowanie danych w żądanym formacie
            # print("Czas (timestamp),Różnica (diff)")
            # for time, diff in rsa_ids:
            #     print(f"{time},{diff:.3f}")

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
                try:
                    self.filtered_data = np.concatenate(
                        (
                            self.filtered_data,
                            np.column_stack((new_data[:, 0], filtered_data)),
                        )
                    )
                except:
                    pass

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

    def calculate_rsa_indexes(self):
        # Inicjalizacja listy do przechowywania czasu i różnic
        rsa_indexes = []

        # Dodanie danych z wdechów
        for _, time, _, diff, rr_diff in self.wdechy:
            rsa_indexes.append([time, abs(rr_diff)])

        # Dodanie danych z wydechów
        for _, time, _, diff, rr_diff in self.wydechy:
            rsa_indexes.append([time, abs(rr_diff)])

        # Konwersja listy na tablicę numpy dla wygodniejszego przetwarzania
        rsa_indexes = np.array(rsa_indexes)

        rsa_indexes = rsa_indexes[rsa_indexes[:, 0].argsort()]

        # Zwrócenie tablicy w formacie wymaganym
        return rsa_indexes

    def calculate_mean_heart_rate(self):
        hr_values = self.__hr[:, 1]
        return np.mean(hr_values) if len(hr_values) > 0 else 0

    def calculate_mean_heart_rate_diff(self):
        total_heart_rate_diff = 0
        number_of_cycles = len(self.wdechy) + len(self.wydechy)

        for _, time, _, diff, _ in self.wdechy:
            total_heart_rate_diff += abs(diff)

        for _, time, _, diff, _ in self.wydechy:
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
        nyquist = 0.5 * self.rr_frequency
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
            ax.axvspan(start_time, start_time + duration, color=colors[idx], alpha=0.6, lw=0)
            start_time += duration

    def gaussian_smooth(self, data, weights):
        return np.convolve(data, weights, mode='same')

    def __refresh_data(self):
        # self.__find_r_peaks()

        self.__find_new_r_peaks()
        try:
            self.__find_new_r_peaks_filtered()
        except:
            pass

        #self.clean_r_peaks()

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
        try:
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
        except:
            pass
    
        self.sdann, self.sdnn_index = doba_analysis.analyze(self.raw_data, self.rr_intervals)

        return

    def __find_hr_downs(self):
        self.__hr_downs = self.find_hr_downs(
            self.__hr, self.rr_frequency, self.lowcutHR, self.highcutHR
        )
        return self.__hr_downs

    def __find_hr_ups(self):
        self.__hr_ups = self.find_hr_ups(
            self.__hr, self.rr_frequency, self.lowcutHR, self.highcutHR
        )
        return self.__hr_ups

    def __find_r_peaks(self):
        self.__r_peaks = self.__r_peaks_finder.find_r_peaks_values_with_timestamps(
            self.__raw_data,
            self.frequency if self.target_frequency is None else self.target_frequency,
        )
        self.__r_peaks_ind = self.__r_peaks_finder.find_r_peaks_ind(
            self.__raw_data[:, 1],
            self.frequency if self.target_frequency is None else self.target_frequency,
        )
        return self.__r_peaks

    def __find_r_peaks_filtered(self):
        self.__r_peaks_filtered = (
            self.__r_peaks_finder.find_r_peaks_values_with_timestamps(
                self.filtered_data,
                (
                    self.frequency
                    if self.target_frequency is None
                    else self.target_frequency
                ),
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
            data[-index:],
            self.frequency if self.target_frequency is None else self.target_frequency,
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

    def extract_windows(self, window_size, stride=None):
        if stride is None:
            stride = window_size
        win_count = int((len(self.__raw_data) - window_size) / stride) + 1
        # filt_values = PanTompkins.bandpass_filter(self.__raw_data[:, 1], self.target_frequency if self.target_frequency is not None else self.frequency)
        # filt_data = np.column_stack((self.__raw_data[:, 0], filt_values))
        filt_data = self.__raw_data
        X_train = np.zeros((win_count, window_size), dtype=np.float64)
        y_train = np.zeros((win_count, window_size))
        R_per_w = []

        normalize = partial(processing.normalize_bound, lb=-1, ub=1)

        for i in range(win_count):
            win_start = i * stride
            end = win_start + window_size
            r_peaks_ind = np.where(
                (self.__r_peaks_ind >= win_start) & (self.__r_peaks_ind < end)
            )
            R_per_w.append(self.__r_peaks_ind[r_peaks_ind] - win_start)

            for j in self.__r_peaks_ind[r_peaks_ind]:
                r = int(j) - win_start
                y_train[i, r - 2 : r + 3] = 1

            if filt_data[win_start:end][1].any():
                X_train[i, :] = np.squeeze(
                    np.apply_along_axis(normalize, 0, filt_data[win_start:end, 1])
                )
            else:
                X_train[i, :] = filt_data[win_start:end, 1].T

        X_train = np.expand_dims(X_train, axis=2)
        y_train = np.expand_dims(y_train, axis=2)

        return X_train, y_train, R_per_w

    def extract_piotr(self, window_size, metrics: list = ["SDNN"], stride: int = None):
        metrics = [s.lower() for s in metrics]
        X, _, y = self.extract_windows(window_size, stride)
        intervals = [np.diff(row) / 130 for row in y]
        result_y = None
        if "sdnn" in metrics:
            sdnn = np.array([self.calc_sdnn(row) for row in intervals])
            if result_y is None:
                result_y = sdnn[:, None]
            else:
                result_y = np.column_stack((result_y, sdnn))
        if "rmssd" in metrics:
            rmssd = np.array([self.calc_rmssd(row) for row in intervals])
            if result_y is None:
                result_y = rmssd[:, None]
            else:
                result_y = np.column_stack((result_y, rmssd))
        if "lf" in metrics or "hf" in metrics or "lf/hf" in metrics:
            lf, hf = zip(*[FFT.calc_lf_hf(row) for row in intervals])
            lf = np.array(lf)
            hf = np.array(hf)
            if "lf" in metrics:
                if result_y is None:
                    result_y = lf[:, None]
                else:
                    result_y = np.column_stack((result_y, lf))
            if "hf" in metrics:
                if result_y is None:
                    result_y = hf[:, None]
                else:
                    result_y = np.column_stack((result_y, hf))
            if "lf/hf" in metrics:
                lfhf = np.divide(
                    lf, hf, out=np.zeros_like(lf, dtype=float), where=hf != 0
                )
                if result_y is None:
                    result_y = lfhf[:, None]
                else:
                    result_y = np.column_stack((result_y, lfhf))
        if "lfn" in metrics or "hfn" in metrics:
            lf, hf = zip(*[FFT.calc_lf_hf(row) for row in intervals])
            lf = np.array(lf)
            hf = np.array(hf)
            lfhf = lf + hf
            if "lfn" in metrics:
                lfn = lf / lfhf
                if result_y is None:
                    result_y = lfn[:, None]
                else:
                    result_y = np.column_stack((result_y, lfn))
            if "hfn" in metrics:
                hfn = hf / lfhf
                if result_y is None:
                    result_y = hfn[:, None]
                else:
                    result_y = np.column_stack((result_y, hfn))

        # result_y = [[self.calc_sdnn(row), self.calc_rmssd(row)] for row in intervals]
        # result_y = np.array(result_y)
        return X, result_y

    def extract_piotr_loaded_peaks(
        self, window_size, metrics: list = ["SDNN"], stride: int = None
    ):
        metrics = [s.lower() for s in metrics]
        X, _, y = self.extract_windows_loaded_peaks(window_size)
        intervals = [np.diff(row) / self.frequency for row in y]
        result_y = None
        if "sdnn" in metrics:
            sdnn = np.array([self.calc_sdnn(row) for row in intervals])
            if result_y is None:
                result_y = sdnn[:, None]
            else:
                result_y = np.column_stack((result_y, sdnn))
        if "rmssd" in metrics:
            rmssd = np.array([self.calc_rmssd(row) for row in intervals])
            if result_y is None:
                result_y = rmssd[:, None]
            else:
                result_y = np.column_stack((result_y, rmssd))
        if "lf" in metrics or "hf" in metrics or "lf/hf" in metrics:
            lf, hf = zip(*[FFT.calc_lf_hf(row) for row in intervals])
            lf = np.array(lf)
            hf = np.array(hf)
            lf_hf_sum = lf + hf  # Sum of lf and hf

            if "lf" in metrics:
                # Calculate normalized lf
                normalized_lf = np.divide(lf, lf_hf_sum, out=np.zeros_like(lf, dtype=float), where=lf_hf_sum != 0)
                if result_y is None:
                    result_y = normalized_lf[:, None]
                else:
                    result_y = np.column_stack((result_y, normalized_lf))

            if "hf" in metrics:
                # Calculate normalized hf
                normalized_hf = np.divide(hf, lf_hf_sum, out=np.zeros_like(hf, dtype=float), where=lf_hf_sum != 0)
                if result_y is None:
                    result_y = normalized_hf[:, None]
                else:
                    result_y = np.column_stack((result_y, normalized_hf))

            if "lf/hf" in metrics:
                # Calculate lf/hf ratio
                lfhf = np.divide(lf, hf, out=np.zeros_like(lf, dtype=float), where=hf != 0)
                if result_y is None:
                    result_y = lfhf[:, None]
                else:
                    result_y = np.column_stack((result_y, lfhf))
        # if "lf" in metrics or "hf" in metrics or "lf/hf" in metrics:
        #         lf, hf = zip(*[FFT.calc_lf_hf(row) for row in intervals])
        #         if "lf" in metrics:
        #             if result_y is None:
        #                 result_y = lf[:, None]
        #             result_y = np.column_stack((result_y, lf))
        #         if "hf" in metrics:
        #             if result_y is None:
        #                 result_y = hf[:, None]
        #             result_y = np.column_stack((result_y, hf))
        #         if "lf/hf" in metrics:
        #             if result_y is None:
        #                 result_y = lf[:, None]
        #             lfhf = np.divide(lf, hf, out=np.zeros_like(lf, dtype=float), where=hf != 0)
        #
        #             result_y = np.column_stack((result_y, lfhf))
            if "lf" in metrics:
                if result_y is None:
                    result_y = lf[:, None]
                result_y = np.column_stack((result_y, lf))
            if "hf" in metrics:
                if result_y is None:
                    result_y = hf[:, None]
                result_y = np.column_stack((result_y, hf))
            if "lf/hf" in metrics:
                if result_y is None:
                    result_y = lf[:, None]
                lfhf = np.divide(
                    lf, hf, out=np.zeros_like(lf, dtype=float), where=hf != 0
                )

                result_y = np.column_stack((result_y, lfhf))

        # result_y = [[self.calc_sdnn(row), self.calc_rmssd(row)] for row in intervals]
        # result_y = np.array(result_y)
        return X, result_y

    def extract_windows_detected_peaks(self, window_size):
        return EcgData.extract_windows_peaks(
            self.__raw_data, self.__r_peaks_ind, window_size
        )

    def extract_windows_loaded_peaks(self, window_size):
        # filtered_values = PanTompkins.bandpass_filter(
        #     self.__raw_data[:, 1], self.frequency
        # )
        # data = np.column_stack((self.__raw_data[:, 0], filtered_values))
        return EcgData.extract_windows_peaks(
            self.__raw_data, self.__loaded_r_peaks_ind, window_size
        )
        # return EcgData.extract_windows_peaks(
        #     data, self.__loaded_r_peaks_ind, window_size
        # )
        """
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
        """

    @staticmethod
    def extract_windows_peaks(data, peaks_ind, window_size):
        win_count = int(len(data) / window_size)

        X_train = np.zeros((win_count, window_size), dtype=np.float64)
        y_train = np.zeros((win_count, window_size))
        R_per_w = []

        normalize = partial(processing.normalize_bound, lb=-1, ub=1)

        for i in range(win_count):
            # for i in tqdm(range(win_count)):
            win_start = i * window_size
            end = win_start + window_size
            r_peaks_ind = np.where((peaks_ind >= win_start) & (peaks_ind < end))
            R_per_w.append(peaks_ind[r_peaks_ind] - win_start)

            for j in peaks_ind[r_peaks_ind]:
                r = int(j) - win_start
                y_train[i, r - 2 : r + 3] = 1

            if data[win_start:end][1].any():
                X_train[i:] = np.squeeze(
                    np.apply_along_axis(normalize, 0, data[win_start:end, 1])
                )
            else:
                X_train[i, :] = data[win_start:end, 1].T

        X_train = np.expand_dims(X_train, axis=2)

        y_train = np.expand_dims(y_train, axis=2)

        return X_train, y_train, R_per_w

    # zwrca okna testowe z indeksami peakow
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

    # eksportuje okna traningowe z obliczonymi wartoscami parametrow takich sdnn czy rmssd
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
            X_train[i] = self.rr_intervals[win_start:win_end, 1]
            # X_train[i] = self.__r_peaks[win_start:win_end, 1]
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
                    
            if "lfn" in metrics or "hfn" in metrics:
                lf, hf = FFT.calc_lf_hf(X_train[i])
                lfhf = lf + hf
                if "lfn" in metrics:
                    y_train[i, metrics.index("lfn")] = lf/lfhf
                if "hfn" in metrics:
                    y_train[i, metrics.index("hfn")] = hf/lfhf
                

            # y_train[i] = (EcgData.calc_sdnn(X_train[i]), EcgData.calc_rmssd(X_train[i]), lf, hf)

        return X_train, y_train

    # liczy cuzlosc precyzje i f1 score chceck detected peaks
    def check_detected_peaks(self):
        intersection1 = np.intersect1d(self.__r_peaks_ind, self.__loaded_r_peaks_ind)
        test = PanTompkins.refine_peak_positions(
            self.raw_data[:, 1], self.__loaded_r_peaks_ind, 20
        )
        test = self.refined_loaded_peaks_ind
        # test = self.__loaded_r_peaks_ind
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
        try:
            recall = TP / (TP + FN)
            prec = TP / (TP + FP)
            f1 = 2 * prec * recall / (prec + recall)
            print("Real values comparison")
            print(f"TP: {TP}")
            print(f"FP: {FP}")
            print(f"FN: {FN}")
            print(f"Recall: {recall}")
            print(f"Precision: {prec}")
            print(f"F1-score: {f1}")
        except:
            pass
        return TP, FP, FN

    def compare_with_Pan_Tompkins(self):
        pn_finder = PanTompkinsFinder()
        peaks = pn_finder.find_r_peaks_ind(self.__raw_data[:, 1], self.frequency)
        self.__loaded_r_peaks_ind = peaks
        # self.__
        refined = self.refined_loaded_peaks_ind
        test = np.setdiff1d(peaks, self.__r_peaks_ind)
        print(test)
        intersection = np.intersect1d(self.__r_peaks_ind, refined)
        TP = intersection.size
        FP = np.setdiff1d(self.__r_peaks_ind, intersection).size
        FN = peaks.size - intersection.size
        recall = None
        prec = None
        f1 = None
        try:
            recall = TP / (TP + FN)
            prec = TP / (TP + FP)
            f1 = 2 * prec * recall / (prec + recall)
        except:
            print("Div by 0")
        print("Pan-Tomkins comparison")
        print(f"TP: {TP}")
        print(f"FP: {FP}")
        print(f"FN: {FN}")
        print(f"Recall: {recall}")
        print(f"Precision: {prec}")
        print(f"F1-score: {f1}")
        return TP, FP, FN

    # zwraca liste zaladoweanych peakow ktroe sa w odlegosci maksymalnie 50ms nieuzwyana
    @property
    def refined_loaded_peaks_ind(self):
        max_distance = int(0.03 * self.frequency)
        # distances = []

        # Refine R peaks indices
        result_indexes = np.array(
            [
                val
                for idx, val in enumerate(self.__r_peaks_ind)
                if any(abs(val - a1) <= max_distance for a1 in self.loaded_r_peak_ind)
            ]
        )

        # Calculate distances
        # for val in result_indexes:
        #     closest_peak = min(self.loaded_r_peak_ind, key=lambda a1: abs(val - a1))
        #     distances.append(abs(val - closest_peak))
            # print(distances[-1])

        # Compute average distance
        # average_distance = np.mean(distances) if distances else 0
        # print(average_distance/self.frequency*1000)

        return result_indexes  # , average_distance
        # return PanTompkins.refine_peak_positions(
        #     self.raw_data[:, 1], self.__loaded_r_peaks_ind, 25
        # )

import numpy as np

import FFT
import PanTompkins
import NAME_THIS_MODULE_YOURSELF_PIOTER
from PanTompkins import derivative_filter
from config import *
from dataSenderEmulator import read_csv
import scipy.signal as signal


class EcgData:

    @property
    def r_peaks(self):
        self.__refresh_if_dirty()
        return self.__r_peaks

    @property
    def hr_ups(self):
        self.__refresh_if_dirty()
        return self.__hr_ups

    @property
    def hr_downs(self):
        self.__refresh_if_dirty()
        return self.__hr_downs

    @property
    def hr(self):
        self.__refresh_if_dirty()
        return self.__hr

    @property
    def inhalation_starts_moments(self):
        self.__refresh_if_dirty()
        return self.__inhalation_starts_moments

    @property
    def exhalation_starts_moments(self):
        self.__refresh_if_dirty()
        return self.__exhalation_starts_moments

    @property
    def r_peaks_filtered(self):
        self.__refresh_if_dirty()
        return self.__r_peaks_filtered

    @property
    def r_peaks_piotr(self):
        self.__refresh_if_dirty()
        return self.__r_peaks_piotr

    @property
    def rr_intervals(self):
        self.__refresh_if_dirty()
        return self.__rr_intervals

    @property
    def mean_rr(self):
        self.__refresh_if_dirty()
        return self.__mean_rr

    @property
    def sdnn(self):
        self.__refresh_if_dirty()
        return self.__sdnn

    @property
    def rmssd(self):
        self.__refresh_if_dirty()
        return self.__rmssd

    @property
    def pnn50(self):
        self.__refresh_if_dirty()
        return self.__pnn50

    def __init__(self, frequency):
        self.frequency = frequency
        self.raw_data = np.empty((0, 2))
        self.filtered_data = np.empty((0, 2))
        self.__r_peaks = np.empty(0)
        self.__hr_ups = np.empty(0)
        self.__hr_downs = np.empty(0)
        self.__r_peaks_filtered = np.empty(0)
        self.__r_peaks_piotr = np.empty(0)
        self.__rr_intervals = np.empty(0)
        self.__hr = np.empty((0, 2))
        self.hr_filtered = np.empty((0, 2))


        self.__mean_rr = -1
        self.__sdnn = -1
        self.__rmssd = -1
        self.__pnn50 = -1
        self.__is_dirty = False

        self.frequency = frequency
        self.lowcut = 5
        self.highcut = 18
        self.buffer_size = 4000
        self.data_buffer = np.zeros(self.buffer_size)  # Inicjalizacja bufora o stałej wielkości
        self.b, self.a = self.create_bandpass_filter(self.lowcut, self.highcut,4)
        self.zi = signal.lfilter_zi(self.b, self.a)  # Stan filtra dla lfilter

        self.lowcutHR = 5
        self.highcutHR = 15
        self.buffer_sizeHR = 200
        self.data_bufferHR = np.zeros(self.buffer_sizeHR)  # Inicjalizacja bufora o stałej wielkości
        self.bHR, self.aHR = self.create_bandpass_filter1(self.lowcutHR, self.highcutHR,4)
        self.ziHR = signal.lfilter_zi(self.bHR, self.aHR)  # Stan filtra dla lfilter

        self.__inhalation_starts_moments = np.empty((0,2))
        self.__exhalation_starts_moments = np.empty((0,2))

        self.wdechy = []  # Lista par (początek wdechu, koniec wdechu)
        self.wydechy = []  # Lista par (początek wydechu, koniec wydechu)

       # self.bHR, self.aHR = signal.butter(2, [0.1, 40], btype='bandpass', fs=100)  # Przykładowy filtr
        # self.ziHR = signal.lfilter_zi(self.bHR, self.aHR) * 0  # Inicjalizacja współczynników stanu filtra

        return

    def update_and_filter(self, new_data):
        # Aktualizacja bufora danych
        self.data_buffer = np.concatenate((self.data_buffer[len(new_data):], new_data))

        filtered_signal, self.zi = signal.lfilter(self.b, self.a, self.data_buffer,
                                                  zi=self.zi * self.data_buffer[0])
        return filtered_signal[-len(new_data):], self.zi

    def update_hr(self, new_data):
        # Aktualizacja bufora poprzez usunięcie najstarszych elementów, jeśli bufor osiągnął maksymalny rozmiar
        if len(self.data_bufferHR) >= self.buffer_sizeHR:
            self.data_bufferHR = np.roll(self.data_bufferHR, -1)
            self.data_bufferHR[-1] = new_data

    def filter_hr(self):
        # Przefiltrowanie sygnału
        #filtered_signal, self.ziHR = signal.lfilter(self.bHR, self.aHR, self.data_bufferHR, zi=self.ziHR)
        filtered_signal = signal.filtfilt(self.bHR, self.aHR, self.__hr[:, 1])
        #return filtered_signal[-len(new_data):]  # Zwróć tylko najnowsze przefiltrowane wartości
        peaks_with_timestamps = np.column_stack((self.__hr[:, 0], filtered_signal))
        return peaks_with_timestamps # Zwróć tylko najnowsze przefiltrowane wartości

    def derivative_filter(self, sig):
        return np.diff(sig, prepend=0)

    def square1(self, sig):
        return sig ** 2

    def moving_window_integration(self, sig, window_size):
        return np.convolve(sig, np.ones(window_size) / window_size, mode="same")

    def load_csv_data(self, path):
        csv_data = np.array(read_csv(path))
        if NORMALIZED_TEST_DATA_TIME:
            csv_data[:, 0] -= csv_data[0, 0]
        if NEGATE_INCOMING_DATA:
            csv_data[:, 1] *= -1
        csv_data[:, 0] /= TIME_SCALE_FACTOR

        filtered_signal = signal.filtfilt(self.b, self.a, csv_data[:, 1])

        # Zapisanie przefiltrowanych danych do raw_data
        self.raw_data = np.column_stack((csv_data[:, 0], csv_data[:, 1]))
        print(len(self.raw_data))




        self.filtered_data = np.column_stack((csv_data[:, 0], filtered_signal))
        print(len(self.filtered_data))


        # self.__is_dirty = True
        self.__refresh_data()

        FFT.fft(self.__rr_intervals)


        #self.hr_filtered = self.filter_hr()



    def print_data(self):
        mean_rr = round(self.mean_rr * 1e3, 2) if self.mean_rr is not None else None
        sdnn = round(self.sdnn, 2) if self.sdnn is not None else None
        rmssd = round(self.rmssd, 2) if self.rmssd is not None else None
        pnn50 = round(self.pnn50, 2) if self.rmssd is not None else None

        print("poczatki wdechu: ")
        print(self.__inhalation_starts_moments)
        print("poczatki wydechu: ")
        print(self.__exhalation_starts_moments)

        self.__find_inex_haletion_moments()

        # Wyświetlenie wyników końcowych
        print("\nLista wdechów (początek, koniec, czas trwania):")
        for start, end, czas, roznicaHR in self.wdechy:
            print(f"Od {start} do {end}, czas trwania: {czas}, roznica hr: {roznicaHR}")

        print("\nLista wydechów (początek, koniec, czas trwania):")
        for start, end, czas, roznicaHR in self.wydechy:
            print(f"Od {start} do {end}, czas trwania: {czas}, roznica hr: {roznicaHR}")

        # Obliczenie i wyświetlenie średnich czasów trwania wdechu i wydechu
        if self.wdechy:
            srednia_wdechu = sum(czas for _, _, czas, _ in self.wdechy) / len(self.wdechy)
            print(f"\nŚredni czas trwania wdechu: {srednia_wdechu:.2f}")

        if self.wydechy:
            srednia_wydechu = sum(czas for _, _, czas, _ in self.wydechy) / len(self.wydechy)
            print(f"Średni czas trwania wydechu: {srednia_wydechu:.2f}")

        print(f"rsa index wynosi: {self.calculate_rsa_index()}")

        print(f"rsa index wzgledny wynosi: {self.calculate_relative_rsa_index_()}")

        print(f"mean heart rate wynosi: {self.calculate_mean_heart_rate_diff()}")

        print(f"średnie tętno wynosi: {self.calculate_mean_heart_rate()}")
        # print("ECG peaks---------------")
        # print (self.r_peaks)
        # print("ECG intervals---------------")
        # print (self.rr_intervals)
        # print("ECG hr---------------")
        # print (self.hr)
        # print("\n*********************\n")
        # if mean_rr is not None:
        #    print(f"Mean RR: {mean_rr} ms")
        # if sdnn is not None:
        #     print(f"SDNN: {sdnn}")
        # if rmssd is not None:
        #     print(f"RMSSD: {rmssd}")
        # if pnn50 is not None:
        #     print(f"PNN50: {pnn50}%")
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
        if isinstance(x, list) and isinstance(y, list):
            new_data = np.array(y)  # Przekształć wartości sygnału na tablicę
            filtered_data, self.zi = self.update_and_filter(new_data)  # Filtrowanie z zachowaniem stanu
            filtered_data, self.zi = self.update_and_filter(new_data)  # Filtrowanie z zachowaniem stanu
            self.filtered_data = np.concatenate((self.filtered_data, np.column_stack((x, filtered_data))))
            self.raw_data = np.concatenate((self.raw_data, np.column_stack((x, y))))
        else:
            # Dla pojedynczej próbki
            new_data = np.array([y])  # Przekształć wartość sygnału na tablicę
            filtered_data = self.update_and_filter(new_data)
            self.filtered_data = np.append(self.filtered_data, [[x, filtered_data[0]]], axis=0)
            self.raw_data = np.append(self.raw_data, [[x, y]], axis=0)

        self.__set_dirty()
        return

    def create_bandpass_filter(self, lowcut, highcut, order):
        nyquist = 0.5 * self.frequency
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype="band")
        return b, a

    def calculate_rsa_index(self):
        hr_values = self.__hr[:, 1]
        return np.max(hr_values) - np.min(hr_values)

    def calculate_mean_heart_rate(self):
        hr_values = self.__hr[:, 1]
        return np.mean(hr_values) if len(hr_values) > 0 else 0

    def calculate_mean_heart_rate_diff(self):
        total_heart_rate_diff = 0
        number_of_cycles = len(self.wdechy) + len(self.wydechy)

        for _, _, _, diff in self.wdechy:
            total_heart_rate_diff += abs(diff)

        for _, _, _, diff in self.wydechy:
            total_heart_rate_diff += abs(diff)

        mean_absolute_difference = total_heart_rate_diff / number_of_cycles if number_of_cycles > 0 else 0

        return mean_absolute_difference

    def calculate_relative_rsa_index_(self):
        max_relative_rsa_diff = max(
            max(abs(diff) for _, _, _, diff in self.wdechy),
            max(abs(diff) for _, _, _, diff in self.wydechy)
        )
        return max_relative_rsa_diff

    def create_bandpass_filter1(self, lowcut, highcut, order):
        nyquist = 0.5 * self.frequency
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype="band")
        return b, a

    def __set_dirty(self):
        self.__is_dirty = True
        return

    def __refresh_if_dirty(self):
        if self.__is_dirty == False:
            return

        self.__refresh_data()
        self._is_dirty = False
        return

    def __refresh_data(self):
        # self.__find_r_peaks()

        self.__find_new_r_peaks()
        self.__find_new_r_peaks_filtered()
        self.__find_r_peaks_piotr()
        self.__calc_rr_intervals()
        self.__calc_mean_rr()
        self.__calc_sdnn()
        self.__calc_rmssd()
        self.__calc_pnn50()
        self.__calc_hr()
        if(len(self.__hr) > 28):
            self.hr_filtered = self.filter_hr()
            ups = self.__find_hr_ups()
            self.__exhalation_starts_moments = np.column_stack((
                ups[:, 0] - self.__hr[0][0],  # Znormalizowane czasy (pierwsza kolumna)
                ups[:, 1]  # Oryginalne wartości (druga kolumna)
            ))

            downs = self.__find_hr_downs()
            self.__inhalation_starts_moments = np.column_stack((
                downs[:, 0] - self.__hr[0][0],  # Znormalizowane czasy (pierwsza kolumna)
                downs[:, 1]  # Oryginalne wartości (druga kolumna)
            ))

            # Zakładając posortowane listy minima (wdechy) i maksima (wydechy)





        return

    def __find_hr_downs(self):
        self.__hr_downs = self.find_hr_downs(self.__hr, self.frequency, self.lowcutHR, self.highcutHR)
        return self.__hr_downs

    def __find_hr_ups(self):
        self.__hr_ups = self.find_hr_ups(self.__hr, self.frequency, self.lowcutHR, self.highcutHR)
        return self.__hr_ups

    def __find_r_peaks(self):
        self.__r_peaks = self.find_r_peaks(self.raw_data, self.frequency)
        return self.__r_peaks

    def __find_r_peaks_filtered(self):
        self.__r_peaks_filtered = self.find_r_peaks(self.filtered_data, self.frequency)
        return self.__r_peaks_filtered

    def __find_inex_haletion_moments(self):
        minima = self.__inhalation_starts_moments[:,0]
        maksima = self.__exhalation_starts_moments[:,0]
        minimaHR = self.__inhalation_starts_moments[:,1]
        maksimaHR = self.__exhalation_starts_moments[:,1]
        self.wdechy.clear()
        self.wydechy.clear()

        i, j = 0, 0  # Indeksy do list minimów i maksimów

        while i < len(minima) and j < len(maksima):
            if minima[i] < maksima[j]:
                if not (i + 1 < len(minima) and minima[i + 1] < maksima[j]):
                    czas = maksima[j] - minima[i]
                    roznica_hr = maksimaHR[j] - minimaHR[i]
                    self.wdechy.append((minima[i], maksima[j], czas, roznica_hr))
                i += 1
            else:
                if not (j + 1 < len(maksima) and maksima[j + 1] < minima[i]):
                    czas = minima[i] - maksima[j]
                    roznica_hr = minimaHR[i] - maksimaHR[j]
                    self.wydechy.append((maksima[j], minima[i], czas, roznica_hr))
                j += 1

    @staticmethod
    def find_r_peaks(data: np.ndarray, frequency: int) -> np.ndarray:
        return PanTompkins.find_r_peaks(data, frequency)

    @staticmethod
    def find_hr_ups(data: np.ndarray, frequency: int, a: float, b: float) -> np.ndarray:
        return PanTompkins.find_hr_peaks(data, frequency, a,b, 5)

    @staticmethod
    def find_hr_downs(data: np.ndarray, frequency: int, a: float, b: float) -> np.ndarray:

        return PanTompkins.find_hr_peaks(data, frequency, a,b, 5, -1)

    def __find_new_r_peaks(self):
        if len(self.raw_data) < self.frequency * 2:
            return
        if len(self.__r_peaks) == 0:
            self.__find_r_peaks()
            return

        index = -1
        for i in range(len(self.raw_data) - 1, -1, -1):
            if self.raw_data[i][0] < self.__r_peaks[-1][0]:
                index = i
                break
        # offset = int(self.__mean_rr/2*self.frequency)
        # new_peaks = PanTompkins.find_r_peaks(self.raw_data[-(index+offset):], self.frequency)
        new_peaks = PanTompkins.find_r_peaks(self.raw_data[-index:], self.frequency)

        start_index = -1
        for i in range(len(new_peaks)):
            if self.__r_peaks[-1][0] < new_peaks[i][0]:
                start_index = i
                break

        if -1 == start_index:
            return

        rr_intervals = self.calc_rr_intervals(self.__r_peaks[-4:])
        last_3_mean_rr = self.calc_mean_rr(rr_intervals)

        if self.raw_data[-1][0] - new_peaks[-1][0] > last_3_mean_rr:
            end_index = len(new_peaks - 1)
        else:
            end_index = -1

        nowe_peaki = new_peaks[start_index-1:end_index]

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
            x = x1-x0
            y = 60/x

            # Dodajemy nową wartość do bufora
  #          new_data = np.array([y])  # Przekształć wartość sygnału na tablicę
    #        self.update_hr(new_data)  # Filtracja nowej wartości

    #        if (len(self.__hr) > 15):
        #        filtered_data = self.filter_hr(new_data)  # Filtracja nowej wartości
                # Dodaj przefiltrowany wynik do przechowywanej tablicy wyników
                #self.hr_filtered = np.append(self.hr_filtered, [[x1, filtered_data[0]]], axis=0)
      #          self.hr_filtered = filtered_data
            # else:
            #     self.hr_filtered = np.append(self.hr_filtered, [[x1, new_data[0]]], axis=0)
        self.__r_peaks = np.vstack((self.__r_peaks, new_peaks[start_index:end_index]))

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
        new_peaks = PanTompkins.find_r_peaks(self.filtered_data[-index:], self.frequency)

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

        self.__r_peaks_filtered = np.vstack((self.__r_peaks_filtered, new_peaks[start_index:end_index]))

    def __find_r_peaks_piotr(self):
        self.__r_peaks_piotr = NAME_THIS_MODULE_YOURSELF_PIOTER.find_r_peaks_piotr(
            self.raw_data
        )
        return self.__r_peaks_piotr

    def __calc_rr_intervals(self):
        self.__rr_intervals = self.calc_rr_intervals(self.__r_peaks)
        return self.__rr_intervals

    @staticmethod
    def calc_rr_intervals(r_peaks: np.ndarray) -> np.ndarray:
        if len(r_peaks) < 2:
            return np.empty(0)

        rr_intervals = np.diff([peak[0] for peak in r_peaks])
        return rr_intervals

    def __calc_mean_rr(self):
        self.__mean_rr = self.calc_mean_rr(self.__rr_intervals)
        return self.__mean_rr

    @staticmethod
    def calc_mean_rr(rr_intervals: np.ndarray) -> np.ndarray:
        return np.mean(rr_intervals) if len(rr_intervals) > 0 else None

    def __calc_sdnn(self):
        self.__sdnn = self.calc_sdnn(self.__rr_intervals)
        return self.__sdnn

    @staticmethod
    def calc_sdnn(rr_intervals: np.ndarray) -> float:
        return np.std(rr_intervals) * 1e3 if len(rr_intervals) > 0 else None

    def __calc_hr(self):
        self.__hr = self.calc_hr(self.__r_peaks)
        return self.__hr

    @staticmethod
    def calc_hr(r_peaks: np.ndarray) -> np.ndarray:
        # Sprawdź, czy mamy wystarczającą liczbę pików do obliczenia odstępów RR
        if len(r_peaks) < 2:
            return np.array([])  # Zwróć pustą tablicę, jeśli brak wystarczającej liczby pików

        # Oblicz odstępy RR na podstawie różnic czasowych między kolejnymi pikami
        rr_intervals = np.diff(r_peaks[:, 0])

        # Czas znacznika przypisany do heart_rate to czas końca każdego odstępu RR
        timestamps = r_peaks[1:, 0]

        # Oblicz wartości tętna jako 60 / RR
        heart_rates = 60 / rr_intervals

        # Połącz znaczniki czasowe i wartości tętna w kolumny
        result = np.column_stack((timestamps, heart_rates))

        # if len(result) > 10:
        #     result2 = PanTompkins.filter_ecg_with_timestamps(result, 100)
        #     print(result2)
        #     return result2
        return result
    def __calc_rmssd(self):
        self.__rmssd = self.calc_rmssd(self.__rr_intervals)
        return self.__rmssd

    @staticmethod
    def calc_rmssd(rr_intervals: np.ndarray) -> float:
        diff_rr_intervals = np.diff(rr_intervals)
        return (
            np.sqrt(np.mean(diff_rr_intervals ** 2)) * 1e3
            if len(diff_rr_intervals) > 0
            else None
        )

    def __calc_pnn50(self):
        self.__pnn50 = self.calc_pnn50(self.__rr_intervals)
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

    pass


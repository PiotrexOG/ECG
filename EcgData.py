import numpy as np
import PanTompkins
import NAME_THIS_MODULE_YOURSELF_PIOTER
from config import *
from dataSenderEmulator import read_csv
from Finders.RPeaksFinder import RPeaksFinder
import threading

from functools import partial
from wfdb import processing
import wfdb
from tqdm import tqdm


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

    def __init__(self, frequency, r_peaks_finder: RPeaksFinder):
        self.frequency = frequency
        self.__r_peaks_finder = r_peaks_finder
        self.__raw_data = np.empty((0, 2))
        self.__lock = threading.Lock()
        self.loaded___raw_data = np.empty(0)
        self.__r_peaks = np.empty(0)
        self.__r_peaks_ind = np.empty(0)
        self.__loaded_r_peaks_ind = np.empty(0)
        self.__r_peaks_piotr = np.empty(0)
        self.__rr_intervals = np.empty(0)
        self.__mean_rr = -1
        self.__sdnn = -1
        self.__rmssd = -1
        self.__pnn50 = -1
        self.__is_dirty = False
        return

    def load_csv_data_with_timestamps(self, path):
        csv_data = np.array(read_csv(path))
        if NORMALIZED_TEST_DATA_TIME:
            csv_data[:, 0] -= csv_data[0, 0]
        if NEGATE_INCOMING_DATA:
            csv_data[:, 1] *= -1
        csv_data[:, 0] /= TIME_SCALE_FACTOR
        self.__raw_data = csv_data
        # self.__is_dirty = True
        self.__refresh_data()

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
        # self.__loaded_r_peaks_ind = self.q
        # self.__loaded_r_peaks_ind = EcgData.remove_duplicates_and_adjacent(self.q, self.frequency)
        self.__loaded_r_peaks_ind = EcgData.filter_similar_elements(self.q, self.frequency)

        
        timestamps = np.array([i / sample_rate for i in range(len(ecg_signal))])
        
        self.raw_data = np.column_stack((timestamps, ecg_signal))

    @staticmethod
    def remove_duplicates_and_adjacent(array, frequency):
        # Remove duplicates and sort
        array = np.unique(array)
        result = []
        
        
        
        # for num in array:
        #     if not result or num != result[-1] + 1:
        #         result.append(num)
        
        for i in range(len(array)-1, -1, -1):
            # Add to result if it doesn't differ by 1 from the last added number
            if not result or array[i] < result[-1] - int(0.05* frequency):
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
            if sorted_arr[i] - sorted_arr[i-1] <= int(0.05* frequency):
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
        output = str()
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

    def push_raw_data(self, x, y):
        self.__lock.acquire()
        try:
            if isinstance(x, list) and isinstance(y, list):
                self.__raw_data = np.concatenate(
                    (self.__raw_data, np.column_stack((x, y)))
                )
            else:
                self.__raw_data = np.append(self.__raw_data, [[x, y]], axis=0)
        finally:
            self.__lock.release()

        self.__set_dirty()
        return

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
        self.__find_r_peaks_piotr()
        self.__calc_rr_intervals()
        self.__calc_mean_rr()
        self.__calc_sdnn()
        self.__calc_rmssd()
        self.__calc_pnn50()
        return

    def __find_r_peaks(self):
        self.__r_peaks = self.__r_peaks_finder.find_r_peaks_values_with_timestamps(
            self.__raw_data, self.frequency
        )
        self.__r_peaks_ind = self.__r_peaks_finder.find_r_peaks_ind(
            self.__raw_data[:, 1], self.frequency
        )
        return self.__r_peaks

    # @staticmethod
    # def find_r_peaks(data: np.ndarray, frequency: int) -> np.ndarray:
    #     return PanTompkins.find_r_peaks_values_with_timestamps(data, frequency)

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

        self.__r_peaks = np.vstack((self.__r_peaks, new_peaks[start_index:end_index]))

        # self.__r_peaks_ind = np.hstack((self.__r_peaks_ind, new_peaks_ind[start_index:end_index]))

        if len(self.__r_peaks) != len(self.__r_peaks_ind):
            return
            raise Exception("The arrays are of different lengths!")

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

    def __calc_rmssd(self):
        self.__rmssd = self.calc_rmssd(self.__rr_intervals)
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
                (self.__loaded_r_peaks_ind >= win_start) & (self.__loaded_r_peaks_ind < end)
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

    def extract_hrv_windows_with_loaded_peaks(self, window_size):
        win_count = int(len(self.__rr_intervals) / window_size)
        r_peaks = self.raw_data[self.__loaded_r_peaks_ind]
        rr_intervals = np.diff([peak[0] for peak in r_peaks])

        X_train = np.zeros((win_count, window_size), dtype=np.float64)
        # y_train = np.zeros((win_count, 3))
        y_train = np.zeros((win_count, 2))

        for i in range(win_count):
            win_start = i * window_size
            win_end = win_start + window_size
            # X_train[i] = self.rr_intervals[win_start:win_end]
            X_train[i] = rr_intervals[win_start:win_end]
            # y_train[i] = (EcgData.calc_sdnn(X_train[i]), EcgData.calc_rmssd(X_train[i]), 0)
            y_train[i] = (EcgData.calc_sdnn(X_train[i]), EcgData.calc_rmssd(X_train[i]))

        return X_train, y_train

    def extract_hrv_windows_with_detected_peaks(self, window_size):
        win_count = int(len(self.__rr_intervals) / window_size)

        X_train = np.zeros((win_count, window_size), dtype=np.float64)
        # y_train = np.zeros((win_count, 3))
        y_train = np.zeros((win_count, 2))

        for i in range(win_count):
            win_start = i * window_size
            win_end = win_start + window_size
            # X_train[i] = self.rr_intervals[win_start:win_end]
            X_train[i] = self.__rr_intervals[win_start:win_end]
            # y_train[i] = (EcgData.calc_sdnn(X_train[i]), EcgData.calc_rmssd(X_train[i]), 0)
            y_train[i] = (EcgData.calc_sdnn(X_train[i]), EcgData.calc_rmssd(X_train[i]))

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
        sens = TP/(TP+FN)
        prec = TP/(TP+FP)
        f1 = 2*prec*sens/(prec+sens)
        # print(f"Sensitivity {sens}")
        # print(f"Precision {prec}")
        print(f"F1-score {f1}")
        pass

    @property
    def refined_loaded_peaks_ind(self, max_distance = -1):
        if max_distance == -1:
            max_distance = int(0.05* self.frequency)
        result_indexes = np.array(
            [
                val
                for idx, val in enumerate(self.__r_peaks_ind)
                if any(abs(val - a1) <= max_distance for a1 in self.loaded_r_peak_ind)
            ]
        )
        return result_indexes
        # return PanTompkins.refine_peak_positions(
        #     self.raw_data[:, 1], self.__loaded_r_peaks_ind, 25
        # )

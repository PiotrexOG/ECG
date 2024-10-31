import numpy as np
import PanTompkins
import NAME_THIS_MODULE_YOURSELF_PIOTER
from config import *
from dataSenderEmulator import read_csv


from functools import partial
from wfdb import processing
from tqdm import tqdm


class EcgData:

    @property
    def r_peaks(self):
        self.__refresh_if_dirty()
        return self.__r_peaks

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
        self.__r_peaks = np.empty(0)
        self.__r_peaks_ind = np.empty(0)
        self.__r_peaks_piotr = np.empty(0)
        self.__rr_intervals = np.empty(0)
        self.__mean_rr = -1
        self.__sdnn = -1
        self.__rmssd = -1
        self.__pnn50 = -1
        self.__is_dirty = False
        return

    def load_csv_data(self, path):
        csv_data = np.array(read_csv(path))
        if NORMALIZED_TEST_DATA_TIME:
            csv_data[:, 0] -= csv_data[0, 0]
        if NEGATE_INCOMING_DATA:
            csv_data[:, 1] *= -1
        csv_data[:, 0] /= TIME_SCALE_FACTOR
        self.raw_data = csv_data
        # self.__is_dirty = True
        self.__refresh_data()

    def print_data(self):
        mean_rr = round(self.mean_rr * 1e3, 2) if self.mean_rr is not None else None
        sdnn = round(self.sdnn, 2) if self.sdnn is not None else None
        rmssd = round(self.rmssd, 2) if self.rmssd is not None else None
        pnn50 = round(self.pnn50, 2) if self.rmssd is not None else None
        print("ECG DATA---------------")
        if mean_rr is not None:
            print(f"Mean RR: {mean_rr} ms")
        if sdnn is not None:
            print(f"SDNN: {sdnn}")
        if rmssd is not None:
            print(f"RMSSD: {rmssd}")
        if pnn50 is not None:
            print(f"PNN50: {pnn50}%")
        return

    def push_raw_data(self, x, y):
        if isinstance(x, list) and isinstance(y, list):
            self.raw_data = np.concatenate((self.raw_data, np.column_stack((x, y))))
        else:
            self.raw_data = np.append(self.raw_data, [[x, y]], axis=0)

        self.__set_dirty()
        return

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
        self.__find_r_peaks_piotr()
        self.__calc_rr_intervals()
        self.__calc_mean_rr()
        self.__calc_sdnn()
        self.__calc_rmssd()
        self.__calc_pnn50()
        return

    def __find_r_peaks(self):
        self.__r_peaks = self.find_r_peaks(self.raw_data, self.frequency)
        self.__r_peaks_ind = PanTompkins.find_r_peaks_ind(self.raw_data, self.frequency)
        return self.__r_peaks

    @staticmethod
    def find_r_peaks(data: np.ndarray, frequency: int) -> np.ndarray:
        return PanTompkins.find_r_peaks(data, frequency)

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
        new_peaks_ind = PanTompkins.find_r_peaks_ind(
            self.raw_data[-index:], self.frequency
        )

        start_index = -1
        for i in range(len(new_peaks)):
            if self.__r_peaks[-1][0] < new_peaks[i][0]:
                start_index = i
                break

        if -1 == start_index:
            return

        rr_intervals = self.calc_rr_intervals(self.__r_peaks[-4:])
        last_3_mean_rr = self.calc_mean_rr(rr_intervals)

        if self.raw_data[-1][0] - new_peaks[-1][0] > last_3_mean_rr / 4:
            end_index = len(new_peaks - 1)
        else:
            end_index = -1

        self.__r_peaks = np.vstack((self.__r_peaks, new_peaks[start_index:end_index]))
        self.__r_peaks_ind = np.vstack(
            (self.__r_peaks_ind, new_peaks_ind[start_index:end_index])
        )

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
        win_count = int(len(self.raw_data) / window_size)

        X_train = np.zeros((win_count, window_size), dtype=np.float64)
        y_train = np.zeros((win_count, window_size))
        R_per_w = []

        normalize = partial(processing.normalize_bound, lb=-1, ub=1)

        for i in tqdm(range(win_count)):
            win_start = i * window_size
            end = win_start + window_size
            r_peaks_ind = np.where((self.__r_peaks_ind >= win_start) & (self.__r_peaks_ind < end))
            R_per_w.append(self.__r_peaks_ind[r_peaks_ind]-win_start)

            for j in self.__r_peaks_ind[r_peaks_ind]:
                r = int(j)-win_start
                y_train[i,r-2:r+3] = 1
                
            if self.raw_data[win_start:end][1].any():
                X_train[i:] = np.squeeze(np.apply_along_axis(normalize, 0, self.raw_data[win_start:end, 1]))
            else:
                X_train[i,:] = self.raw_data[win_start:end, 1].T
                
        X_train = np.expand_dims(X_train, axis=2)
    
        y_train = np.expand_dims(y_train, axis=2)
        
        return X_train, y_train, R_per_w
    
    def extract_test_windows(self, win_size, stride):
        normalize = partial(processing.normalize_bound, lb=-1, ub=1)

        signal = np.squeeze(self.raw_data[:, 1])
    
        pad_sig = np.pad(signal, (win_size - stride, win_size), mode='edge')
        
        # Lists of data windows and corresponding indices
        data_windows = []
        win_idx = []

        # Indices for padded signal
        pad_id = np.arange(pad_sig.shape[0])


        # Split into windows and save corresponding padded indices
        for win_id in range(0, len(pad_sig), stride):
            if win_id + win_size < len(pad_sig):
                
                window = pad_sig[win_id:win_id+win_size]
                if window.any():
                    window = np.squeeze(np.apply_along_axis(normalize, 0, window))

                data_windows.append(window)
                win_idx.append(pad_id[win_id:win_id+win_size])


        data_windows = np.asarray(data_windows)
        data_windows = data_windows.reshape(data_windows.shape[0],
                                            data_windows.shape[1], 1)
        win_idx = np.asarray(win_idx)
        win_idx = win_idx.reshape(win_idx.shape[0]*win_idx.shape[1])
        
        return win_idx, data_windows

import numpy as np
import PanTompkins
import NAME_THIS_MODULE_YOURSELF_PIOTER


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
        self.__r_peaks_piotr = np.empty(0)
        self.__rr_intervals = np.empty(0)
        self.__mean_rr = -1
        self.__sdnn = -1
        self.__rmssd = -1
        self.__pnn50 = -1
        self.__is_dirty = False
        return

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
        self.__find_r_peaks()
        self.__find_r_peaks_piotr()
        self.__calc_rr_intervals()
        self.__calc_mean_rr()
        self.__calc_sdnn()
        self.__calc_rmssd()
        self.__calc_pnn50()
        return

    def __find_r_peaks(self):
        self.__r_peaks = PanTompkins.find_r_peaks(self.raw_data, self.frequency)
        return self.__r_peaks

    def __find_r_peaks_piotr(self):
        self.__r_peaks_piotr = NAME_THIS_MODULE_YOURSELF_PIOTER.find_r_peaks_piotr(
            self.raw_data
        )
        return self.__r_peaks_piotr

    def __calc_rr_intervals(self):
        if len(self.__r_peaks) < 2:
            return np.empty(0)

        self.__rr_intervals = np.diff([peak[0] for peak in self.__r_peaks])
        return self.__rr_intervals

    def __calc_mean_rr(self):
        self.__mean_rr = (
            np.mean(self.__rr_intervals) if len(self.__rr_intervals) > 0 else None
        )
        return self.__mean_rr

    def __calc_sdnn(self):
        self.__sdnn = (
            np.std(self.__rr_intervals) * 1e3 if len(self.__rr_intervals) > 0 else None
        )
        return self.__sdnn

    def __calc_rmssd(self):
        diff_rr_intervals = np.diff(self.__rr_intervals)
        self.__rmssd = (
            np.sqrt(np.mean(diff_rr_intervals**2)) * 1e3
            if len(diff_rr_intervals) > 0
            else None
        )
        return self.__rmssd

    def __calc_pnn50(self):
        if len(self.__rr_intervals) < 2:
            self.__pnn50 = None
            return self.__pnn50

        diff_rr_intervals = np.diff(self.__rr_intervals)

        nn50_count = np.sum(np.abs(diff_rr_intervals) > 0.050)  # 50 ms

        self.__pnn50 = (
            (nn50_count / len(diff_rr_intervals)) * 100
            if len(diff_rr_intervals) > 0
            else None
        )

        return self.__pnn50

    pass

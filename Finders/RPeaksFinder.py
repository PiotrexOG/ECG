from abc import *
import numpy as np

class RPeaksFinder():
    
    
    @abstractmethod
    def find_r_peaks_ind(self, ecg_signal, frequency: float):
        pass
    
    
    def find_r_peaks_values_with_timestamps(self, ecg_signal, frequency: float):
        peaks = self.find_r_peaks_ind(ecg_signal[:, 1], frequency)
        if 0 == len(peaks):
            return np.empty((0, 2))
        try:
            peak_values = ecg_signal[peaks, 1]
            peak_timestamps = ecg_signal[peaks, 0]
        except:
            pass
        
        peaks_with_timestamps = np.column_stack((peak_timestamps, peak_values))
        
        return peaks_with_timestamps
    
    def find_r_peaks_values(self, ecg_signal, frequency: float):
        peaks = self.find_r_peaks_ind(ecg_signal, frequency)
        
        peak_values = ecg_signal[peaks, 1]
        return peak_values

    pass
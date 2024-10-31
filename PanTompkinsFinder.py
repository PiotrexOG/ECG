from RPeaksFinder import RPeaksFinder
from abc import *
import PanTompkins

class PanTompkinsFinder(RPeaksFinder):
    
    def find_r_peaks_ind(self, ecg_signal, frequency: float):
        return PanTompkins.find_r_peaks_ind(ecg_signal, frequency)
    
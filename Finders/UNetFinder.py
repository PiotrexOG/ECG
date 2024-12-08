from Finders.RPeaksFinder import RPeaksFinder
from models import sig2sig_unet
from functools import partial
from wfdb import processing
import numpy as np
from nnHelpers import mean_preds, verifier, filter_predictions
from PanTompkins import bandpass_filter

class UNetFinder(RPeaksFinder):

    def __init__(self, model_path, input_size) -> None:
        super().__init__()
        self._win_size = input_size
        model = sig2sig_unet(input_size)
        model.load_weights(model_path)
        self._model = model

    def find_r_peaks_ind(self, ecg_signal, frequency: float, threshold=0.5):
        # print(len(ecg_signal) > 256)
        stride = int(6 / 8 * self._win_size)
        padded_indices, data_windows = self.extract_windows(ecg_signal, stride)
        padded_indices, data_windows = self.extract_windows(ecg_signal, stride, frequency)
        predictions = self._model.predict(data_windows, verbose = 0)
        predictions = mean_preds(
            win_idx=padded_indices,
            preds=predictions,
            orig_len=ecg_signal.shape[0],
            win_size=self._win_size,
            stride=stride,
        )
        filtered_peaks, filtered_proba = filter_predictions(
            signal=ecg_signal, preds=predictions, threshold=threshold, frequency = frequency
        )
        
        filtered_peaks, _ = verifier(ecg_signal, filtered_peaks, filtered_proba, ver_wind = (80/400)*frequency)
        
        if 0 == len(filtered_peaks) or 0 == len(filtered_proba):
            return np.empty(0)
        
        #R_peaks_ver, _ = verifier(ecg_signal, filtered_peaks, filtered_proba, ver_wind=7)
        return filtered_peaks

    def extract_windows(self, ecg_signal, stride, frequency):
        normalize = partial(processing.normalize_bound, lb=-1, ub=1)

        signal = np.squeeze(ecg_signal)
        # signal = bandpass_filter(ecg_signal, frequency)

        pad_sig = np.pad(
            signal, (self._win_size - stride, self._win_size), mode="edge"
        )

        data_windows = []
        win_idx = []

        pad_id = np.arange(pad_sig.shape[0])

        for win_id in range(0, len(pad_sig), stride):
            if win_id + self._win_size < len(pad_sig):

                window = pad_sig[win_id : win_id + self._win_size]
                if window.any():
                    window = np.squeeze(np.apply_along_axis(normalize, 0, window))

                data_windows.append(window)
                win_idx.append(pad_id[win_id : win_id + self._win_size])

        data_windows = np.asarray(data_windows)
        data_windows = data_windows.reshape(
            data_windows.shape[0], data_windows.shape[1], 1
        )
        win_idx = np.asarray(win_idx)
        win_idx = win_idx.reshape(win_idx.shape[0] * win_idx.shape[1])


        return win_idx, data_windows

    pass

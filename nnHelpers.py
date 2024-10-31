import numpy as np
from EcgData import EcgData
import tensorflow as tf
import time
import os
from models import sig2sig_unet, sig2sig_cnn
from wfdb import processing
from tqdm import tqdm



def calculate_stats(r_ref, r_ans, thr_, fs):
    # Threshold region to consider correct detection. in samples
    # thr_ = 0.15 #(150/4 ms)

    print("______________________________________________")
    print("_________Calculating Stats____________________")
    print("______________________________________________")
    FP_index_array = []
    FN_index_array = []

    FP = 0
    TP = 0
    FN = 0
    for j in range(len(r_ref)):
        loc = np.where(np.abs(r_ans - r_ref[j]) <= thr_ * fs)[0]
        if j == 0:
            err = np.where(
                (r_ans >= 0.5 * fs + thr_ * fs) & (r_ans <= r_ref[j] - thr_ * fs)
            )[0]
        elif j == len(r_ref) - 1:
            err = np.where(
                (r_ans >= r_ref[j] + thr_ * fs) & (r_ans <= 9.5 * fs - thr_ * fs)
            )[0]
        else:
            err = np.where(
                (r_ans >= r_ref[j] + thr_ * fs) & (r_ans <= r_ref[j + 1] - thr_ * fs)
            )[0]

        FP = FP + len(err)

        if err.any():
            # print(err)
            # print(len(err))
            for er in err:
                # print(r_ans[er])
                FP_index_array.append(r_ans[er])

        if len(loc) >= 1:
            TP += 1
            FP = FP + len(loc) - 1
        elif len(loc) == 0:
            FN += 1
            FN_index_array.append(r_ref[j])

    all_FP = FP
    all_FN = FN
    all_TP = TP

    Recall = float(str(round((all_TP / (all_FN + all_TP)) * 100, 2)))
    Precision = float(str(round((all_TP / (all_FP + all_TP)) * 100, 2)))

    if Recall + Precision == 0:
        F1_score = 0
    else:
        F1_score = float(str(round((2 * Recall * Precision / (Recall + Precision)), 2)))
    print("TP's:{} FN's:{} FP's:{}".format(all_TP, all_FN, all_FP))
    print(
        "Recall:{}, Precision(FNR):{}, F1-Score:{}".format(Recall, Precision, F1_score)
    )
    print("Total {}".format(len(r_ref)))

    return [
        len(r_ref),
        TP,
        FN,
        FP,
        Recall,
        Precision,
        F1_score,
        FN_index_array,
        FP_index_array,
    ]


def verifier(ecg, R_peaks, R_probs, ver_wind=60):

    del_indx = []
    wind = 30
    check_ind = np.squeeze(np.array(np.where(R_probs < 0.95)))

    if R_peaks[check_ind[-1]] == R_peaks[-1]:
        check_ind = check_ind[:-1]

    for ind in check_ind:

        two_ind = [ind, ind + 1]

        diff = R_peaks[two_ind[1]] - R_peaks[two_ind[0]]

        # 60 for chinese data
        if diff < ver_wind:

            try:

                two_probs = [R_probs[two_ind[0]], R_probs[two_ind[1]]]
                two_peaks = [R_peaks[two_ind[0]], R_peaks[two_ind[1]]]

                beat1 = ecg[two_peaks[0] - wind : two_peaks[0] + wind]
                beat2 = ecg[two_peaks[1] - wind : two_peaks[1] + wind]

                for i in range(two_ind[0] - 1, two_ind[0] - 1 - 30, -1):
                    for thr_p in [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
                        if R_probs[i] > thr_p:
                            # print(i)
                            prv_beat = ecg[R_peaks[i] - wind : R_peaks[i] + wind]
                            break

                for i in range(two_ind[1] + 1, two_ind[1] + 1 + 30):

                    for thr_p in [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
                        if i == len(R_probs):
                            nxt_beat = ecg[
                                R_peaks[i - 1] - wind : R_peaks[i - 1] + wind
                            ]
                            break

                        if R_probs[i] > thr_p:
                            # print(i)
                            nxt_beat = ecg[R_peaks[i] - wind : R_peaks[i] + wind]
                            break

                    else:
                        # Continue if the inner loop wasn't broken.
                        continue
                    # Inner loop was broken, break the outer.
                    break

                if len(nxt_beat) != 60:
                    nxt_beat = prv_beat
                if len(prv_beat) != 60:
                    prv_beat = nxt_beat
                if len(beat1) != 60:
                    beat1 = beat2
                if len(beat2) != 60:
                    beat2 = beat1

                X1 = np.corrcoef(np.squeeze(beat1), np.squeeze(prv_beat))[0, 1]
                X2 = np.corrcoef(np.squeeze(beat1), np.squeeze(nxt_beat))[0, 1]

                Y1 = np.corrcoef(np.squeeze(beat2), np.squeeze(prv_beat))[0, 1]
                Y2 = np.corrcoef(np.squeeze(beat2), np.squeeze(nxt_beat))[0, 1]

                si = np.argmin([X1 * X2, Y1 * Y2])
                del_indx.append(two_ind[si])
            except:
                pass

    R_peaks_ver = np.delete(R_peaks, del_indx)

    R_probs_ver = np.delete(R_probs, del_indx)

    return R_peaks_ver, R_probs_ver


def calculate_means(indices, values):
    """
    Calculate means of the values that have same index.
    Function calculates average from the values that have same
    index in the indices array.
    Parameters
    ----------
    indices : array
        Array of indices.
    values : array
        Value for every indice in the indices array.
    Returns
    -------
    mean_values : array
        Contains averages for the values that have the duplicate
        indices while rest of the values are unchanged.
    """
    assert indices.shape == values.shape

    # Combine indices with predictions
    comb = np.column_stack((indices, values))

    # Sort based on window indices and split when indice changes
    comb = comb[comb[:, 0].argsort()]
    split_on = np.where(np.diff(comb[:, 0]) != 0)[0] + 1

    # Take mean from the values that have same index
    startTime = time.time()
    mean_values = [arr[:, 1].mean() for arr in np.split(comb, split_on)]
    executionTime = time.time() - startTime
    # print('Execution time in seconds: ' + str(executionTime))
    mean_values = np.array(mean_values)

    return mean_values


def mean_preds(win_idx, preds, orig_len, win_size, stride):
    """
    Calculate mean of overlapping predictions.
    Function takes window indices and corresponding predictions as
    input and then calculates mean for predictions. One mean value
    is calculated for every index of the original padded signal. At
    the end padding is removed so that just the predictions for
    every sample of the original signal remain.
    Parameters
    ----------
    win_idx : array
        Array of padded signal indices before splitting.
    preds : array
        Array that contain predictions for every data window.
    orig_len : int
        Lenght of the signal that was used to extract data windows.
    Returns
    -------
    pred_mean : int
        Predictions for every point for the original signal. Average
        prediction is calculated from overlapping predictions.
    """
    # flatten predictions from different windows into one vector
    preds = preds.reshape(preds.shape[0] * preds.shape[1])
    assert preds.shape == win_idx.shape

    pred_mean = calculate_means(indices=win_idx, values=preds)

    # Remove paddig
    pred_mean = pred_mean[int(win_size - stride) : (win_size - stride) + orig_len]

    return pred_mean


def filter_predictions(signal, preds, threshold):
    """
    Filter model predictions.
    Function filters model predictions by using following steps:
    1. selects only the predictions that are above the given
    probability threshold.
    2. Correct these predictions upwards with respect the given ECG
    3. Check if at least five points are corrected into the same
    location.
    4. If step 3 is true, then location is classified as an R-peak
    5. Calculate probability of location being an R-peak by taking
    mean of the probabilities from predictions in the same location.
    Aforementioned steps can be thought as an noise reducing measure as
    in original training data every R-peak was labeled with 5 points.
    Parameters
    ----------
    signal : array
        Same signal that was used with extract_windows function. It is
        used in correct_peaks function.
    preds : array
        Predictions for the sample points of the signal.
    Returns
    -------
    filtered_peaks : array
        locations of the filtered peaks.
    filtered_probs : array
        probability that filtered peak is an R-peak.
    """

    signal = np.squeeze(signal)

    assert signal.shape == preds.shape

    # Select points probabilities and indices that are above
    # self.threshold
    above_thresh = preds[preds > threshold]
    above_threshold_idx = np.where(preds > threshold)[0]

    # Keep only points above self.threshold and correct them upwards
    correct_up = processing.correct_peaks(
        sig=signal,
        peak_inds=above_threshold_idx,
        search_radius=30,
        smooth_window_size=30,
        peak_dir="up",
    )

    filtered_peaks = []
    filtered_probs = []

    for peak_id in tqdm(np.unique(correct_up)):
        # Select indices and take probabilities from the locations
        # that contain at leas 5 points
        points_in_peak = np.where(correct_up == peak_id)[0]
        if points_in_peak.shape[0] >= 3:
            filtered_probs.append(above_thresh[points_in_peak].mean())
            filtered_peaks.append(peak_id)

    print(len(filtered_peaks))
    filtered_peaks = np.asarray(filtered_peaks)
    filtered_probs = np.asarray(filtered_probs)

    return filtered_peaks, filtered_probs


def train(X_train, y_train, R_p_w, input_size, epochs):
    start = time.process_time()

    model = sig2sig_unet(input_size)

    model_path = "models/proszePieknie.keras"

    if not os.path.exists("models/"):
        os.makedirs("models/")

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.MeanIoU(num_classes=2)],
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path, monitor="loss", verbose=1, save_best_only=True, mode="min"
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=6)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=16,
        callbacks=[checkpoint, callback],
        shuffle=True,
    )

    print(time.process_time() - start)
    pass


def test(model_name, epochs, input_size, ecg_data: EcgData, threshold=0.4):
    stats_R = []
    win_size = input_size
    stride = int(6 / 8 * win_size)
    model_path = "models/proszePieknie.keras"
    ecg = ecg_data.raw_data[:, 1]
    # R_ind = ecg_data.__r_peaks_ind

    model = sig2sig_unet(input_size)
    model.load_weights(model_path)
    padded_indices, data_windows = ecg_data.extract_test_windows(win_size, stride)
    predictions = model.predict(data_windows, verbose=0)
    predictions = mean_preds(
        win_idx=padded_indices,
        preds=predictions,
        orig_len=ecg.shape[0],
        win_size=win_size,
        stride=stride,
    )
    filtered_peaks, filtered_proba = filter_predictions(
        signal=ecg, preds=predictions, threshold=threshold
    )

    R_peaks_ver, _ = verifier(ecg, filtered_peaks, filtered_proba, ver_wind=7)
    stats_R = calculate_stats(r_ref=R_ann, r_ans=R_peaks_ver, thr_=0.15, fs=400)
    pass
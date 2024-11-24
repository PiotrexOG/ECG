from config import *
from EcgData import EcgData
from PanTompkinsFinder import PanTompkinsFinder
from nnHelpers import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from CnnFinder import CnnFinder
from PanTompkinsFinder import PanTompkinsFinder


def get_patient_data_mitbih(path, input_length = WINDOW_SIZE):
    data = EcgData(SAMPLING_RATE, PanTompkinsFinder())
    # data = EcgData(SAMPLING_RATE, CnnFinder(f"models/model_{WINDOW_SIZE}_{EPOCHS}.keras", WINDOW_SIZE))
    data.load_data_from_mitbih(path)
    data.check_detected_peaks()
    # X_train, y_train = data.extract_hrv_windows_with_loaded_peaks(input_length)
    X_train, y_train, R_per_w = data.extract_windows(input_length)
    return X_train, y_train, R_per_w


def get_patients_data_mithbih(dir_path: str, patients: list):
    X_train = None
    y_train = None
    R_per_w = None
    for patient in patients:
        x, y, r= get_patient_data_mitbih(dir_path + "\\" + patient)
        if X_train is None:
            X_train = x
            y_train = y
            R_per_w = r
        else:
            X_train = np.vstack((X_train, x))
            y_train = np.vstack((y_train, y))
            R_per_w = R_per_w + r

    return X_train, y_train, R_per_w


if __name__ == "__main__":
    X_train, y_train, R_p_w= get_patients_data_mithbih(
    "data\\mit-bih",
    ["100", "101", "102", "103", "106", "107", "109"],
    )


    # X_train, y_train = shuffle(X_train, y_train)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)

    epochs = EPOCHS
    data = EcgData(SAMPLING_RATE, PanTompkinsFinder())
    # # data.load_csv_data_with_timestamps("data/arkusz_rsa7.csv")
    # data.load_data_from_mitbih("data\\mit-bih\\100")
    # X_train, y_train, R_p_w = data.extract_windows(WINDOW_SIZE)
    train(X_train, y_train, R_p_w, WINDOW_SIZE, epochs, model_file_name=f"model_{WINDOW_SIZE}_{epochs}")

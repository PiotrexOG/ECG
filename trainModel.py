from config import *
from EcgData import EcgData
from Finders.PanTompkinsFinder import PanTompkinsFinder
from nnHelpers import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from Finders.UNetFinder import UNetFinder
from Finders.PanTompkinsFinder import PanTompkinsFinder


def get_patient_data_mitbih(path, input_length = WINDOW_SIZE):
    data = EcgData(SAMPLING_RATE, PanTompkinsFinder())
    # data = EcgData(SAMPLING_RATE, UNetFinder(f"models/model_{WINDOW_SIZE}_{EPOCHS}.keras", WINDOW_SIZE))
    data.load_data_from_mitbih(path)
    data.check_detected_peaks()
    X_train, y_train, R_per_w = data.extract_windows_loaded_peaks(input_length)
    # X_train, y_train, R_per_w = data.extract_windows(input_length)
    return X_train, y_train, R_per_w

def get_patient_data_qt(path, input_length = WINDOW_SIZE):
    data = EcgData(SAMPLING_RATE, PanTompkinsFinder())
    # data = EcgData(SAMPLING_RATE, UNetFinder(f"models/model_{WINDOW_SIZE}_{EPOCHS}.keras", WINDOW_SIZE))
    data.load_data_from_qt(path)
    data.check_detected_peaks()
    X_train, y_train, R_per_w = data.extract_windows_loaded_peaks(input_length)
    # X_train, y_train, R_per_w = data.extract_windows(input_length)
    return X_train, y_train, R_per_w

def get_patient_data_csv(path, input_length = WINDOW_SIZE):
    data = EcgData(SAMPLING_RATE, PanTompkinsFinder())
    data.load_csv_data(path)
    X_train, y_train, R_p_w = data.extract_windows(WINDOW_SIZE)
    return X_train, y_train, R_p_w


def get_patients_data(dir_path: str, patients: list, get_patient_data_fun):
    X_train = None
    y_train = None
    R_per_w = None
    for patient in patients:
        x, y, r= get_patient_data_fun(dir_path + "\\" + patient)
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

    # X_train, y_train, R_p_w= get_patients_data(
    # MITBIH_PATH,
    # load_all_patient_indexes(f"{MITBIH_PATH}\\RECORDS"),
    # get_patient_data_mitbih
    
    # QT_PATH,
    # load_all_patient_indexes(f"{QT_PATH}\\RECORDS"),
    # get_patient_data_qt
    
    # ["100"]
    # ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109"],
    # )


    # data = EcgData(SAMPLING_RATE, PanTompkinsFinder())
    # data.load_csv_data(CSV_PATH)
    # # data.load_data_from_mitbih("data\\mit-bih\\100")
    # X_train, y_train, R_p_w = data.extract_windows(WINDOW_SIZE)
    csvs = ["sen_merged.csv", "poranek_merged.csv", "popoludnie_merged2.csv"]
    X_train, y_train, R_p_w = get_patients_data("data", csvs, get_patient_data_csv)

    # X_train, y_train = shuffle(X_train, y_train)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)

    epochs = EPOCHS
    data = EcgData(SAMPLING_RATE, PanTompkinsFinder())
    
    train_unet(X_train, y_train, R_p_w, WINDOW_SIZE, epochs, model_file_name=f"model_{WINDOW_SIZE}_{epochs}")
    # train_cnn(X_train, y_train, R_p_w, WINDOW_SIZE, epochs, model_file_name=f"model_{WINDOW_SIZE}_{epochs}")











    # # data.load_csv_data("data/arkusz_rsa7.csv")
    # data.load_data_from_mitbih("data\\mit-bih\\100")
    # X_train, y_train, R_p_w = data.extract_windows(WINDOW_SIZE)
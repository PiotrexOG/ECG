from config import *
from EcgData import EcgData
from PanTompkinsFinder import PanTompkinsFinder
from nnHelpers import *


if __name__ == "__main__":

    epochs = EPOCHS
    data = EcgData(SAMPLING_RATE, PanTompkinsFinder())
    # data.load_csv_data_with_timestamps("data/arkusz_rsa7.csv")
    data.load_data_from_mitbih("data\\mit-bih\\100")
    X_train, y_train, R_p_w = data.extract_windows(WINDOW_SIZE)
    train(X_train, y_train, R_p_w, WINDOW_SIZE, epochs, model_file_name=f"model_{WINDOW_SIZE}_{epochs}")

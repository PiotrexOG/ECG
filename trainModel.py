from config import *
from EcgData import EcgData
from PanTompkinsFinder import PanTompkinsFinder
from nnHelpers import *


if __name__ == "__main__":

    epochs = 100
    data = EcgData(SAMPLING_RATE, PanTompkinsFinder())
    data.load_csv_data_with_timestamps("data/arkusz_rsa7.csv")
    X_train, y_train, R_p_w = data.extract_windows(WINDOW_SIZE)
    train(X_train, y_train, R_p_w, WINDOW_SIZE, epochs, model_file_name="model")

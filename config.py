from enum import Enum

def load_all_patient_indexes(path):
    with open(path, encoding="utf-8", mode="r") as file:
        return [line.rstrip("\n") for line in file]

HOST = "127.0.0.1"
PORT = 12345
SAMPLING_RATE = 130  # HZ
VALUES_IN_PACKET_COUNT = int(SAMPLING_RATE / 2)  # send packet twice a second
VALUE_SIZE = 4
TIMESTAMP_SIZE = 8
SINGLE_ENTRY_SIZE = VALUE_SIZE + TIMESTAMP_SIZE
csvs = [
    "ecg_data1.csv",
    "arkusz_rsa5.csv",
    "arkusz_rsa7.csv",
    "arkusz_rsa10.csv",
    "arkusz2.csv",
    "arkusz3.csv",
    "nowe_danemecz1.csv",
    "luz0lezedlugie_dane_rsa.csv"
]
CSV_PATH = "data\\" + csvs[7]


PRINT_ECG_DATA = True
NEGATE_INCOMING_DATA = False


### TEST DATA SETTINGS
class AppModeEnum(Enum):
    NORMAL = 0
    SIMULATION = 1
    LOAD_CSV = 2
    LOAD_MITBIH = 3
    LOAD_QT = 4


APP_MODE = AppModeEnum.LOAD_CSV
TIME_SCALE_FACTOR = 1e9
LOOP_DATA = False
NORMALIZED_TEST_DATA_TIME = False
MITBIH_PATH = "data\\mit-bih"
MITBIH_PATIENT = "108"
MITBIH_PATIENTS = load_all_patient_indexes(f"{MITBIH_PATH}\\RECORDS")
QT_PATH = "data\\qt-database"
QT_PATIENT = "sel104"
QT_PATIENTS = load_all_patient_indexes(f"{QT_PATH}\\RECORDS")

###

### PLOT SETTINGS
SECONDS_TO_PLOT = 6000

###

### NN Settings
WINDOW_SIZE = 512
EPOCHS = 10
###


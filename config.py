from enum import Enum


def load_all_patient_indexes(path):
    with open(path, encoding="utf-8", mode="r") as file:
        return [line.rstrip("\n") for line in file]


HOST = "127.0.0.1"
PORT = 12345
SAMPLING_RATE = 130  # HZ
VALUES_IN_PACKET_COUNT = 73  # int(SAMPLING_RATE / 2)  # send packet twice a second
VALUE_SIZE = 4
TIMESTAMP_SIZE = 8
SINGLE_ENTRY_SIZE = VALUE_SIZE + TIMESTAMP_SIZE
SLEEP_MULTIPLIER = 20
csvs = {
    1: "24h\\merged.csv",
    2: "sen_merged.csv",
    3: "poranek_merged.csv",
    4: "popoludnie.csv",
    5: "ecg_data1.csv",
    6: "nowe_arkusz_rsa5.csv",
    7: "nowe_arkusz_rsa7.csv",
    8: "nowe_arkusz_rsa10.csv",
    9: "arkusz2.csv",
    10: "arkusz3.csv",
    11: "nowe_danemecz1.csv",
    12: "luz0lezedlugie_dane_rsa.csv",
    13: "dlugie_dane_rsa5.csv",
    14: "popoludnie.csv"
}

CSV_PATH = "data\\" + csvs[6]
#CSV_PATH = "C:\\Users\\User\\Desktop\\sen\\measurement_20241127_034415.csv"
#CSV_PATH = "C:\\Users\\User\\Desktop\\popoludnie\\popoludnie_merged1.csv"


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
MITBIH_PATIENT = "207"
MITBIH_PATIENTS = load_all_patient_indexes(f"{MITBIH_PATH}\\RECORDS")
QT_PATH = "data\\qt-database"
QT_PATIENT = "sel30"
QT_PATIENTS = load_all_patient_indexes(f"{QT_PATH}\\RECORDS")

### Automatyczne mapowanie SIZE, LOW, i HIGH
breath_length_to_params = {
    2: {"SIZE": 2, "LOW": 20, "HIGH": 35},
    #  2: {"SIZE": 2, "LOW": 35, "HIGH": 50},
    #  2: {"SIZE": 1, "LOW": 16, "HIGH": 50},
    3: {"SIZE": 4, "LOW": 14, "HIGH": 33},
    4: {"SIZE": 6, "LOW": 10, "HIGH": 30},
    5: {"SIZE": 8, "LOW": 8, "HIGH": 25},
    6: {"SIZE": 10, "LOW": 8, "HIGH": 23},
    7: {"SIZE": 12, "LOW": 6, "HIGH": 17},
    8: {"SIZE": 12, "LOW": 3, "HIGH": 14},
    9: {"SIZE": 14, "LOW": 3, "HIGH": 10},
    10: {"SIZE": 14, "LOW": 3, "HIGH": 10},
}


def get_params_for_breath_length(breath_length):
    if breath_length in breath_length_to_params:
        return breath_length_to_params[breath_length]
    else:
        raise ValueError("Invalid breath length. Supported values are from 2 to 10.")


# Przykład użycia
BREATH_LENGTH = 7  # Możesz zmienić wartość na dowolną liczbę od 2 do 10
params = get_params_for_breath_length(BREATH_LENGTH)

SIZE = params["SIZE"]
LOW = params["LOW"]
HIGH = params["HIGH"]

# CSV_PATH = f"data/MECZ1.csv"  # Dynamiczna ścieżka do pliku
# CSV_PATH = f"data/01leze.csv"  # Dynamiczna ścieżka do pliku
# CSV_PATH = f"data/nowe_arkusz_rsa5.csv"  # Dynamiczna ścieżka do pliku
# CSV_PATH = f"data/sen_merged.csv"  # Dynamiczna ścieżka do pliku
# CSV_PATH = f"nowedane/measurement_20241127_050916.csv"  # Dynamiczna ścieżka do pliku

### PLOT SETTINGS
SECONDS_TO_PLOT = 600000

###

### NN Settings
WINDOW_SIZE = (130 * 60 )%64 * 64
# WINDOW_SIZE = 256 * 2
EPOCHS = 30
###

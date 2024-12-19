from enum import Enum

#PLIK KONFIGURACYJNY

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
SLEEP_MULTIPLIER = 1

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
    14: "popoludnie_merged2.csv",
    15: "nowszy_wieczor.csv",
    16: "noc_i_ranek.csv",
    17: "nowetesty1.csv",
    18: "druganoc2.csv",
    19: "24h.csv",
    20: "NOCGW4.csv",
    21: "donocy.csv",
    22: "OSTATNIA_CZESC.csv",
    23: "szum1.csv",
    24: "szum2.csv",
    25: "szum3.csv",
    26: "szum4.csv",
    27: "szum5.csv",
    28: "szum6.csv",
    29: "szum7.csv",
    30: "szum8.csv",
}
# Przykład użycia
BREATH_LENGTH = 2  # Możesz zmienić wartość na dowolną liczbę od 2 do 10
CSV_PATH = "data\\" + csvs[8]

PRINT_ECG_DATA = True
NEGATE_INCOMING_DATA = False


### TEST DATA SETTINGS
class AppModeEnum(Enum):
    NORMAL = 0
    SIMULATION = 1
    LOAD_CSV = 2
    LOAD_MITBIH = 3
    LOAD_QT = 4

class RPeakDetectionAlgorithm(Enum):
    PAN_TOMPKINS = 0
    UNET = 1
    CNN = 2


APP_MODE = AppModeEnum.SIMULATION

# Scale factor for converting incoming data timestamps to seconds.
TIMESTAMP_SCALE_FACTOR = 1e9

### Simulation settings
LOOP_DATA = True
NORMALIZED_TEST_DATA_TIME = False
###

R_PEAK_DETECTION_METHOD = RPeakDetectionAlgorithm.PAN_TOMPKINS

### NN detection settings
# WINDOW_SIZE = round((130 * 10 )%64) * 64
WINDOW_SIZE = round((10 * 130 * 36 / 13) / 64) * 64
# WINDOW_SIZE = 512
# WINDOW_SIZE = 256
# WINDOW_SIZE = 256 * 2
EPOCHS = 30
MODEL_SUFFIX = "" #+ "_noise"
###

MITBIH_PATH = "data\\mit-bih" if "noise" not in MODEL_SUFFIX else "data\\mit-bih-noise-stress"
MITBIH_PATIENT = "101"
MITBIH_PATIENTS = load_all_patient_indexes(f"{MITBIH_PATH}\\RECORDS")
QT_PATH = "data\\qt-database"
QT_PATIENT = "101"
QT_PATIENTS = load_all_patient_indexes(f"{QT_PATH}\\RECORDS")


central_freq = 1/(BREATH_LENGTH * 2)
ratio = 2
#central_freq = 0.0467
RR_FREQ = 4
SIZE = BREATH_LENGTH
LOW = central_freq/ratio
HIGH = central_freq * ratio


GAUSS_WINDOW_SIZE = 900
GAUSS_WINDOW_SIZEHR = 600

### PLOT SETTINGS
SECONDS_TO_PLOT = 5
DISPLAY_X_INDEXES = False
ECG_PLOT_BACKGROUND_COLOR = "black"
ECG_SIGNAL_COLOR = "green"
ECG_BORDER_COLOR = "green"
# ENABLE TIMER-BASED PLOT REFRESH INSTEAD OF EVENT-BASED.
USE_TIMER = False
TIMER_INTERVAL = 500 #ms
###

### NN Settings
#WINDOW_SIZE = (130 * 60 )%64 * 64
WINDOW_SIZE = 256 * 2
EPOCHS = 30
###

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
SLEEP_MULTIPLIER = 5

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
    22: "OSTATNIA_CZESC.csv"
}
# Przykład użycia
BREATH_LENGTH = 2  # Możesz zmienić wartość na dowolną liczbę od 2 do 10
#CSV_PATH = "data\\" + csvs[22]
#CSV_PATH = "bledy\\measurement_20241206_085600.csv"
CSV_PATH = "bledy\\WYNIK.csv"
#CSV_PATH = f"dataRSA\\plik_wyjściowy_part_{BREATH_LENGTH-1}.csv"
#CSV_PATH = "C:\\Users\\User\\Desktop\\sen\\measurement_20241127_034915.csv"



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
    PAN_TOMKINS = 0
    UNET = 1
    CNN = 2


APP_MODE = AppModeEnum.LOAD_CSV

# Scale factor for converting incoming data timestamps to seconds.
TIMESTAMP_SCALE_FACTOR  = 1e9

### Simulation settings
LOOP_DATA = False
NORMALIZED_TEST_DATA_TIME = False
###

R_PEAK_DETECTION_METHOD = RPeakDetectionAlgorithm.PAN_TOMKINS

### NN detection settings
WINDOW_SIZE = (130 * 60 )%64 * 64
# WINDOW_SIZE = 256 * 2
EPOCHS = 30
###



MITBIH_PATH = "data\\mit-bih"
MITBIH_PATIENT = "207"
MITBIH_PATIENTS = load_all_patient_indexes(f"{MITBIH_PATH}\\RECORDS")
QT_PATH = "data\\qt-database"
QT_PATIENT = "sel30"
QT_PATIENTS = load_all_patient_indexes(f"{QT_PATH}\\RECORDS")





#CSV_PATH = f"data\\nowe_arkusz_rsa{BREATH_LENGTH}.csv"


central_freq = 1/(BREATH_LENGTH * 2)
ratio = 2
#central_freq = 0.0467
RR_FREQ = 4
SIZE = BREATH_LENGTH
LOW = central_freq/ratio
HIGH = central_freq * ratio


GAUSS_WINDOW_SIZE = 900
GAUSS_WINDOW_SIZEHR = 600

# CSV_PATH = f"data/MECZ1.csv"  # Dynamiczna ścieżka do pliku
# CSV_PATH = f"data/01leze.csv"  # Dynamiczna ścieżka do pliku
# CSV_PATH = f"data/nowe_arkusz_rsa5.csv"  # Dynamiczna ścieżka do pliku
# CSV_PATH = f"data/sen_merged.csv"  # Dynamiczna ścieżka do pliku
# CSV_PATH = f"nowedane/measurement_20241127_050916.csv"  # Dynamiczna ścieżka do pliku

### PLOT SETTINGS
SECONDS_TO_PLOT = 600000

###

### NN Settings
#WINDOW_SIZE = (130 * 60 )%64 * 64
WINDOW_SIZE = 256 * 2
EPOCHS = 30
###

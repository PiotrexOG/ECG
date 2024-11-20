from enum import Enum

HOST = "127.0.0.1"
PORT = 12345
SAMPLING_RATE = 130  # HZ
VALUES_IN_PACKET_COUNT = int(SAMPLING_RATE / 2)  # send packet twice a second
VALUE_SIZE = 4
TIMESTAMP_SIZE = 8
SINGLE_ENTRY_SIZE = VALUE_SIZE + TIMESTAMP_SIZE

PRINT_ECG_DATA = False
NEGATE_INCOMING_DATA = False

### TEST DATA SETTINGS
class AppModeEnum(Enum):
    NORMAL = 0
    SIMULATION = 1
    LOAD_CSV = 2

APP_MODE = AppModeEnum.LOAD_CSV
TIME_SCALE_FACTOR = 1e9
LOOP_DATA = False
NORMALIZED_TEST_DATA_TIME = False

### Automatyczne mapowanie SIZE, LOW, i HIGH
breath_length_to_params = {
   #  2: {"SIZE": 2, "LOW": 20, "HIGH": 35},
    2: {"SIZE": 2, "LOW": 35, "HIGH": 50},
  #  2: {"SIZE": 2, "LOW": 16, "HIGH": 45},
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
BREATH_LENGTH = 2 # Możesz zmienić wartość na dowolną liczbę od 2 do 10
params = get_params_for_breath_length(BREATH_LENGTH)

SIZE = params["SIZE"]
LOW = params["LOW"]
HIGH = params["HIGH"]

#CSV_PATH = f"data/0lezedlugie_dane_rsa{BREATH_LENGTH}.csv"  # Dynamiczna ścieżka do pliku
#CSV_PATH = f"data/nowe_danemecz1.csv"  # Dynamiczna ścieżka do pliku
CSV_PATH = f"data/fragment7.csv"  # Dynamiczna ścieżka do pliku

### PLOT SETTINGS
SECONDS_TO_PLOT = 1000

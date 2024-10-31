from enum import Enum

HOST = "127.0.0.1"
PORT = 12345
SAMPLING_RATE = 130  # HZ
VALUES_IN_PACKET_COUNT = int(SAMPLING_RATE / 2)  # send packet twice a second
VALUE_SIZE = 4
TIMESTAMP_SIZE = 8
SINGLE_ENTRY_SIZE = VALUE_SIZE + TIMESTAMP_SIZE
csvs = ["ecg_data1.csv", "arkusz_rsa5.csv", "arkusz_rsa7.csv", "arkusz_rsa10.csv", "arkusz2.csv", "arkusz3.csv"]
CSV_PATH = "data\\" + csvs[1]


PRINT_ECG_DATA = True
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

###

### PLOT SETTINGS
SECONDS_TO_PLOT = 600

###

### NN Settings
WINDOW_SIZE = 256

###

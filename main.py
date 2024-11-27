import socket
import struct
import matplotlib.pyplot as plt
import threading
import queue
from EcgData import *
from EcgPlotter import *
from EcgPlotterFILTERED import *
from HRPlotter import HRPlotter
from config import *
from dataSenderEmulator import run_emulator_thread
from nnHelpers import *
from Finders.PanTompkinsFinder import PanTompkinsFinder
from Finders.UNetFinder import UNetFinder
from Finders.CnnFinder import CnnFinder


import tensorflow as tf
import time
import os

VERBOSE = False
PRINT_PACKETS = False


def handle_client_connection(client_socket):
    # try:
    while True:
        raw_data = receive_packet(client_socket)
        process_packet(raw_data)


def receive_packet(client_socket):
    buffer = b""
    while len(buffer) < VALUES_IN_PACKET_COUNT * SINGLE_ENTRY_SIZE:
        chunk = client_socket.recv(
            VALUES_IN_PACKET_COUNT * SINGLE_ENTRY_SIZE - len(buffer)
        )
        if not chunk:
            raise ValueError("Error while receiving data...")
        buffer += chunk
    return buffer


def process_packet(raw_data):
    timestamps = [
        struct.unpack(
            "!q",
            raw_data[
                i * SINGLE_ENTRY_SIZE
                + VALUE_SIZE : i * SINGLE_ENTRY_SIZE
                + SINGLE_ENTRY_SIZE
            ],
        )[0]
        / TIME_SCALE_FACTOR
        for i in range(VALUES_IN_PACKET_COUNT)
    ]
    if NEGATE_INCOMING_DATA:
        values = [
            -struct.unpack(
                "!f",
                raw_data[i * SINGLE_ENTRY_SIZE : i * SINGLE_ENTRY_SIZE + VALUE_SIZE],
            )[0]
            for i in range(VALUES_IN_PACKET_COUNT)
        ]
    else:
        values = [
            struct.unpack(
                "!f",
                raw_data[i * SINGLE_ENTRY_SIZE : i * SINGLE_ENTRY_SIZE + VALUE_SIZE],
            )[0]
            for i in range(VALUES_IN_PACKET_COUNT)
        ]

    data.push_raw_data(timestamps, values)
    # data1.raw_data = data.raw_data


def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))

        server_socket.listen(1)  # accept only one client
        print("Server listening on port", HOST, PORT)

        # plot_thread = threading.Thread(target=plot_data)
        # plot_thread.daemon = True
        # plot_thread.start()

        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Client connected: {client_address}")

            client_thread = threading.Thread(
                target=handle_client_connection, args=(client_socket,)
            )
            client_thread.daemon = True
            client_thread.start()


def run_server():
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()


def run_normal_mode():
    run_server()


def run_simulation():
    run_emulator_thread()
    run_normal_mode()


def run_load_CSV(data):
    data.load_csv_data(CSV_PATH)
    
def run_load_mitbih(data):
    data.load_data_from_mitbih(f"{MITBIH_PATH}\\{MITBIH_PATIENT}")
    data.check_detected_peaks()
    
def run_load_qt(data):
    data.load_data_from_qt(f"{QT_PATH}\\{QT_PATIENT}")
    data.check_detected_peaks()
    
def test_mitbih_patients(finder):
    for patient in MITBIH_PATIENTS:
        data = EcgData(SAMPLING_RATE, finder)
        data.load_data_from_mitbih(f"{MITBIH_PATH}\\{patient}")
        data.check_detected_peaks()
        
def test_qt_patients(finder):
    for patient in QT_PATIENTS:
        data = EcgData(SAMPLING_RATE, finder)
        data.load_data_from_qt(f"{QT_PATH}\\{patient}")
        data.check_detected_peaks()


if __name__ == "__main__":
    finder = PanTompkinsFinder()
    # finder = UNetFinder(f"models/model_{WINDOW_SIZE}_{EPOCHS}_unet.keras", WINDOW_SIZE)
    # finder = CnnFinder(f"models/model_{WINDOW_SIZE}_{EPOCHS}_cnn.keras", WINDOW_SIZE)
    
    # test_mitbih_patients(finder)
    # test_qt_patients(finder)
    # exit()
    
    
    # data = EcgData(SAMPLING_RATE, finder, 360)
    data = EcgData(SAMPLING_RATE, finder)
    ecg_plotter = EcgPlotter("ECG", data)
    #ecg_plotter_filtered = EcgPlotterFILTERED("ECG", data)
    # hr_plotter = HRPlotter("HR", data)

    if APP_MODE == AppModeEnum.NORMAL:
        run_normal_mode()
    elif APP_MODE == AppModeEnum.SIMULATION:
        run_simulation()
    elif APP_MODE == AppModeEnum.LOAD_CSV:
        run_load_CSV(data)        
    elif APP_MODE == AppModeEnum.LOAD_MITBIH:
        run_load_mitbih(data)        
    elif APP_MODE == AppModeEnum.LOAD_QT:
        run_load_qt(data)


    plt.show()    
    os.system("pause")
    

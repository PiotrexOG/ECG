import socket
import struct
import matplotlib.pyplot as plt
import threading
import queue
from EcgData import *
from EcgPlotter import *
from config import *
from dataSenderEmulator import run_emulator_thread
from nnHelpers import *


import tensorflow as tf
import time
import os
from models import sig2sig_unet, sig2sig_cnn

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



if __name__ == "__main__":
    data = EcgData(SAMPLING_RATE)
    # ecg_plotter = EcgPlotter("ECG", data)

    match APP_MODE:
        case AppModeEnum.NORMAL:
            run_normal_mode()

        case AppModeEnum.SIMULATION:
            run_simulation()

        case AppModeEnum.LOAD_CSV:
            run_load_CSV(data)

    plt.show()
    epochs = 150

    print(f"data length: {len(data.raw_data)}")
    X_train, y_train, R_p_w = data.extract_windows(256)
    #train(X_train, y_train, R_p_w, 256, epochs)
    test("", epochs, 256, data)

    model_name = "sig2sig_unet"

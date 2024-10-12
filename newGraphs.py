import socket
import struct
import matplotlib.pyplot as plt
import numpy as np
import threading
import queue
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.signal import lfilter, find_peaks
from EcgData import *
from EcgPlotter import *

HOST = "127.0.0.1"
PORT = 12345
SAMPLING_RATE = 130  # HZ
VERBOSE = False
PRINT_PACKETS = False

REC_VALUES_COUNT = int(130/2)  # 73
VALUE_SIZE = 4 + 8

data = EcgData()

ecg_plotter = EcgPlotter("ECG", SAMPLING_RATE)
data_queue = queue.Queue()


def handle_client_connection(client_socket):
    # try:
    while True:
        raw_data = receive_packet(client_socket)
        process_packet(raw_data)


def receive_packet(client_socket):
    buffer = b""
    while len(buffer) < REC_VALUES_COUNT * VALUE_SIZE:
        chunk = client_socket.recv(REC_VALUES_COUNT * VALUE_SIZE - len(buffer))
        if not chunk:
           raise ValueError("Error while receiving data...")
        buffer += chunk
    return buffer


def process_packet(raw_data):
    for i in range(REC_VALUES_COUNT):
        offset = i * (4 + 8)
        float_value = struct.unpack("!f", raw_data[offset : offset + 4])[
            0
        ]  # Wyciągnięcie floata
        long_value = struct.unpack("!q", raw_data[offset + 4 : offset + 12])[
            0
        ]  # Wyciągnięcie long
        if PRINT_PACKETS:
            print("First value:" + "%f" % float_value)
            print("First value timestamp:" + str(long_value))
        data.pushRawData(long_value / 1e9, -float_value)
        data_queue.put(data.rawData[-1])


def plot_data():
    while True:
        try:
            timestamp, mV = data_queue.get(timeout=1)
            ecg_plotter.send_single_sample(timestamp, mV)
        except queue.Empty:
            continue


def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))

        server_socket.listen(1)  # accept only one client
        print("Server listening on port", HOST, PORT)

        plot_thread = threading.Thread(target=plot_data)
        plot_thread.daemon = True
        plot_thread.start()

        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Client connected: {client_address}")

            client_thread = threading.Thread(
                target=handle_client_connection, args=(client_socket,)
            )
            client_thread.daemon = True
            client_thread.start()


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    plt.show()

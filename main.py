import socket
import struct
import matplotlib.pyplot as plt
import threading
import queue
from EcgData import *
from EcgPlotter import *
from HRPlotter import HRPlotter
from config import *
from dataSenderEmulator import run_emulator_thread

VERBOSE = False
PRINT_PACKETS = False



data = EcgData(SAMPLING_RATE)

ecg_plotter = EcgPlotter("ECG", data)
hr_plotter = HRPlotter("ECG", data)
data_queue = queue.Queue()


def handle_client_connection(client_socket):
    # try:
    while True:
        raw_data = receive_packet(client_socket)
        process_packet(raw_data)


def receive_packet(client_socket):
    buffer = b""
    while len(buffer) < VALUES_IN_PACKET_COUNT * SINGLE_ENTRY_SIZE:
        chunk = client_socket.recv(VALUES_IN_PACKET_COUNT * SINGLE_ENTRY_SIZE - len(buffer))
        if not chunk:
            raise ValueError("Error while receiving data...")
        buffer += chunk
    return buffer


def process_packet(raw_data):
    timestamps = [
        struct.unpack("!q", raw_data[i * SINGLE_ENTRY_SIZE + VALUE_SIZE : i * SINGLE_ENTRY_SIZE + SINGLE_ENTRY_SIZE])[0] / 1e9
        for i in range(VALUES_IN_PACKET_COUNT)
    ]
    if NEGATE_INCOMING_DATA:
        values = [
            -struct.unpack("!f", raw_data[i * SINGLE_ENTRY_SIZE : i * SINGLE_ENTRY_SIZE + VALUE_SIZE])[0]
            for i in range(VALUES_IN_PACKET_COUNT)
        ]
    else:
        values = [
            struct.unpack("!f", raw_data[i * SINGLE_ENTRY_SIZE : i * SINGLE_ENTRY_SIZE + VALUE_SIZE])[0]
            for i in range(VALUES_IN_PACKET_COUNT)
        ]
        
    """
    # for i in range(VALUES_IN_PACKET_COUNT):
    #     offset = i * (4 + 8)
    #     float_value = struct.unpack("!f", raw_data[offset : offset + 4])[
    #         0
    #     ]  # Wyciągnięcie floata
    #     long_value = struct.unpack("!q", raw_data[offset + 4 : offset + 12])[
    #         0
    #     ]  # Wyciągnięcie long
    #     timestamps.append(long_value / 1e9)
    #     values.append(-float_value)

    #     if PRINT_PACKETS:
    #         print("First value:" + "%f" % float_value)
    #         print("First value timestamp:" + str(long_value))
    #     # data.push_raw_data(long_value / 1e9, -float_value)
    #     # data_queue.put(data.raw_data[-1])
    #     data_queue.put((long_value / 1e9, -float_value))
    """

    data.push_raw_data(timestamps, values)

"""
# def plot_data():
#     while True:
#         try:
#             timestamp, mV = data_queue.get(timeout=1)
#             # ecg_plotter.send_single_sample(timestamp, mV)
#         except queue.Empty:
#             continue
"""

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


if __name__ == "__main__":
    if RUN_TEST_DATA:
        run_emulator_thread()

    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    plt.show()

import csv
import struct
import time
import threading
import socket

from config import *


def read_csv(path: str):
    data = []

    with open(path, mode="r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append((int(row[0]), float(row[1])))

    return data


def send_loop(data):
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((HOST, PORT))
                index = 0
                first_timestamp = data[0][0] if NORMALIZED_TEST_DATA_TIME else 0
                offset = 0
                while True:
                    byte_buffer = b""
                    first_ts_in_iteration = data[index][0] + offset
                    for i in range(VALUES_IN_PACKET_COUNT):
                        timestamp = data[index][0] - first_timestamp + offset
                        value = data[index][1]
                        byte_buffer += struct.pack("!f", value)
                        byte_buffer += struct.pack("!q", timestamp)

                        index = index + 1
                        if index == len(data):
                            if not LOOP_DATA:
                                return
                            index = 0
                            offset += (
                                (data[-1][0] - first_timestamp)
                                + data[-1][0]
                                - data[-2][0]
                            )

                    sock.sendall(byte_buffer)

                    sleep_time = (data[index][0] + offset - first_ts_in_iteration) / TIMESTAMP_SCALE_FACTOR / SLEEP_MULTIPLIER
                    time.sleep(sleep_time)
        except Exception as ex:
            print(f"Exception was thrown: {ex}")


def emulator_thread_entry():
    csv_data = read_csv(CSV_PATH)
    send_loop(csv_data)


def run_emulator_thread():
    thread = threading.Thread(target=emulator_thread_entry)
    thread.daemon = True
    thread.start()

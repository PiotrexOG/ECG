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
            data.append((int(row[0]), int(row[1])))

    return data


def send_loop(data):
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((HOST, PORT))
                index = 0
                first_timestamp = data[0][0]
                offset = 0
                while True: 
                    byte_buffer = b''
                    first_ts_in_iteration = data[index][0]
                    for i in range(VALUES_IN_PACKET_COUNT):
                        timestamp = data[index][0] - first_timestamp + offset
                        value = data[index][1]
                        byte_buffer += struct.pack('!f', value)
                        byte_buffer += struct.pack('!q', timestamp)

                        index = index + 1
                        if index > len(data):
                            index = 0
                            offset = offset + data[-1] - first_timestamp
                    
                    sock.sendall(byte_buffer)
                    
                    time.sleep((data[index][0]-first_ts_in_iteration)/1e9)   
        except Exception as ex:
            print(f"Exception was thrown: {ex}")

def emulator_thread_entry():
    csv_data = read_csv(CSV_PATH)
    send_loop(csv_data)  

def run_emulator_thread():
    thread = threading.Thread(target=emulator_thread_entry)
    thread.daemon = True
    thread.start()
    
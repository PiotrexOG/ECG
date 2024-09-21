import socket
import struct
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import threading
import queue


class EcgPlotter:
    def __init__(self, title):
        self.SECONDS_TO_PLOT = 5
        self.plot_data = deque(maxlen=130 * self.SECONDS_TO_PLOT)  # Assuming 130 Hz max frequency
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('mV')
        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.update_plot)
        self.timer.start()

    def send_single_sample(self, timestamp, mV):
        self.plot_data.append((timestamp, mV))

    def update_plot(self):
        if len(self.plot_data) > 0:
            timestamps, values = zip(*self.plot_data)
            # Convert timestamps from ms to seconds
            x = np.array(timestamps)
            self.line.set_data(x - x[0], values)  # Normalize time to start from 0
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw_idle()  # Update the plot safely for threading


# Define the host and port to listen on
HOST = '127.0.0.1'  # Listen on all network interfaces
PORT = 12345  # Choose a port number
ecg_plotter = EcgPlotter("ECG")
data_queue = queue.Queue()


def handle_client_connection(client_socket):
    try:
        while True:
            raw_data = receive_packet(client_socket)
            process_packet(raw_data)
    except (ConnectionResetError, ValueError):
        print("Client disconnected.")
    finally:
        client_socket.close()


def receive_packet(client_socket):
    raw_data = client_socket.recv(73 * (4 + 8))
    if len(raw_data) < 73 * (4 + 8):
        raise ValueError("Brak danych lub błąd połączenia.")
    return raw_data


def process_packet(raw_data):
    siema = 0
    for i in range(73):
        offset = i * (4 + 8)
        float_value = struct.unpack('!f', raw_data[offset:offset + 4])[0]  # Wyciągnięcie floata
        long_value = struct.unpack('!q', raw_data[offset + 4:offset + 12])[0]  # Wyciągnięcie long
        if siema == 0:
            print("First value:" + "%f" % float_value)
            print("First value timestamp:" + str(long_value))
        siema = 1
        data_queue.put((long_value, float_value))  # Dodanie wartości do kolejki


def plot_data():
    while True:
        try:
            timestamp, mV = data_queue.get(timeout=1)
            ecg_plotter.send_single_sample(timestamp, mV)
        except queue.Empty:
            continue


def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Bind the socket to the address and port
        server_socket.bind((HOST, PORT))

        # Start listening for incoming connections
        server_socket.listen(1)
        print("Server listening on port", HOST, PORT)

        plot_thread = threading.Thread(target=plot_data)
        plot_thread.daemon = True
        plot_thread.start()

        while True:
            # Accept incoming connections
            client_socket, _ = server_socket.accept()
            print("Client connected.")

            # Handle the client connection
            client_thread = threading.Thread(target=handle_client_connection, args=(client_socket,))
            client_thread.daemon = True
            client_thread.start()


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    plt.show()  # Upewnij się, że główny wątek pozostaje w pętli wyświetlania wykresu

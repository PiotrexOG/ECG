import socket
import struct
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import threading
import queue
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.signal import lfilter, find_peaks

HOST = '127.0.0.1' 
PORT = 12345
SAMPLING_RATE = 130 #HZ
VERBOSE = False
PRINT_PACKETS = False

REC_VALUES_COUNT = 73 #73
VALUE_SIZE = 4 + 8



class EcgPlotter:
    def __init__(self, title, sampling_frequency):
        self.SECONDS_TO_PLOT = 5
        self.plot_data = deque(maxlen=sampling_frequency * self.SECONDS_TO_PLOT) 
        self.r_peaks = []  # List to store R-peaks timestamps
        
        # Create subplots: one for ECG, one for moving average signal
        self.fig, (self.ax_ecg) = plt.subplots()
        
        # ECG plot
        self.line_ecg, = self.ax_ecg.plot([], [], color='green')
        self.ax_ecg.set_title(title + " - ECG Signal")
        self.ax_ecg.set_ylabel('uV')

        self.ax_ecg.set_facecolor('black')
        self.ax_ecg.spines['bottom'].set_color('green')
        self.ax_ecg.spines['top'].set_color('green')
        self.ax_ecg.spines['right'].set_color('green')
        self.ax_ecg.spines['left'].set_color('green')
        self.ax_ecg.tick_params(axis='x',)
        self.ax_ecg.tick_params(axis='y',)
        self.ax_ecg.set_title('ECG Waveform')
        self.ax_ecg.set_xlabel('Time (s)')
        
        # self.timer = self.fig.canvas.new_timer(interval=100)
        # self.timer.add_callback(self.update_plot)
        # self.timer.start()

        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)


    def send_single_sample(self, timestamp, mV):
        self.plot_data.append((timestamp, mV))

    def update_plot(self, frame):
        if len(self.plot_data) > 0:
            timestamps, ecg_values = zip(*self.plot_data)
            x = np.array(timestamps)
            x_normalized = x - x[0]  # Normalize time to start from 0

            # Update ECG plot
            self.line_ecg.set_data(x_normalized, ecg_values)  # Use normalized time
            self.ax_ecg.relim()  # Recalculate limits
            self.ax_ecg.autoscale_view()  # Auto scale the view

        return self.line_ecg,  # Return the updated line object for FuncAnimation
            


ecg_plotter = EcgPlotter("ECG", SAMPLING_RATE)
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
    raw_data = client_socket.recv(REC_VALUES_COUNT * VALUE_SIZE)
    if len(raw_data) < REC_VALUES_COUNT * (4 + 8):
        raise ValueError("Brak danych lub błąd połączenia.")
    return raw_data


def process_packet(raw_data):
    for i in range(REC_VALUES_COUNT):
        offset = i * (4 + 8)
        float_value = struct.unpack('!f', raw_data[offset:offset + 4])[0]  # Wyciągnięcie floata
        long_value = struct.unpack('!q', raw_data[offset + 4:offset + 12])[0]  # Wyciągnięcie long
        if PRINT_PACKETS:
            print("First value:" + "%f" % float_value)
            print("First value timestamp:" + str(long_value))
        data_queue.put((long_value/1e9, -float_value))  # Dodanie wartości do kolejki


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

        server_socket.listen(1) #accept only one client
        print("Server listening on port", HOST, PORT)

        plot_thread = threading.Thread(target=plot_data)
        plot_thread.daemon = True
        plot_thread.start()

        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Client connected: {client_address}")

            client_thread = threading.Thread(target=handle_client_connection, args=(client_socket,))
            client_thread.daemon = True
            client_thread.start()


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    plt.show()

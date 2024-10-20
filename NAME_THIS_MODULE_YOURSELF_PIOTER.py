import numpy as np

def find_r_peaks_piotr(raw_data):
    r_peaks = []  # Lista do przechowywania timestampów i wartości R-peaków
    threshold = 600  # Próg dla wartości sygnału (w mV)
    last_peak_time = -1  # Czas ostatniego wykrytego załamka R (inicjalnie brak)

    for index, (timestamp, value) in enumerate(raw_data):
        # Sprawdzamy, czy dane są nowsze od ostatnio wykrytego R-peaku
        if timestamp > last_peak_time:
            # Sprawdź, czy wartość sygnału przekracza próg
            if value > threshold:
                # Jeżeli to pierwsza wartość, ustaw ją jako początkową
                if index > 2:
                    # Oblicz pochodną (różnica wartości / różnica czasów)
                    derivative = (value - raw_data[index - 1][1])
                    prev_derivate = raw_data[index - 1][1] - raw_data[index - 2][1]
                    # Sprawdź, czy pochodna zmienia znak lub jest równa zero (szczyt załamka R)
                    if derivative <= 0 and prev_derivate >= 0:
                        # Zidentyfikowano szczyt, zapisujemy ten punkt jako załamek R
                        r_peaks.append(
                            (raw_data[index - 1][0], raw_data[index - 1][1]))  # Dodaj timestamp i wartość załamka R
                        last_peak_time = raw_data[index - 1][0]  # Zaktualizuj czas ostatniego załamka R

    # Zamień listę krotek na dwuwymiarową tablicę za pomocą np.column_stack()
    if r_peaks:
        peak_timestamps, peak_values = zip(*r_peaks)
        peaks_with_timestamps = np.column_stack((peak_timestamps, peak_values))
    else:
        peaks_with_timestamps = np.empty((0, 2))  # Zwróć pustą 2D tablicę, jeśli brak załamków R

    return peaks_with_timestamps
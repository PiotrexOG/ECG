import os
import pandas as pd

def find_and_merge_files(directory):
    # Pobierz listę plików CSV w katalogu i posortuj je alfabetycznie
    files = [f for f in os.listdir(directory) if f.startswith("measurement_") and f.endswith(".csv")]
    files.sort()  # Sortowanie według nazwy pliku

    merged_dataframes = []

    for file in files:
        filepath = os.path.join(directory, file)
        # Wczytaj każdy plik bez nagłówka
        df = pd.read_csv(filepath, header=None)
        merged_dataframes.append(df)

    # Połącz wszystkie DataFrame w jeden
    if merged_dataframes:
        final_dataframe = pd.concat(merged_dataframes, ignore_index=True)
        output_file = os.path.join(directory, "merged.csv")
        final_dataframe.to_csv(output_file, index=False, header=False)
        print(f"Połączony plik został zapisany jako: {output_file}")
    else:
        print("Nie znaleziono plików do połączenia.")

# Użycie funkcji
directory = "data\\24h"  # Zmień na właściwą ścieżkę
find_and_merge_files(directory)
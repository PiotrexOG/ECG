from EcgData import EcgData
from PanTompkinsFinder import PanTompkinsFinder
from EcgPlotter import EcgPlotter
import matplotlib.pyplot as plt


if __name__ == "__main__":

    data = EcgData(0, PanTompkinsFinder())
    data.load_data_from_qt("data\\qt-database\\sel30")
    EcgPlotter("QT DATABASE TEST", data)

    plt.show()

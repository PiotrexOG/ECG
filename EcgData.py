import numpy as np


class EcgData:

    def __init__(self):
        self.rawData = np.empty((0, 2))
        pass

    def pushRawData(self, x, y):
        self.rawData = np.append(self.rawData, [[x, y]], axis=0)
        return

    pass

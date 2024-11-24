from nnHelpers import *
from Finders.UNetFinder import UNetFinder

class CnnFinder(UNetFinder):

    def __init__(self, model_path, input_size) -> None:
        self._win_size = input_size
        model = sig2sig_cnn(input_size)
        model.load_weights(model_path)
        self._model = model
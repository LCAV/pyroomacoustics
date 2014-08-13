
import numpy as np


class Signal(np.ndarray):

    def __init__(self, Fs, *args):

        np.ndarray.__init__(self, args)

        self.Fs = Fs

    @classmethod
    def signal(class, Fs, x):
        Signal

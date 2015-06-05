# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

import numpy as np

class Signal(np.ndarray):

    def __init__(self, Fs, *args):

        np.ndarray.__init__(self, args)

        self.Fs = Fs

    @classmethod
    def signal(class, Fs, x):
        Signal

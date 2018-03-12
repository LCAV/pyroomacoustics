'''
Google Speech Commands Dataset
=================

Pyroomacoustics includes a wrapper around the Google Speech Commands dataset [TODO add reference].
'''

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pyroomacoustics as pra
import os, argparse

if __name__ == '__main__':

    # create object
    dataset = pra.datasets.GoogleSpeechCommands(download=True)

    # print dataset info and 10 sentences
    print(dataset)
    dataset.head(n=10)

    # # TODO subset example
    # subset = dataset.subset()
    # print(subset)


    # show the spectrogram
    import matplotlib.pyplot as plt

    plt.figure()
    dataset[10].plot()

    plt.show()



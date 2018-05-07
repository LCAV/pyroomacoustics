'''
Google Speech Commands Dataset
=================
Pyroomacoustics includes a wrapper around the Google Speech Commands dataset.

More info about the dataset can be found at the link below:

https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html
'''

import pyroomacoustics as pra
import os, argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of using the GoogleSpeechCommands wrapper')
    parser.add_argument('--noplot', action='store_true',
            help='Do not display any plot')
    parser.add_argument('--playsound', action='store_true',
            help='Play one example sentence')
    args = parser.parse_args()

    # create a subset of the Google Speech Commands dataset that contains 10 of each word and the noise samples
    dataset = pra.datasets.GoogleSpeechCommands(download=True, subset=10)

    # print dataset info and first 10 entries
    print(dataset)
    print()
    dataset.head(n=10)

    # separate the noise and the speech samples
    print()
    noise_samps = dataset.filter(speech=0)
    print("Number of noise samples : %d" % len(noise_samps))
    speech_samps = dataset.filter(speech=1)
    print("Number of speech samples : %d" % len(speech_samps))

    # print info of first speech sample
    print()
    print(speech_samps[0])

    # list sounds in our dataset and number of occurences
    print()
    print("All sounds in the dataset:")
    print(dataset.classes)

    # filter by specific word
    selected_word = dataset.classes[1]
    matches = speech_samps.filter(word=selected_word)
    print()
    print("Number of '%s' samples : %d" % (selected_word, len(matches)))

    # play sound
    if args.playsound:
        matches[0].play()

    # show the spectrogram
    if not args.noplot:
        import matplotlib.pyplot as plt

        plt.figure()
        matches[0].plot()

        plt.show()

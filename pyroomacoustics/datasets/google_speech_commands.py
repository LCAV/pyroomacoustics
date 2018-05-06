# -*- coding: utf-8 -*-
'''
Google's Speech Commands Dataset
======================
The Speech Commands Dataset has 65'000 one-second long utterances of 30 short 
words, by thousands of different people, contributed by members of the public
through the AIY website. It’s released under a Creative Commons BY 4.0 license.

More info about the dataset can be found at the link below:

https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html

AIY website for contributing recordings:

https://aiyprojects.withgoogle.com/open_speech_recording

Tutorial on creating a word classifier:

https://www.tensorflow.org/versions/master/tutorials/audio_recognition
'''

import os, glob
import numpy as np
from scipy.io import wavfile

try:
    import sounddevice as sd
    have_sounddevice = True
except:
    have_sounddevice = False

from .utils import download_uncompress
from .base import Meta, AudioSample, Dataset


google_speech_commands_sounds = {
    'zero' : { 'speech' : 1},
    'yes': {'speech' : 1},
    'wow': {'speech' : 1},
    'up' : {'speech' : 1},
    'two' : {'speech' : 1},
    'tree' : {'speech' : 1},
    'stop' : {'speech' : 1},
    'six' : {'speech' : 1},
    'sheila' : {'speech' : 1},
    'seven' : {'speech' : 1},
    'right' : {'speech' : 1},
    'one' : {'speech' : 1},
    'on' : {'speech' : 1},
    'off' : {'speech' : 1},
    'no' : {'speech' : 1},
    'nine' : {'speech' : 1},
    'marvin' : {'speech' : 1},
    'left' : {'speech' : 1},
    'house' : {'speech' : 1},
    'happy' : {'speech' : 1},
    'go' : {'speech' : 1},
    'four' : {'speech' : 1},
    'five' : {'speech' : 1},
    'eight' : {'speech' : 1},
    'down' : {'speech' : 1},
    'dog' : {'speech' : 1},
    'cat' : {'speech' : 1},
    'bird' : {'speech' : 1},
    'bed' : {'speech' : 1},
    '_background_noise_' : {'speech' : 0},
    }


url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"


class GoogleSpeechCommands(Dataset):
    '''
    Parameters
    ----------
    basedir: str, optional
        The directory where the Google Speech Command dataset is located/downloaded. By
        default, this is the current directory.
    download: bool, optional
        If the corpus does not exist, download it.
    build: bool, optional
        Can be 'female' or 'male'
    '''

    def __init__(self, basedir=None, subset=None, download=False, build=True, 
        seed=0, **kwargs):

        # initialize
        Dataset.__init__(self)
        self.size_by_samples = {}

        # default base directory is the current one
        self.basedir = basedir
        if basedir is None:
            self.basedir = './google_speech_commands'

        # check the directory exists and download otherwise
        if not os.path.exists(self.basedir):
            if download:
                print('Downloading', url, 'into', self.basedir, '...')
                download_uncompress(url=url, path=self.basedir)
            else:
                raise ValueError('Dataset directory does not exist. Create or set download option.')
        else:
            print("Dataset exists! Using %s" % self.basedir)

        # set seed of random number generator
        self.rng = np.random.RandomState(seed)

        if build:
            self.build_corpus(subset, **kwargs)


    def build_corpus(self, subset=None, **kwargs):
        '''
        Build the corpus with some filters (speech or not speech, sound type).
        '''

        self.subdirs = glob.glob(os.path.join(self.basedir,'*','.'))
        self.classes = [s.split(os.sep)[-2] for s in self.subdirs]

        # go through all subdirectories / soundtypes
        for idx, word in enumerate(self.classes):

            if word == '_background_noise_':
                speech = 0
            else:
                speech = 1

            # get all list of all files in the subdirectory
            word_path = self.subdirs[idx]
            files = glob.glob(os.path.join(word_path, '*.wav'))

            # for speech files, select desired subset
            if subset and speech:
                rand_idx = np.arange(len(files)) 
                n_files = min(subset, len(files))
                self.rng.shuffle(rand_idx)
                files = [files[i] for i in rand_idx[:n_files]]
            
            self.size_by_samples[word] = len(files)

            # add each file to the corpus
            for filename in files:

                file_loc = os.path.join(self.basedir, word, os.path.basename(filename))

                # could also add score of original model for each word?
                if speech:
                    meta = Meta(word=word, speech=speech, file_loc=file_loc)
                else:
                    noise_type = os.path.basename(filename).split(".")[0]
                    meta = Meta(noise_type=noise_type, speech=speech, file_loc=file_loc)

                if meta.match(**kwargs):
                    self.add_sample(GoogleSample(filename, **meta.as_dict()))


class GoogleSample(AudioSample):
    '''
    Create the sound object.
    Parameters
    ----------
    path: str
      the path to the audio file
    **kwargs:
      metadata as a list of keyword arguments
    Attributes
    ----------
    data: array_like
      the actual audio signal
    fs: int
      sampling frequency
    '''

    def __init__(self,path,**kwargs):
        '''
        Create the the sound object
        path: string
          the path to a particular sample
        '''

        fs,data = wavfile.read(path)
        AudioSample.__init__(self, data, fs, **kwargs)

    def __str__(self):
        '''string representation'''

        if self.meta.speech:
            template = 'speech: ''{speech}''; word: ''{word}''; file_loc: ''{file_loc}'''
        else:
            template = 'speech: ''{speech}''; noise type: ''{noise_type}''; file_loc: ''{file_loc}'''
        s = template.format(**self.meta.as_dict())
        return s


    def plot(self,**kwargs):
        '''Plot the spectogram'''
        try:
            import matplotlib.pyplot as plt 
        except ImportError:
            print('Warning: matplotlib is required for plotting')
            return
        AudioSample.plot(self,**kwargs)
        if self.meta.speech:
            plt.title(self.meta.file_loc)
        else:
            plt.title(self.meta.file_loc)
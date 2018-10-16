# -*- coding: utf-8 -*-
'''
Google's Speech Commands Dataset
================================
The Speech Commands Dataset has 65,000 one-second long utterances of 30 short 
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

url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"


class GoogleSpeechCommands(Dataset):
    '''
    This class will load the Google Speech Commands Dataset in a
    structure that is convenient to be processed.

    Attributes
    ----------
    basedir: str
        The directory where the Speech Commands Dataset is located/downloaded.
    size_by_samples: dict
        A dictionary whose keys are the words in the dataset. The values are the number of occurances for that particular word.
    subdirs: list
        The list of subdirectories in ``basedir``, where each sound type is the name of a subdirectory.
    classes: list
        The list of all sounds, same as the keys of ``size_by_samples``.

    Parameters
    ----------
    basedir: str, optional
        The directory where the Google Speech Commands dataset is located/downloaded. By default, this is the current directory.
    download: bool, optional
        If the corpus does not exist, download it.
    build: bool, optional
        Whether or not to build the dataset. By default, it is.
    subset: int, optional
        Build a dataset that contains all noise samples and `subset` samples per word. By default, the dataset will be built with all samples.
    seed: int, optional
        Which seed to use for the random generator when selecting a subset of samples. By default, ``seed=0``.
    '''

    def __init__(self, basedir=None, download=False, build=True, subset=None,
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
        self.seed = seed
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
                speech = False
            else:
                speech = True

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
                    meta = Meta(word="NA", noise_type=noise_type, speech=speech, file_loc=file_loc)

                if meta.match(**kwargs):
                    self.add_sample(GoogleSample(filename, **meta.as_dict()))


    def filter(self, **kwargs):
        '''
        Filter the dataset and select samples that match the criterias provided
        The arguments to the keyword can be 1) a string, 2) a list of strings, 3)
        a function. There is a match if one of the following is True.

        1. ``value == attribute``
        2. ``value`` is a list and ``attribute in value == True``
        3. ``value`` is a callable (a function) and ``value(attribute) == True``
        '''

        # first, create the new empty corpus
        new_corpus = GoogleSpeechCommands(basedir=self.basedir, build=False,
            seed=self.seed)

        # finally, add all the sentences
        for s in self.samples:
            new_corpus.add_sample_matching(s, **kwargs)

        return new_corpus


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

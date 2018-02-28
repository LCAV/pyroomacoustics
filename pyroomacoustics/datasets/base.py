'''
Base class for some data corpus and the samples it contains
'''

from collections import namedtuple

class Meta(object):
    '''
    A simple class that will take a dictionary as input
    and put the values in attributes named after the keys.
    We use it to store metadata for the samples

    The parameters can be any set of keyword arguments.
    They will all be transformed into attribute of the object.

    Methods:
    --------
    match:
        This method takes any number of keyword arguments
        and will return True if they all match exactly similarly
        named attributes of the object. If some keys are missing,
        an error will be raised. Omitted keys will be ignored.
    as_dict:
        Returns a dictionary representation of the object
    '''
    def __init__(self, **attr):
        for key, val in attr.items():
            self.__setattr__(key, val)

    def match(self, **kwargs):
        '''
        This is a method that will return True if all keyword arguments match
        the attributes of the object
        '''
        for key, val in kwargs.items():
            attr = self.__getattribute__(key)
            if attr != val and not (isinstance(val, list) and attr in val):
                return False
        return True

    def as_dict(self):
        return self.__dict__.copy()

    def __str__(self):
        return str(self.as_dict())

    def __repr__(self):
        return self.as_dict()


class SampleBase(object):
    '''
    The base class for a dataset sample. The idea is that different
    corpus will have different attributes for the samples. They
    should at least have a data attribute.

    Attributes
    ----------
    data: array_like
        The actual data
    meta: pyroomacoustics.datasets.Meta
        An object containing the sample metadata. They can be accessed using the
        dot operator
    '''

    def __init__(self, data, metadata):
        ''' Dummy init method '''
        self.data = data
        self.meta = metadata


class AudioSample(SampleBase):
    '''
    We add some method specific to display and listen to audio samples

    Attributes
    ----------
    data: array_like
        The actual data
    fs: int
        The sampling frequency of the input signal
    meta: pyroomacoustics.datasets.Meta
        An object containing the sample metadata. They can be accessed using the
        dot operator
    '''
    def __init__(self, data, fs, metadata):
        self.data = data
        self.fs = fs
        self.meta = metadata

    def play(self):
        ''' Play the sound sample '''
        try:
            import sounddevice as sd
        except ImportError as e:
            print('Warning: sounddevice package is required to play audiofiles.')
            return

        sd.play(self.data, samplerate=self.fs, blocking=True)

    def plot(self, L=512, hop=128, zpb=0, phonems=False, **kwargs):

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('Warning: matplotlib is required for plotting')
            return

        plt.specgram(self.data, NFFT=L, Fs=self.fs, noverlap=L-hop, pad_to=L+zpb)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')


class CorpusBase(object):
    '''
    The base class for a data corpus. It has basically a list of
    samples and a filter function

    Attributes
    ----------
    samples: list
        A list of all the Samples in the dataset

    '''
    def __init__(self):
        self.samples = []
        self.info = {}

    def add_sample(self, sample, **kwargs):
        ''' 
        Add a sample to the list and keep track of the metadata.
        '''
        # keep track of the metadata going in the corpus
        for key, val in sample.meta.__dict__.items():
            if key not in self.info:
                self.info[key] = set()
            self.info[key].add(val)

        # add the sample to the list
        self.samples.append(sample)

        '''
        import pdb
        pdb.set_trace()
        '''

    def add_sample_matching(self, sample, **kwargs):
        '''
        If keyword arguments are given they are used to match
        the metadata. The sample is only added if the keywords are
        matching.
        '''
        # check if the keyword arguments are matching
        if sample.meta.match(**kwargs):
            self.add_sample(sample)
        
    def __getitem__(self, r):
        return self.samples[r]

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        s = '\n'.join(*self.samples)




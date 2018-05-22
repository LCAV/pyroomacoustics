'''
Base class for some data corpus and the samples it contains.
'''

from itertools import islice

def _take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

class Meta(object):
    '''
    A simple class that will take a dictionary as input
    and put the values in attributes named after the keys.
    We use it to store metadata for the samples

    The parameters can be any set of keyword arguments.
    They will all be transformed into attribute of the object.
    '''
    def __init__(self, **attr):
        for key, val in attr.items():
            self.__setattr__(key, val)

    def match(self, **kwargs):
        '''
        The key/value pairs given by the keyword arguments are compared
        to the attribute/value pairs of the object. If the values all
        match, True is returned. Otherwise False is returned. If a keyword
        argument has no attribute counterpart, an error is raised. Attributes
        that do not have a keyword argument counterpart are ignored.

        There are three ways to match an attribute with keyword=value:
        1. ``value == attribute``
        2. ``value`` is a list and ``attribute in value == True``
        3. ``value`` is a callable (a function) and ``value(attribute) == True``
        '''
        for key, val in kwargs.items():
            attr = self.__getattribute__(key)
            if callable(val) and val(attr):
                continue
            if isinstance(val, list) and attr in val:
                continue
            if attr == val:
                continue
            return False
            '''
            if (not (callable(val) val(attr))
                    and not (isinstance(val, list) and attr in val)
                    and attr != val):
                return False
            if attr != val and not (isinstance(val, list) and attr in val):
                return False
            '''
        return True

    def as_dict(self):
        ''' Returns all the attribute/value pairs of the object as a dictionary '''
        return self.__dict__.copy()

    def __str__(self):
        r = 'Metadata:\n'
        for attr, val in self.__dict__.items():
            r += '    {} : {}\n'.format(attr, val)
        return r[:-1]  # remove the trailing '\n'

    def __repr__(self):
        return self.__dict__.__repr__()


class Sample(object):
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

    def __init__(self, data, **kwargs):
        ''' Dummy init method '''
        self.data = data
        self.meta = Meta(**kwargs)

    def __str__(self):
        r = 'Data : ' + self.data.__str__() + '\n'
        r += self.meta.__str__()
        return r


class AudioSample(Sample):
    '''
    We add some methods specific to display and listen to audio samples.
    The sampling frequency of the samples is an extra parameter.

    For multichannel audio, we assume the same format used by 
    ```scipy.io.wavfile <https://docs.scipy.org/doc/scipy-0.14.0/reference/io.html#module-scipy.io.wavfile>`_``,
    that is ``data`` is then a 2D array with each column being a channel.

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
    def __init__(self, data, fs, **kwargs):
        Sample.__init__(self, data, **kwargs)
        self.fs = fs

    def play(self, **kwargs):
        '''
        Play the sound sample. This function uses the 
        `sounddevice <https://python-sounddevice.readthedocs.io>`_ package for playback.

        It takes the same keyword arguments as 
        `sounddevice.play <https://python-sounddevice.readthedocs.io/en/0.3.10/#sounddevice.play>`_.
        '''
        try:
            import sounddevice as sd
        except ImportError as e:
            print('Warning: sounddevice package is required to play audiofiles.')
            return

        sd.play(self.data, samplerate=self.fs, **kwargs)

    def plot(self, NFFT=512, noverlap=384, **kwargs):
        '''
        Plot the spectrogram of the audio sample. 

        It takes the same keyword arguments as
        `matplotlib.pyplot.specgram <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.specgram.html>`_.
        '''

        import numpy as np
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('Warning: matplotlib is required for plotting')
            return

        # Handle single channel case
        if self.data.ndim == 1:
            data = self.data[:,None]

        nchannels = data.shape[1]

        # Try to have a square looking plot
        pcols = int(np.ceil(np.sqrt(nchannels)))
        prows = int(np.ceil(nchannels / pcols))

        for c in range(nchannels):
            plt.specgram(data[:,c], NFFT=NFFT, Fs=self.fs, noverlap=noverlap, **kwargs)
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            plt.title('Channel {}'.format(c+1))


class Dataset(object):
    '''
    The base class for a data corpus. It has basically a list of
    samples and a filter function

    Attributes
    ----------
    samples: list
        A list of all the Samples in the dataset
    info: dict
        This dictionary keeps track of all the fields
        in the metadata. The keys of the dictionary are
        the metadata field names. The values are again dictionaries,
        but with the keys being the possible values taken by the 
        metadata and the associated value, the number of samples
        with this value in the corpus.
    '''
    def __init__(self):
        self.samples = []
        self.info = {}

    def add_sample(self, sample):
        ''' 
        Add a sample to the Dataset and keep track of the metadata.
        '''
        # keep track of the metadata going in the corpus
        for key, val in sample.meta.__dict__.items():
            if key not in self.info:
                self.info[key] = {}

            if val not in self.info[key]:
                self.info[key][val] = 1
            else:
                self.info[key][val] += 1

        # add the sample to the list
        self.samples.append(sample)

    def add_sample_matching(self, sample, **kwargs):
        '''
        The sample is added to the corpus only if all the keyword arguments
        match the metadata of the sample.  The match is operated by
        ``pyroomacoustics.datasets.Meta.match``.
        '''
        # check if the keyword arguments are matching
        if sample.meta.match(**kwargs):
            self.add_sample(sample)

    def filter(self, **kwargs):
        '''
        Filter the corpus and selects samples that match the criterias provided
        The arguments to the keyword can be 1) a string, 2) a list of strings, 3)
        a function. There is a match if one of the following is True.

        1. ``value == attribute``
        2. ``value`` is a list and ``attribute in value == True``
        3. ``value`` is a callable (a function) and ``value(attribute) == True``
        '''

        new_corpus = Dataset()

        for s in self.samples:
            new_corpus.add_sample_matching(s, **kwargs)

        return new_corpus
        
    def __getitem__(self, r):
        return self.samples[r]

    def __len__(self):
        return len(self.samples)

    def head(self, n=5):
        ''' Print n samples from the dataset '''
        print('The first', n, '/', len(self.samples), 'samples:')
        for sample in self.samples[:n]:
            print(sample)

    def __str__(self):
        r = 'The dataset contains {} samples.\n'.format(len(self))
        r += 'Metadata attributes are:\n'
        for field, values in self.info.items():

            r += '  {} ({}) :\n'.format(field, len(values))

            # for attributes with lots of values, we just print a few
            if len(values) > 6:
                short_list = _take(6, values.items())
                for value, number in short_list[:3]:
                    r += '      * {} occurs {} times\n'.format(value, number)
                r += '      ...\n'
                for value, number in short_list[3:]:
                    r += '      * {} occurs {} times\n'.format(value, number)

            else:
                for value, number in values.items():
                    r += '      * {} occurs {} times\n'.format(value, number)

        return r[:-1]  # remove trailing '\n'






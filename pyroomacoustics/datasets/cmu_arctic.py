'''
The CMU ARCTIC Dataset
======================

The CMU_ARCTIC databases were constructed at the Language Technologies
Institute at Carnegie Mellon University as phonetically balanced, US English
single speaker databases designed for unit selection speech synthesis research.
A detailed report on the structure and content of the database and the
recording environment etc is available as a Carnegie Mellon University,
Language Technologies Institute Tech Report CMU-LTI-03-177 and is also
available here.

The databases consist of around 1150 utterances carefully selected from
out-of-copyright texts from Project Gutenberg. The databses include US English
male (bdl) and female (slt) speakers (both experinced voice talent) as well as
other accented speakers.

The 1132 sentence prompt list is available from cmuarctic.data

The distributions include 16KHz waveform and simultaneous EGG signals. Full
phoentically labelling was perfromed by the CMU Sphinx using the FestVox based
labelling scripts. Complete runnable Festival Voices are included with the
database distributions, as examples though better voices can be made by
improving labelling etc.

License: Permissive, attribution required

Price: Free

URL: http://www.festvox.org/cmu_arctic/
'''
import os
import numpy as np
from scipy.io import wavfile

try:
    import sounddevice as sd
    have_sounddevice = True
except:
    have_sounddevice = False

from .utils import download_uncompress
from .base import Meta, AudioSample, Dataset

# The speakers codes and attributes
cmu_arctic_speakers = {
        'aew' : { 'sex' : 'male',   'lang' : 'US English', 'accent' : 'US' },
        'ahw' : { 'sex' : 'male',   'lang' : 'US English', 'accent' : 'German' },
        'aup' : { 'sex' : 'male',   'lang' : 'US English', 'accent' : 'Indian' },
        'awb' : { 'sex' : 'male',   'lang' : 'US English', 'accent' : 'Scottish' },
        'axb' : { 'sex' : 'female', 'lang' : 'US English', 'accent' : 'Indian' },
        'bdl' : { 'sex' : 'male',   'lang' : 'US English', 'accent' : 'US' },
        'clb' : { 'sex' : 'female', 'lang' : 'US English', 'accent' : 'US' },
        'eey' : { 'sex' : 'female', 'lang' : 'US English', 'accent' : 'US' },
        'fem' : { 'sex' : 'male',   'lang' : 'US English', 'accent' : 'Irish' },
        'gka' : { 'sex' : 'male',   'lang' : 'US English', 'accent' : 'Indian' },
        'jmk' : { 'sex' : 'male',   'lang' : 'US English', 'accent' : 'Canadian' },
        'ksp' : { 'sex' : 'male',   'lang' : 'US English', 'accent' : 'Indian' },
        'ljm' : { 'sex' : 'female', 'lang' : 'US English', 'accent' : 'US' },
        'lnh' : { 'sex' : 'female', 'lang' : 'US English', 'accent' : 'US' },
        'rms' : { 'sex' : 'male',   'lang' : 'US English', 'accent' : 'US' },
        'rxr' : { 'sex' : 'male',   'lang' : 'US English', 'accent' : 'Dutch' },
        'slp' : { 'sex' : 'female', 'lang' : 'US English', 'accent' : 'Indian' },
        'slt' : { 'sex' : 'female', 'lang' : 'US English', 'accent' : 'US' },
        }

# The sentences in the database
cmu_arctic_sentences = {}

# Directory structure
speaker_dir = 'cmu_us_{}_arctic'

# Download info
url_base = 'http://festvox.org/cmu_arctic/packed/{}.tar.bz2'.format(speaker_dir)

class CMUArcticCorpus(Dataset):
    '''
    This class will load the CMU ARCTIC corpus in a
    structure amenable to be processed.

    Attributes
    ----------
    basedir: str, option
        The directory where the CMU ARCTIC corpus is located/downloaded. By
        default, this is the current directory.
    info: dict
        A dictionary whose keys are the labels of metadata fields attached to the samples.
        The values are lists of all distinct values the field takes.
    sentences: list of CMUArcticSentence
        The list of all utterances in the corpus

    Parameters
    ----------
    basedir: str, optional
        The directory where the CMU ARCTIC corpus is located/downloaded. By
        default, this is the current directory.
    download: bool, optional
        If the corpus does not exist, download it.
    speaker: str or list of str, optional
        A list of the CMU ARCTIC speakers labels. If provided, only
        those speakers are loaded. By default, all speakers are loaded.
    sex: str or list of str, optional
        Can be 'female' or 'male'
    lang: str or list of str, optional
        The language, only 'English' is available here
    accent: str of list of str, optional
        The accent of the speaker
    '''


    def __init__(self, basedir=None, download=False, build=True, **kwargs):

        # initialize
        Dataset.__init__(self)

        # we give a meaningful alias to the sample list from the base class
        self.sentences = self.samples

        # default base directory is the current one
        self.basedir = basedir
        if basedir is None:
            self.basedir = './CMU_ARCTIC'

        # if no speaker is specified, use all the speakers
        if 'speaker' not in kwargs:
            kwargs['speaker'] = list(cmu_arctic_speakers.keys())

        # we need this to know which speakers to download
        speakers = kwargs['speaker']
        if not isinstance(speakers, list):
            speakers = [speakers]

        # check everything requested is there or download
        if not os.path.exists(self.basedir):
            if download:
                os.mkdir(self.basedir)
            else:
                raise ValueError('Corpus directory does not exist. Create or set download option.')

        # remove invalid speaker keys
        n_speakers = len(speakers)
        speakers = [speaker for speaker in speakers if speaker in cmu_arctic_speakers.keys()]
        if n_speakers != len(speakers):
            import warnings
            warnings.warn('Some invalid speakers were removed from the list.', RuntimeWarning)


        # now crawl the speakers directories, download when necessary
        for speaker in speakers:
            sdir = os.path.join(self.basedir, speaker_dir.format(speaker))

            # check the directory exists and download otherwise
            if not os.path.exists(sdir):
                if download:
                    url = url_base.format(speaker)
                    print('Download', url, 'into', self.basedir, '...')
                    download_uncompress(url, path=self.basedir)
                else:
                    raise ValueError('Missing speaker directory. Please download.')

        # now, we populate the sentence data structure containing the list
        # of all distinct sentences spoken in the database
        for speaker in speakers:
            # Now crawl the directory to get all the data
            sdir = os.path.join(self.basedir, speaker_dir.format(speaker))

            all_files = []

            with open(os.path.join(sdir, 'etc/txt.done.data'), 'r') as f:
                for line in f.readlines():
                    # extract the file path
                    tag = line.split(' ')[1]
                    path = os.path.join(sdir, 'wav/' + tag + '.wav')
                    text = line.split('"')[1]

                    # now shorten the tag for internal use
                    tag = tag[-5:]

                    # add the dict of sentences
                    if tag not in cmu_arctic_sentences:
                        cmu_arctic_sentences[tag] = { 
                                'text': text, 
                                'paths' : { speaker : path } 
                                }
                    else:
                        cmu_arctic_sentences[tag]['paths'][speaker] = path

        if build:
            self.build_corpus(**kwargs)

    def build_corpus(self, **kwargs):
        '''
        Build the corpus with some filters (sex, lang, accent, sentence_tag, sentence)
        '''

        # Check all the sentences
        for tag, info in cmu_arctic_sentences.items():

            # And all speakers for each sentence
            for speaker, path in info['paths'].items():

                # This is the metadata for this sample
                meta = Meta(speaker=speaker, tag=tag, text=info['text'], **cmu_arctic_speakers[speaker])

                # it there is a match, add it
                # The reason we do the match before creating the Sentence object is that
                # we don't want to read the file from disk if there is no match
                if meta.match(**kwargs):
                    self.add_sample(CMUArcticSentence(path, **meta.as_dict()))

    def filter(self, **kwargs):
        '''
        Filter the corpus and selects samples that match the criterias provided
        The arguments to the keyword can be 1) a string, 2) a list of strings, 3)
        a function. There is a match if one of the following is True.

        1. ``value == attribute``
        2. ``value`` is a list and ``attribute in value == True``
        3. ``value`` is a callable (a function) and ``value(attribute) == True``
        '''

        # first, create the new empty corpus
        new_corpus = CMUArcticCorpus(basedir=self.basedir, build=False, speaker=[])

        # finally, add all the sentences
        for s in self.samples:
            new_corpus.add_sample_matching(s, **kwargs)

        return new_corpus


class CMUArcticSentence(AudioSample):
    '''
    Create the sentence object

    Parameters
    ----------
    path: str
        the path to the audio file
    **kwargs:
        metadata as a list of keyword arguments

    Attributes
    ----------
    data: array_like
        The actual audio signal
    fs: int
        sampling frequency
    '''

    def __init__(self, path, **kwargs):
        '''
        Create the sentence object

        path: (string)
            the path to the particular sample
        '''

        # Read in the wav file
        fs, data = wavfile.read(path)
        
        # initialize the parent object
        AudioSample.__init__(self, data, fs, **kwargs)


    def __str__(self):
        ''' String representation '''
        template = '{speaker} ({sex}, {lang}/{accent}); {tag}: ''{text}'''
        s = template.format(**self.meta.as_dict())
        return s

    def plot(self, **kwargs):
        ''' Plot the spectrogram '''
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('Warning: matplotlib is required for plotting')
            return
        AudioSample.plot(self, **kwargs)
        plt.title(self.meta.text)


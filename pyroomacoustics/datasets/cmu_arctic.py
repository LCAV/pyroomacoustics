'''
The CMU ARCTIC Dataset
======================

The CMU_ARCTIC databases were constructed at the Language Technologies Institute at Carnegie Mellon University as phonetically balanced, US English single speaker databases designed for unit selection speech synthesis research. A detailed report on the structure and content of the database and the recording environment etc is available as a Carnegie Mellon University, Language Technologies Institute Tech Report CMU-LTI-03-177 and is also available here.

The databases consist of around 1150 utterances carefully selected from out-of-copyright texts from Project Gutenberg. The databses include US English male (bdl) and female (slt) speakers (both experinced voice talent) as well as other accented speakers.

The 1132 sentence prompt list is available from cmuarctic.data

The distributions include 16KHz waveform and simultaneous EGG signals. Full phoentically labelling was perfromed by the CMU Sphinx using the FestVox based labelling scripts. Complete runnable Festival Voices are included with the database distributions, as examples though better voices can be made by improving labelling etc.

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

from .utils import download_uncompress_tar_bz2

# The speakers codes and attributes
cmu_arctic_speakers = {
        'aew' : { 'sex' : 'male', 'lang' : 'English', 'accent' : 'US?' },
        'ahw' : { 'sex' : 'male', 'lang' : 'English', 'accent' : 'US?' },
        'aup' : { 'sex' : 'male', 'lang' : 'English', 'accent' : 'Indian?' },
        'awb' : { 'sex' : 'male', 'lang' : 'English', 'accent' : 'Scottish' },
        'axb' : { 'sex' : 'female', 'lang' : 'English', 'accent' : 'Indian?' },
        'bdl' : { 'sex' : 'male', 'lang' : 'English', 'accent' : 'US' },
        'clb' : { 'sex' : 'female', 'lang' : 'English', 'accent' : 'US' },
        'eey' : { 'sex' : 'female', 'lang' : 'English', 'accent' : 'US?' },
        'fem' : { 'sex' : 'male', 'lang' : 'English', 'accent' : 'US?' },
        'gka' : { 'sex' : 'male', 'lang' : 'English', 'accent' : 'Indian?' },
        'jmk' : { 'sex' : 'male', 'lang' : 'English', 'accent' : 'Canadian' },
        'ksp' : { 'sex' : 'male', 'lang' : 'English', 'accent' : 'Indian' },
        'ljm' : { 'sex' : 'female', 'lang' : 'English', 'accent' : 'US?' },
        'lnh' : { 'sex' : 'female', 'lang' : 'English', 'accent' : 'US?' },
        'rms' : { 'sex' : 'male', 'lang' : 'English', 'accent' : 'US' },
        'rxr' : { 'sex' : 'male', 'lang' : 'English', 'accent' : 'US?' },
        'slp' : { 'sex' : 'female', 'lang' : 'English', 'accent' : 'Indian?' },
        'slt' : { 'sex' : 'female', 'lang' : 'English', 'accent' : 'US' },
        }

# Directory structure
corpus_dir = 'CMU_ARCTIC'
speaker_dir = 'cmu_us_{}_arctic'

# Download info
url_base = 'http://festvox.org/cmu_arctic/packed/{}.tar.bz2'.format(speaker_dir)

for speaker, info in cmu_arctic_speakers.items():
    info['dir'] = os.path.join(corpus_dir, speaker_dir.format(speaker))
    info['url'] = url_base.format(speaker)

class CMUArcticCorpus(object):

    def __init__(self, basedir=None, download=False, speakers=None):

        # default base directory is the current one
        self.basedir = basedir
        if basedir is None:
            self.basedir = '.'

        # by default use all speakers
        self.speakers = speakers
        if self.speakers is None:
            self.speakers = cmu_arctic_speakers.keys()

        # this is where the speakers directories should be
        self.basedir = os.path.join(self.basedir, corpus_dir)
        
        # check everything requested is there or download
        if not os.path.exists(self.basedir):
            if download:
                os.mkdir(self.basedir)
            else:
                raise ValueError('Corpus directory does not exist. Create or set download option.')

        # now crawl the speakers directories
        for speaker in self.speakers:
            sdir = os.path.join(self.basedir, speaker_dir.format(speaker))

            # check the directory exists and download otherwise
            if not os.path.exists(sdir):
                if download:
                    url = cmu_arctic_speakers[speaker]['url']
                    print('Download', url, 'into', self.basedir, '...')
                    download_uncompress_tar_bz2(url, self.basedir)
                else:
                    raise ValueError('Missing speaker directory. Please download.')

        self.sentences = []
        for speaker in self.speakers:
            # Now crawl the directory to get all the data
            sdir = os.path.join(self.basedir, speaker_dir.format(speaker))

            all_files = []

            with open(os.path.join(sdir, 'etc/txt.done.data'), 'r') as f:
                for line in f.readlines():
                    # extract the file path
                    tag = line.split(' ')[1]
                    path = os.path.join(sdir, 'wav/' + tag + '.wav')

                    text = line.split('"')[1]

                    self.sentences.append(CMUArcticSentence(path, tag, text, speaker, cmu_arctic_speakers[speaker]))



class CMUArcticSentence(object):
    '''
    Create the sentence object

    Parameters
    ----------
    path: str
        the path to the audio file
    tag: str
        the sentence tag in the CMU ARCTIC database
    text: str
        The text spoken in the sample
    speaker: str
        The speaker name
    speaker_info: dict
        A dictionary with info such as sex, lang, accent, etc

    Attributes
    ----------
    speaker: str
        Speaker initials
    sex: str
        Speaker gender (M or F)
    dialect: str
        Speaker dialect region number:
    fs: int
        sampling frequency
    samples: array_like (n_samples,)
        the audio track
    tag: str
        The sentence tag
    text: str
        the text of the sentence
    '''

    def __init__(self, path, tag, text, speaker, speaker_info):
        '''
        Create the sentence object

        path: (string)
            the path to the particular sample
        '''

        # extract the attributes
        self.tag = tag
        self.text = text
        self.dialect = speaker_info['lang'] + '/' + speaker_info['accent']
        self.sex = speaker_info['sex']
        self.speaker = speaker

        # Read in the wav file
        self.fs, self.data = wavfile.read(path)
        self.samples = self.data


    def __str__(self):
        s = " ".join([self.dialect, self.sex, self.speaker, self.text])
        return s

    def play(self):
        ''' Play the sound sample '''
        if have_sounddevice:
            sd.play(self.data, samplerate=self.fs, blocking=True)
        else:
            print('Warning: sounddevice package is required to play audiofiles.')

    def plot(self, L=512, hop=128, zpb=0, phonems=False, **kwargs):

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            return

        sns.set_style('white')
        plt.specgram(self.data, NFFT=L, Fs=self.fs, noverlap=L-hop, pad_to=L+zpb)

        plt.title(self.text)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')


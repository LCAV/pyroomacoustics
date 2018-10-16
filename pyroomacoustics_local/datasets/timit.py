'''
The TIMIT Dataset
=================

The TIMIT corpus of read speech is designed to provide speech data for acoustic-phonetic studies and for the development and evaluation of automatic speech recognition systems. TIMIT contains broadband recordings of 630 speakers of eight major dialects of American English, each reading ten phonetically rich sentences. The TIMIT corpus includes time-aligned orthographic, phonetic and word transcriptions as well as a 16-bit, 16kHz speech waveform file for each utterance. Corpus design was a joint effort among the Massachusetts Institute of Technology (MIT), SRI International (SRI) and Texas Instruments, Inc. (TI). The speech was recorded at TI, transcribed at MIT and verified and prepared for CD-ROM production by the National Institute of Standards and Technology (NIST).

The TIMIT corpus transcriptions have been hand verified. Test and training subsets, balanced for phonetic and dialectal coverage, are specified. Tabular computer-searchable information is included as well as written documentation.

Unfortunately, this is a proprietary dataset. A licensed can be obtained for $125 to $250 depending on your
status (academic or otherwise).

**Deprecation Warning:** The interface of TimitCorpus will change in the near
future to match that of :py:obj:`pyroomacoustics.datasets.cmu_arctic.CMUArcticCorpus`

URL: https://catalog.ldc.upenn.edu/ldc93s1
'''

import os
import numpy as np

from pyroomacoustics import stft

try:
    import sounddevice as sd
    have_sounddevice = True
except:
    have_sounddevice = False


def _read_nist_wav(filename):
    '''
    A custom implementation of wav reader that let's read the NIST files from TIMIT

    Decode wav header and create numpy array out of raw data.

    Parameters
    ----------
    filename: str
        the name of the file to decode
    '''
    _byte_format = { 2 : np.int16, 4 : np.int32 }  # helper

    # get raw data
    f = open(filename, 'rb')
    file_data = f.read()
    f.close()

    file_type = file_data[:8].decode()
    if file_type != 'NIST_1A\n':
        raise ValueError('Format might be wrong. We need a NIST_1A file.')

    # separate  header/data
    header_size = int(file_data[8:16].decode())
    header = file_data[16:header_size].decode()
    data = file_data[header_size:]


    # extract all the key/values for all fields of the header
    fields = [line.split() for line in header.split('\n')]

    header_dict = dict()
    for field in fields:
        if len(field) != 3:
            continue

        key = field[0]
        t = field[1][1]

        if t == 'i':
            val = int(field[2])
        elif t == 's':
            val = field[2]
        elif t == 'r':
            val = float(field[2])
        else:
            raise ValueError('Unknown type field for NIST header')

        header_dict[key] = val

    # extract parameters from header
    nch = header_dict['channel_count']
    fs = header_dict['sample_rate']
    bytes_per_sample = header_dict['sample_n_bytes']
    try:
        dt = _byte_format[bytes_per_sample]
    except KeyError:
        raise ValueError("Currently we only support 16 or 32 bits files. Sorry")

    # Build the array
    array = np.frombuffer(data, dtype=_byte_format[bytes_per_sample])
    if nch > 1:
        array = array.reshape((-1,nch))

    return fs, array


class Word:
    '''
    A class used for words of the TIMIT corpus

    Attributes
    ----------
    word: str
        The spelling of the word
    boundaries: list
        The limits of the word within the sentence
    samples: array_like
        A view on the sentence samples containing the word
    fs: int
        The sampling frequency
    phonems: list
        A list of phones contained in the word
    features: array_like
        A feature array (e.g. MFCC coefficients)

    Parameters
    ----------
    word: str
        The spelling of the word
    boundaries: list
        The limits of the word within the sentence
    data: array_like
        The nd-array that contains all the samples of the sentence
    fs: int
        The sampling frequency
    phonems: list, optional
        A list of phones contained in the word

    '''


    def __init__(self, word, boundaries, data, fs, phonems=None):

        self.word = word
        self.phonems = phonems
        self.boundaries = boundaries
        self.samples = data[boundaries[0]:boundaries[1]]
        self.fs = fs
        self.features = None

    def __str__(self):
        return self.word

    def plot(self):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            return

        sns.set_style('white')

        L = self.samples.shape[0]
        plt.plot(np.arange(L)/self.fs, self.samples)
        plt.xlim((0,L/self.fs))
        plt.xlabel('Time')
        plt.title(self.word)

    def play(self):
        ''' Play the sound sample '''
        if have_sounddevice:
            sd.play(self.samples, samplerate=self.fs)
        else:
            print('Warning: sounddevice package is required to play audiofiles.')

    def mfcc(self, frame_length=1024, hop=512):
        ''' compute the mel-frequency cepstrum coefficients of the word samples '''
        self.features = mfcc(self.samples, L=frame_length, hop=hop)


class Sentence:
    '''
    Create the sentence object

    Parameters
    ----------
    path: (string)
        the path to the particular sample

    Attributes
    ----------
    speaker: str
        Speaker initials
    id: str
        a digit to disambiguate identical initials
    sex: str
        Speaker gender (M or F)
    dialect: str
        Speaker dialect region number:

        1. New England
        2. Northern
        3. North Midland
        4. South Midland
        5. Southern
        6. New York City
        7. Western
        8. Army Brat (moved around)
    fs: int
        sampling frequency
    samples: array_like (n_samples,)
        the audio track
    text: str
        the text of the sentence
    words: list
        list of Word objects forming the sentence
    phonems: list
        List of phonems contained in the sentence. Each element is a
        dictionnary containing a 'bnd' with the limits of the phonem and 'name'
        that is the phonem transcription.
    '''

    def __init__(self, path):
        '''
        Create the sentence object

        path: (string)
            the path to the particular sample
        '''

        path, ext = os.path.splitext(path)

        t = path.split('/')

        # extract the attributes
        self.dialect = t[-3]
        self.sex = t[-2][0]
        self.speaker = t[-2][1:5]
        self.id = t[-1]

        # Read in the wav file
        self.fs, self.data = _read_nist_wav(path + '.WAV')
        self.samples = self.data

        # Read the sentence text
        f = open(path + '.TXT', 'r')
        lines = f.readlines()
        self.text = ' '.join(lines[0].split()[2:])
        f.close()

        # Read the word list
        self.words = []
        self.phonems = []

        # open the word file
        f = open(path + '.WRD', 'r')
        w_lines = f.readlines()
        f.close()

        # get all lines from the phonem file
        f_ph = open(path + '.PHN', 'r')
        ph_lines = f_ph.readlines()
        ph_l_index = 0
        f_ph.close()

        for line in w_lines:
            t = line.split()

            # just a sanity check
            if len(t) == 3:

                # the word boundary
                w_bnd = np.array([int(t[0]), int(t[1])])

                # recover the phonems making up the word
                w_ph_list = []
                while ph_l_index < len(ph_lines):

                    ph_line = ph_lines[ph_l_index]
                    u = ph_line.split()

                    # phonem boundary
                    ph_bnd = np.array([int(u[0]), int(u[1])])

                    # Check phonem boundary does not exceeds word boundary
                    if ph_bnd[1] > w_bnd[1]:
                        break

                    # add to sentence object phonems list
                    self.phonems.append({'name':u[2], 'bnd':ph_bnd})

                    # increase index
                    ph_l_index += 1

                    # now skip until beginning of word
                    if ph_bnd[0] < w_bnd[0]:
                        continue

                    # add phonem to word if (with adjusted boundaries wrt to start of word)
                    w_ph_list.append({'name':u[2], 'bnd':ph_bnd - w_bnd[0]})

                # Finally create word object
                self.words.append(Word(t[2], w_bnd, self.data, self.fs, phonems=w_ph_list))

        # Read the remaining phonem(s)
        while ph_l_index < len(ph_lines):
            ph_line = ph_lines[ph_l_index]
            u = ph_line.split()

            if len(u) == 3:
                # phonem boundary
                ph_bnd = (int(u[0]), int(u[1]))

                # add to sentence object phonems list
                self.phonems.append({'name':u[2], 'bnd':ph_bnd})

            ph_l_index += 1


    def __str__(self):
        s = " ".join([self.dialect, self.sex, self.speaker, self.id, self.text])
        return s

    def play(self):
        ''' Play the sound sample '''
        if have_sounddevice:
            sd.play(self.data, samplerate=self.fs)
        else:
            print('Warning: sounddevice package is required to play audiofiles.')

    def plot(self, L=512, hop=128, zpb=0, phonems=False, **kwargs):

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            return

        sns.set_style('white')
        X = stft(self.data, L=L, hop=hop, zp_back=zpb, transform=np.fft.rfft, win=np.hanning(L+zpb))
        X = 10*np.log10(np.abs(X)**2).T

        plt.imshow(X, origin='lower', aspect='auto')

        ticks = []
        ticklabels = []

        if phonems:
            for phonem in self.phonems:
                plt.axvline(x=phonem['bnd'][0]/hop)
                plt.axvline(x=phonem['bnd'][1]/hop)
                ticks.append((phonem['bnd'][1]+phonem['bnd'][0])/2/hop)
                ticklabels.append(phonem['name'])

        else:
            for word in self.words:
                plt.axvline(x=word.boundaries[0]/hop)
                plt.axvline(x=word.boundaries[1]/hop)
                ticks.append((word.boundaries[1]+word.boundaries[0])/2/hop)
                ticklabels.append(word.word)

        plt.xticks(ticks, ticklabels, rotation=-45)
        plt.yticks([],[])
        plt.tick_params(axis='both', which='major', labelsize=14)


class TimitCorpus:
    '''
    TimitCorpus class

    Parameters
    ----------
    basedir : (string)
        The location of the TIMIT database
    directories : (list of strings)
        The subdirectories containing the data (['TEST','TRAIN'])
    sentence_corpus : (dict)
        A dictionnary that contains a list of Sentence objects for each sub-directory
    word_corpus : (dict)
        A dictionnary that contains a list of Words objects for each sub-directory
        and word available in the corpus
    '''

    def __init__(self, basedir):
        ''' Initialize basic attributes of the class '''

        import warnings
        warnings.warn("This interface for TIMIT is deprecated "
                    + "and will be replaced soon to match the "
                    + "Dataset base class and the CMUArcticCorpus "
                    + "interface.", FutureWarning)

        if not os.path.exists(basedir):
            raise ValueError('The directory ''{}'' does not exist.'.format(basedir))

        if not os.path.exists(basedir + '/TEST') or not os.path.exists(basedir + '/TRAIN'):
            raise ValueError('The directory ''{}'' does not contain sub-directories TEST and TRAIN'.format(basedir))

        self.basedir = basedir
        self.directories = ['TEST','TRAIN']
        self.sentence_corpus = None
        self.word_corpus = None


    def build_corpus(self, sentences=None, dialect_region=None, speakers=None, sex=None):
        '''
        Build the corpus

        The TIMIT database structure is encoded in the directory sturcture:

        basedir
            TEST/TRAIN
                Regional accent index (1 to 8)
                    Speakers (one directory per speaker)
                        Sentences (one file per sentence)

        Parameters
        ----------
        sentences: (list)
            A list containing the sentences to which we want to restrict the corpus
            Example: sentences=['SA1','SA2']
        dialect_region: (list of int)
            A list to which we restrict the dialect regions
            Example: dialect_region=[1, 4, 5]
        speakers: (list)
            A list of speakers acronym to which we want to restrict the corpus
            Example: speakers=['AKS0']
        sex: (string)
            Restrict to a single sex: 'F' for female, 'M' for male
        '''
        self.sentence_corpus = dict(zip(self.directories, [[],[]]))
        self.word_corpus = dict(zip(self.directories, [{},{}]))

        if dialect_region is not None:
            dialect_region = ['DR' + str(i) for i in dialect_region]

        # Read in all the sentences making use of TIMIT special directory structure
        for d in self.directories:
            dir1 = os.path.join(self.basedir, d)
            for dialect in next(os.walk(dir1))[1]:
                # check if dialect is in exclusion list
                if dialect_region is not None and dialect not in dialect_region:
                    continue

                dir2 = os.path.join(dir1, dialect)

                for speaker in next(os.walk(dir2))[1]:
                    # check if sex is matching
                    if sex is not None and speaker[0] != sex:
                        continue
                    # check if speaker is not in exclusion list
                    if speakers is not None and speaker not in speakers:
                        continue

                    dir3 = os.path.join(dir2, speaker)

                    for fil in os.listdir(dir3):
                        # just look at wav files to avoid duplicates
                        if fil.endswith('.WAV'):
                            sentence = os.path.splitext(fil)[0]
                            # check if sentence should be excluded
                            if sentences is not None and sentence not in sentences:
                                continue

                            # Create a new sentence object
                            path = os.path.join(dir3, sentence)
                            self.sentence_corpus[d].append(Sentence(path))

                            # Now add the words to the word corpus
                            for w in self.sentence_corpus[d][-1].words:
                                if not w.word in self.word_corpus[d]:
                                    self.word_corpus[d][w.word] = [w]
                                else:
                                    self.word_corpus[d][w.word].append(w)


    def get_word(self, d, w, index=0):
        ''' return instance index of word w from group (test or train) d '''
        return self.word_corpus[d][w][index]


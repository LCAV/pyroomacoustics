
import numpy as np
import os

from scikits.audiolab import Sndfile, play
from scikits.samplerate import resample
class HMM:

    def __init__(self, nstates, odim):
        '''
        Initialize a Hidden Markov Model with nstates and Gaussian observations 
        
        nstates: (int)
            The number of states in the Markov chain
        odim: (int)
            The dimension of the observation variable
        '''

        self.K = nstates    # number of states
        self.O = odim       # dimension of observation vector

        # The Markov chain parameters
        self.A = np.zeros((self.K, self.K)) # the state transition matrix
        self.p = np.zeros((self.K))         # the initial distribution

        # The observation parameters
        self.mu = np.zeros((self.K, self.O))
        self.Sigma = np.zeros((self.K, self.O, self.O))

    def training(self, X):
        '''
        Training of the HMM using the EM algorithm
        '''

    def likelihood(self, X):
        '''
        Compute the likelihood of a sample vector using the sum-product algorithm
        '''
        x = 1

    def forward(self):
        x=1

    def backward(self):
        x=1

    def viterbi(self):
        x=1


class Word:

    def __init__(self, word, boundaries, data, fs):

        self.word = word
        self.samples = data[boundaries[0]:boundaries[1]].copy()
        self.fs = fs

    def play(self):
        play(resample(self.samples, 44100./self.fs, 'sinc_best'))

    def mfcc(self, frame_length, overlap):
        ''' compute the mfcc of the word samples '''


class Sentence:

    def __init__(self, path):
        '''
        Create the sentence object

        path: (string)
            the path to the particular sample
        '''

        path, ext = os.path.splitext(path)

        t = path.split('/')

        # extract the attributes
        self.dialect = t[0]
        self.sex = t[1][0]
        self.speaker = t[1][1:5]
        self.id = t[2]

        # Read in the wav file
        f = Sndfile(path + '.WAV', 'r')
        self.data = f.read_frames(f.nframes)
        self.fs = f.samplerate

        # Read the word list
        self.words = []
        f = open(path + '.WRD', 'r')
        for line in f.readlines():
            t = line.split()
            if len(t) == 3:
                self.words.append(Word(t[2], (int(t[0]), int(t[1])), self.data, self.fs))

    def play(self):
        play(resample(self.data, 44100./self.fs, 'sinc_best'))


class TimitCorpus:

    def __init__(self, basedir):

        self.basedir = basedir
        self.directories = ['TEST','TRAIN']
        self.sentence_corpus = None
        self.word_corpus = None
        self.words = []


    def build_corpus(self, sentences=None, dialect_region=None, speakers=None, sex=None):
        '''
        Build the corpus

        Arguments
        ---------
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
                                if not self.word_corpus[d].has_key(w.word):
                                    self.word_corpus[d][w.word] = [w]
                                else:
                                    self.word_corpus[d][w.word].append(w)


    def get_word(self, d, w, index=0):
        return self.word_corpus[d][w][index]




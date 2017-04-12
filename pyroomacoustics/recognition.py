
from __future__ import division, print_function

import numpy as np
import os
from scipy.stats import multivariate_normal

try:
    from scikits.audiolab import Sndfile, play
    have_sk_audiolab = True
except ImportError:
    have_sk_audiolab = False

try:
    from scikits.samplerate import resample
    have_sk_samplerate = True
except ImportError:
    have_sk_samplerate = False

from .stft import stft
from .acoustics import mfcc

class CircularGaussianEmission:

    def __init__(self, nstates, odim=1, examples=None):
        ''' Initialize the Gaussian emission object '''

        # The emissions parameters
        self.K = nstates

        if examples is None:
            
            # Initialize to random components
            self.O = odim
            self.mu = np.random.normal(size=(self.K, self.O))
            self.Sigma = np.ones((self.K, self.O))*10

        else:
            # Initialize all components to the same mean and variance of the data
            self.O = examples[0].shape[1]

            X = np.concatenate(examples, axis=0)

            self.mu = np.array([np.mean(X, axis=0)]*self.K)
            centered = X - self.mu[0]
            self.Sigma = np.array([np.mean(centered**2, axis=0)]*self.K)

    def update_parameters(self, examples, gamma):

        g = np.concatenate(gamma, axis=0)
        X = np.concatenate(examples, axis=0)
        Z = g.sum(axis=0)

        for k in range(self.K):
            self.mu[k] = np.sum(X.T * g[:,k], axis=1)/Z[k]
            centered = (X - self.mu[k])**2
            self.Sigma[k] = np.sum(centered.T * g[:,k], axis=1)/Z[k]


    def get_pdfs(self):
        ''' Return the pdf of all the emission probabilities '''
        return [multivariate_normal(self.mu[k], np.diag(self.Sigma[k])) for k in range(self.K)]


    def prob_x_given_state(self, examples):
        ''' 
        Recompute the probability of the observation given the state of the
        latent variables
        '''
        distribution = [multivariate_normal(self.mu[k], np.diag(self.Sigma[k])) for k in range(self.K)]
        p_x_given_z = []

        for X in examples:
            p_x_given_z.append(np.zeros((X.shape[0], self.K)))
            for k in range(self.K):
                p_x_given_z[-1][:,k] = distribution[k].pdf(X)

        return p_x_given_z


class GaussianEmission:

    def __init__(self, nstates, odim=1, examples=None):
        ''' Initialize the Gaussian emission object '''

        # The emissions parameters
        self.K = nstates

        if examples is None:
            # initialize to random mean unit variance
            self.O = odim
            self.mu = np.random.normal(size=(self.K, self.O))
            self.Sigma = np.random.normal(size=(self.K, self.O, self.O))
            for k in range(self.K):
                self.Sigma[k] = np.dot(self.Sigma[k].T, self.Sigma[k]) + np.eye(self.O)

        else:
            # Initialize using mean and covariance of dataset
            self.O = examples[0].shape[1]

            X = np.concatenate(examples, axis=0)

            self.mu = np.array([np.mean(X, axis=0)]*self.K)
            centered = X - self.mu[0]
            self.Sigma = np.array([np.diag(np.mean(centered**2, axis=0))]*self.K)

    def update_parameters(self, examples, gamma):

        g = np.concatenate(gamma, axis=0)
        X = np.concatenate(examples, axis=0)
        Z = g.sum(axis=0)

        for k in range(self.K):
            self.mu[k] = np.sum(X.T * g[:,k], axis=1)/Z[k]
            centered = X - self.mu[k]
            self.Sigma[k] = np.dot(centered.T*g[:,k], centered/Z[k])


    def get_pdfs(self):
        ''' Return the pdf of all the emission probabilities '''
        return [multivariate_normal(self.mu[k], self.Sigma[k]) for k in range(self.K)]


    def prob_x_given_state(self, examples):
        ''' 
        Recompute the probability of the observation given the state of the
        latent variables
        '''
        distribution = [ multivariate_normal(self.mu[k], self.Sigma[k]) for k in range(self.K)]
        p_x_given_z = []

        for X in examples:
            p_x_given_z.append(np.zeros((X.shape[0], self.K)))
            for k in range(self.K):
                p_x_given_z[-1][:,k] = distribution[k].pdf(X)

        return p_x_given_z


class HMM:
    '''
    Hidden Markov Model with Gaussian emissions

    Attributes
    ----------
    K : int
        Number of states in the model
    O : int
        Number of dimensions of the Gaussian emission distribution
    A : ndarray
        KxK transition matrix of the Markov chain
    pi : ndarray
        K dim vector of the initial probabilities of the Markov chain
    emission : (GaussianEmission or CircularGaussianEmission)
        An instance of emission_class
    model : string, optional
        The model used for the chain, can be 'full' or 'left-right'
    leftright_jum_max : int, optional
        The number of non-zero upper diagonals in a 'left-right' model
    '''

    def __init__(self, nstates, emission, model='full', leftright_jump_max=3):
        '''
        Initialize a Hidden Markov Model with nstates and Gaussian observations 
        
        nstates: int
            The number of states in the Markov chain
        emission : emission object, optional
            The emission object (CircularGaussianEmission or GaussianEmission)
        model : string, optional
            The model used for the chain, can be 'full' or 'left-right'
        leftright_jump_max : int
            The maximum jump length in the Left-Right chain model
        '''

        self.K = nstates            # number of states
        self.emission = emission    # The observation parameters

        # The Markov chain parameters
        self.model = model

        self.leftright_jump_max = leftright_jump_max
        self.A = np.zeros((self.K, self.K)) # the state transition matrix
        self.pi = np.zeros((self.K))        # the initial distribution

        # Initialize the HMM parameters to some random values
        if self.model == 'full':
            self.A = np.random.uniform(size=(self.K,self.K))
            self.pi = np.random.uniform(size=(self.K))

        elif self.model == 'left-right':
            self.A = np.triu(np.tril(np.random.uniform(size=(self.K,self.K)), k=self.leftright_jump_max))
            self.A += np.diag(np.sum(self.A[:,:], axis=1)*2)
            self.pi = np.zeros(self.K)
            self.pi[0] = 1

        # Normalize the distributions
        for row in self.A:
            row /= row.sum()
        self.pi /= self.pi.sum()

    def fit(self, examples, tol=0.1, max_iter=10, verbose=False):
        '''
        Training of the HMM using the EM algorithm

        Parameters
        ----------
        examples : (list)
            A list of examples used to train the model. Each example is
            an array of feature vectors, each row is a feature vector,
            the sequence runs on axis 0
        tol : (float)
            The training stops when the progress between to steps is less than
            this number (default 0.1)
        max_iter : (int)
            Alternatively the algorithm stops when a maximum number of
            iterations is reached (default 10)
        verbose : bool, optional
            When True, prints extra information about convergence
        '''

        # Make sure to normalize parameters that should be...
        for row in self.A:
            row[:] /= row.sum()
        self.pi[:] /= self.pi.sum()

        # Run the EM algorithm
        loglikelihood_old = -np.inf # log-likelihood
        n_iter = 0
        while True:

            # Initialize new parameters value for accumulation
            loglikelihood = 0.

            # We need to run the forward/backward algorithm for each example and
            # and combine the result to form the new estimates
            gamma = []
            xhi = []
            p_x_given_z = self.emission.prob_x_given_state(examples)

            # Expectation-step
            #-----------------

            for X,pxz in zip(examples, p_x_given_z):

                # check dimension of emission
                if X.shape[1] != self.emission.O:
                    raise ValueError("Error: Emission vectors of all examples should have the same size")

                # First compute alpha and beta using forward/backward algo
                alpha, c = self.forward(X, pxz)
                beta = self.backward(X, pxz, c)

                # Recompute the likelihood of the sequence
                # (Bishop 13.63)
                loglikelihood += np.sum(np.log(c))

                # Now the more interesting quantities
                # gamma(z_n) = p(z_n | X, theta_old) 
                # xhi(z_{n-1}, z_n) = p(z_{n-1}, z_n | X, theta_old)
                gamma.append(alpha * beta)
                xhi.append(np.zeros((X.shape[0]-1, self.K, self.K)))
                for n in range(1,X.shape[0]):
                    xhi[-1][n-1] = np.outer(alpha[n-1], beta[n]*pxz[n])*self.A/c[n]

            # Maximization-step
            #------------------

            # update the Markov Chain parameters
            self.update_parameters(examples, gamma, xhi)

            # Update the emission distribution parameters
            self.emission.update_parameters(examples, gamma)

            # Now check for convergence
            #--------------------------
            n_iter += 1
            epsilon = loglikelihood - loglikelihood_old
            if verbose:
                print('Iterations:', n_iter, 'epsilon:', epsilon, 'LL_new:', loglikelihood)

            # some checks here
            if epsilon < tol:
                if verbose:
                    print('Tolerance reached: stopping.')
                break
            if  n_iter == max_iter:
                if verbose:
                    print('Maximum iterations reached: stopping.')
                break

            loglikelihood_old = loglikelihood

        # return the number of iterations performed
        return n_iter


    def update_parameters(self, examples, gamma, xhi):
        ''' Update the parameters of the Markov Chain '''

        X = np.concatenate(examples, axis=0)
        x = np.concatenate(xhi, axis=0)

        self.pi[:] = np.sum([g[0,:] for g in gamma], axis=0)
        self.A = x.sum(axis=0)

        # normalize to enforce distribution constraints
        self.pi /= np.sum(self.pi)
        for k in range(self.K):
            den = np.sum(self.A[k,:])
            if den < 1e-15:
                self.A[k,:] = 0.
            else:
                self.A[k,:] /= den


    def generate(self, N):
        ''' Generate a random sample of length N using the model '''
        X = np.zeros((N, self.emission.O))
        distributions = self.emission.get_pdfs()

        # pick initial state
        state = np.random.choice(self.K, p=self.pi)

        # now run the chain
        for n in range(0,N):
            # produce emission vector according to current state
            X[n,:] = distributions[state].rvs()
            # pick next state
            state = np.random.choice(self.K, p=self.A[state,:])

        return X


    def loglikelihood(self, X):
        '''
        Compute the log-likelihood of a sample vector using the sum-product algorithm
        '''
        p_x_given_z = self.emission.prob_x_given_state([X])[0]
        alpha, c = self.forward(X, p_x_given_z)

        return np.sum(np.log(c))

    def forward(self, X, p_x_given_z):
        ''' The forward recursion for HMM as described in Bishop Ch. 13 '''

        # initialize the alpha vector
        alpha = np.zeros((X.shape[0], self.K))
        c = np.zeros(X.shape[0])

        # initialize the recursion as
        # p(X | z_k) pi_k
        alpha[0] = p_x_given_z[0]*self.pi
        c[0] = np.sum(alpha[0])
        alpha[0] /= c[0]

        # Run the forward recursion
        for n in range(1,X.shape[0]):
            alpha[n] = p_x_given_z[n]*np.dot(self.A.T, alpha[n-1])
            c[n] = np.sum(alpha[n])
            alpha[n] /= c[n]

        return alpha, c

    def backward(self, X, p_x_given_z, c):
        ''' The backward recursion for HMM as described in Bishop Ch. 13 '''

        # intialize the beta vectors
        beta = np.zeros((X.shape[0], self.K))

        # initialize the recursion
        beta[-1,:] = 1

        # Run the backward recursion
        for n in range(X.shape[0]-2,-1,-1):
            beta[n] = np.dot(self.A, p_x_given_z[n+1]*beta[n+1])/c[n+1]

        return beta


    def viterbi(self):
        x=1


class Word:

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
        if have_sk_audiolab and have_sk_samplerate:
            play(np.array(resample(self.samples, 44100./self.fs, 'sinc_best'), dtype=np.float64))
        else:
            print('Warning: scikits.audiolab and scikits.samplerate are required to play audiofiles.')

    def mfcc(self, frame_length=1024, hop=512):
        ''' compute the mel-frequency cepstrum coefficients of the word samples '''
        self.features = mfcc(self.samples, L=frame_length, hop=hop)


class Sentence:

    def __init__(self, path):
        '''
        Create the sentence object

        path: (string)
            the path to the particular sample
        '''

        if not have_sk_audiolab:
            raise ValueError('scikits.audiolab module is required to read the TIMIT database.')

        path, ext = os.path.splitext(path)

        t = path.split('/')

        # extract the attributes
        self.dialect = t[-3]
        self.sex = t[-2][0]
        self.speaker = t[-2][1:5]
        self.id = t[-1]

        # Read in the wav file
        f = Sndfile(path + '.WAV', 'r')
        self.data = f.read_frames(f.nframes)
        self.fs = f.samplerate

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
        if have_sk_audiolab and have_sk_samplerate:
            play(np.array(resample(self.data, 44100./self.fs, 'sinc_best'), dtype=np.float64))
        else:
            print('Warning: scikits.audiolab and scikits.samplerate are required to play audiofiles.')

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

    Attributes
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
                                if not self.word_corpus[d].has_key(w.word):
                                    self.word_corpus[d][w.word] = [w]
                                else:
                                    self.word_corpus[d][w.word].append(w)


    def get_word(self, d, w, index=0):
        ''' return instance index of word w from group (test or train) d '''
        return self.word_corpus[d][w][index]


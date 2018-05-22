
from __future__ import division, print_function

import numpy as np
import os
from scipy.stats import multivariate_normal
import sys
import struct

try:
    import sounddevice as sd
    have_sounddevice = True
except:
    have_sounddevice = False

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


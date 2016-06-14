import numpy as np
from pyroomacoustics import HMM, CircularGaussianEmission, GaussianEmission

def init_func(A, pi, emission, examples):
    '''
    Initializes all the parameters for an HMM with circular Gaussian emissions

    All the parameters are references and should not be thrown away

    The function do not return anything

    Parameters
    ----------
    A : reference to ndarray
        The transition matrix of the Markov chain
    pi : reference to ndarray
        The initial state probability of the Markov chain
    emission : Emission object
        The emission object that describes the distribution of the observation
        It has an attributes
            - mu : K by O ndarray that contains the means
            - Sigma : K by O ndarray that contains the diagonals
                of the covariance matrices
    examples : list of ndarray
        A list that contains all the sequences to train the HMM
    '''

    K = A.shape[0]
    O = examples[0].shape[1]

    # concatenate all examples
    X = np.concatenate(examples, axis=0)

    # Initialize for a left right model
    pi[:] = 0
    pi[0] = 1
    A[:,:] = np.triu(np.tril(np.random.uniform(size=(K,K)), k=K))
    A[:,:] += np.diag(np.sum(A[:,:], axis=1)*2)
    # Normalize the distributions
    for row in A:
        row[:] /= row.sum()
    pi[:] /= pi.sum()

    # Initialize the emission parameters
    emission.mu[:,:] = np.array([np.mean(X, axis=0)]*K)
    centered = X - emission.mu[0]
    emission.Sigma[:,:] = np.array([np.mean(centered**2, axis=0)]*K)


if __name__ == '__main__':

    K = 4
    O = 6
    emission_class = CircularGaussianEmission
    model = 'left-right'
    leftright_jump_max = K
    n_examples = 200
    example_size = np.arange(40,60)

    hmm = HMM(K, emission_class=emission_class, model=model,
              leftright_jump_max=leftright_jump_max)
    hmm.initialize(O)

    # Draw a 100 examples
    examples = []
    for i in range(n_examples):
        N = np.random.choice(example_size)
        examples.append(hmm.generate(N))

    # Now create a new model and train it
    hmm2 = HMM(K, emission_class=emission_class, model=model)
    hmm2.fit(examples, tol=1e-8, max_iter=1000, init_func=init_func)

    for k in range(K):
        print 'True mu:',hmm.emission.mu[k],'Estimated:',hmm2.emission.mu[k]

    print 'hmm A:',hmm.A
    print 'hmm2 A:',hmm2.A


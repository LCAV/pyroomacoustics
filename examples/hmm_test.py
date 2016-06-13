import numpy as np
from pyroomacoustics import HMM, CircularGaussianEmission, GaussianEmission

if __name__ == '__main__':

    K = 4
    O = 3
    emission_class = GaussianEmission
    n_examples = 100
    example_size = np.arange(40,60)

    hmm = HMM(K, emission_class=emission_class)
    hmm.initialize(O)

    # Draw a 100 examples
    examples = []
    for i in range(n_examples):
        N = np.random.choice(example_size)
        examples.append(hmm.generate(N))

    # Now create a new model and train it
    hmm2 = HMM(K, emission_class=emission_class)
    hmm2.fit(examples, tol=1e-8, max_iter=1000)

    for k in range(K):
        print 'True mu:',hmm.emission.mu[k],'Estimated:',hmm2.emission.mu[k]

    print 'hmm A:',hmm.A
    print 'hmm2 A:',hmm2.A


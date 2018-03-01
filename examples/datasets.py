'''
Example of the basic operations with ``pyroomacoustics.datasets.CorpusBase``
and ``pyroomacoustics.datasets.SampleBase`` classes
'''
from pyroomacoustics.datasets import SampleBase, CorpusBase

# Prepare a few artificial samples
samples = [
    {
        'data' : 0.99,
        'metadata' : { 'speaker' : 'alice', 'sex' : 'female', 'age' : 37, 'number' : 'one' },
    },
    {
        'data' : 2.1,
        'metadata' : { 'speaker' : 'alice', 'sex' : 'female', 'age' : 37, 'number' : 'two' },
    },
    {
        'data' : 1.02,
        'metadata' : { 'speaker' : 'bob', 'sex' : 'male', 'age' : 48, 'number' : 'one' },
    },
    {
        'data' : 2.07,
        'metadata' : { 'speaker' : 'bob', 'sex' : 'male', 'age' : 48, 'number' : 'two' },
    },
    {
        'data' : 3.0,
        'metadata' : { 'speaker' : 'bob', 'sex' : 'male', 'age' : 48, 'number' : 'three' },
    },
    {
        'data' : 1.97,
        'metadata' : { 'speaker' : 'charles', 'sex' : 'male', 'age' : 25, 'number' : 'two' },
    },
    ]

corpus = CorpusBase()
for s in samples:
    new_sample = SampleBase(s['data'], **s['metadata'])
    corpus.add_sample(new_sample)

# Then, it possible to display summary info about the corpus
print(corpus)
print()

# The number of samples in the corpus is given by ``len``
print('Number of samples:', len(corpus))
print()

# And we can access samples with the slice operator
print('Sample #2:')
print(corpus[2])
print()

# We can obtain a new corpus with only male subject
corpus_male_only = corpus.filter(sex='male')
print(corpus_male_only)

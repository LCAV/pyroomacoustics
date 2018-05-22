'''
TIMIT Corpus
============

Pyroomacoustics includes a wrapper around the popular TIMIT corpus [1]
that can be handy in some situations. Here is a quick rundown of how
to use it.

Since the TIMIT corpus is not open, this example can only be run if
you have access to it. Sorry for that.

To successfully run the example, you will need to set an environment variable
`TIMIT_PATH` to point to the location of your local instance of the TIMIT corpus.

[1] Garofolo, John S., et al. TIMIT Acoustic-Phonetic Continuous Speech Corpus LDC93S1. Web Download. Philadelphia: Linguistic Data Consortium, 1993. https://catalog.ldc.upenn.edu/ldc93s1
'''

import pyroomacoustics as pra
import matplotlib.pyplot as plt
import os

# add an environment variable with the TIMIT location
# e.g. /path/to/timit/TIMIT
try:
    timit_path = os.environ['TIMIT_PATH']
except:
    raise ValueError('An environment variable ''TIMIT_PATH'' pointing to the TIMIT base location is needed.')

# Load the corpus, be patient
corpus = pra.datasets.TimitCorpus(timit_path)
corpus.build_corpus()

# the corpus is split between train and test
# let's pick a sentence from each
sentence1 = corpus.sentence_corpus['TEST'][0]
sentence2 = corpus.sentence_corpus['TRAIN'][14]

# let's find all the sentences from male speakers in the training set
test_male = list(filter(lambda x: x.sex == 'M', corpus.sentence_corpus['TEST']))

# show the spectrogram
plt.figure()
test_male[13].plot()

# uncomment next line to play the sentence
# test_male.play()

# Each sentence has a list of words
words = test_male[27].words
plt.figure()
words[3].plot()  # plot time domain
# words[3].play()  # uncomment to play

# It is possible to query the corpus for all instances of a given word
example_word = 'she'
if example_word in corpus.word_corpus['TRAIN'].keys():
    print('The word ''{}'' is in the corpus'.format(example_word))
    training_set = corpus.word_corpus['TRAIN'][example_word]
    test_set = corpus.word_corpus['TEST'][example_word]
    print('Number of training examples for {}:'.format(example_word), len(training_set))
    print('Number of test examples for {}:'.format(example_word), len(test_set))

plt.show()

'''
The Datasets Sub-package is responsible to deliver
wrappers around a few popular audio datasets to make
them easier to use.

Two base class :py:obj:`pyroomacoustics.datasets.base.Dataset` and
:py:obj:`pyroomacoustics.datasets.base.Sample` wrap
together the audio samples and their meta data.
The general idea is to create a sample object with an attribute
containing all metadata. Dataset objects that have a collection
of samples can then be created and can be filtered according
to the values in the metadata.

Many of the functions with ``match`` or ``filter`` will take an
arbitrary number of keyword arguments. The keys should match some
metadata in the samples. Then there are three ways that match occurs
between a ``key/value`` pair and an ``attribute`` sharing the same key.

1. ``value == attribute``
2. ``value`` is a list and ``attribute in value == True``
3. ``value`` is a callable (a function) and ``value(attribute) == True``

Example 1
---------

::

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
        ]

    corpus = Dataset()
    for s in samples:
        new_sample = Sample(s['data'], **s['metadata'])
        corpus.add_sample(new_sample)

    # Then, it possible to display summary info about the corpus
    print(corpus)

    # The number of samples in the corpus is given by ``len``
    print('Number of samples:', len(corpus))

    # And we can access samples with the slice operator
    print('Sample #2:')
    print(corpus[2])    # (shortcut for `corpus.samples[2]`)

    # We can obtain a new corpus with only male subject
    corpus_male_only = corpus.filter(sex='male')
    print(corpus_male_only)

    # Only retain speakers above 40 years old
    corpus_older = corpus.filter(age=lambda a : a > 40)
    print(corpus_older)

Example 2 (CMU ARCTIC)
----------------------

::

    # This example involves the CMU ARCTIC corpus available at
    # http://www.festvox.org/cmu_arctic/

    import matplotlib.pyplot as plt
    import pyroomacoustics as pra

    # Here, the corpus for speaker bdl is automatically downloaded
    # if it is not available already
    corpus = pra.datasets.CMUArcticCorpus(download=True, speaker=['bdl'])

    # print dataset info and 10 sentences
    print(corpus)
    corpus.head(n=10)

    # let's extract all samples containing the word 'what'
    keyword = 'what'
    matches = corpus.filter(text=lambda t : keyword in t)
    print('The number of sentences containing "{}": {}'.format(keyword, len(matches)))
    for s in matches.sentences:
        print('  *', s)

    # if the sounddevice package is available, we can play the sample
    matches[0].play()

    # show the spectrogram
    plt.figure()
    matches[0].plot()
    plt.show()

Example 3 (Google's Speech Commands Dataset)
--------------------------------------------

::

    # This example involves Google's Speech Commands Dataset available at
    # https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html

    import matplotlib.pyplot as plt
    import pyroomacoustics as pra

    # The dataset is automatically downloaded if not available and 10 of each word is selected
    dataset = pra.datasets.GoogleSpeechCommands(download=True, subset=10, seed=0)

    # print dataset info, first 10 entries, and all sounds
    print(dataset)
    dataset.head(n=10)
    print("All sounds in the dataset:")
    print(dataset.classes)

    # filter by specific word
    selected_word = 'yes'
    matches = dataset.filter(word=selected_word)
    print("Number of '%s' samples : %d" % (selected_word, len(matches)))

    # if the sounddevice package is available, we can play the sample
    matches[0].play()

    # show the spectrogram
    plt.figure()
    matches[0].plot()
    plt.show()
  
'''

from .base import Meta, Sample, AudioSample, Dataset
from .timit import Word, Sentence, TimitCorpus
from .cmu_arctic import CMUArcticCorpus, CMUArcticSentence, cmu_arctic_speakers
from .google_speech_commands import GoogleSpeechCommands, GoogleSample

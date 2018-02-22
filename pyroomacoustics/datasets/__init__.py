'''
The Datasets Sub-package is responsible to deliver
wrappers around a few popular audio datasets to make
them easier to use.
'''

from .timit import Word, Sentence, TimitCorpus
from .cmu_arctic import CMUArcticCorpus, CMUArcticSentence, cmu_arctic_speakers
